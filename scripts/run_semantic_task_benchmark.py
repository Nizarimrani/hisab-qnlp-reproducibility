#!/usr/bin/env python3
"""Synthetic structured semantic microtask: exact Qiskit circuit vs classical baselines.

This task validates one explicit operator rule and its factuality probability
encoding. It is not free-text NLP and cannot establish quantum advantage.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SEED = 20261028
FACTUALITY_PROBABILITIES = (0.20, 0.50, 0.80)
ASPECT_NAMES = ("not_started", "ongoing", "completed")
STATUS_NAMES = ("inactive", *ASPECT_NAMES)
N_QUBITS = 11
PROBABILITY_TOLERANCE = 1e-9


def probability_to_ry_angle(probability: float) -> float:
    """Prepare a qubit whose probability of measuring 1 is the given value."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    import math
    return 2.0 * math.asin(math.sqrt(probability))


def symbolic_oracle(row: dict[str, Any]) -> str:
    """Return the licensed semantic state, independently of its confidence.

    Negation toggles the underlying proposition truth. If the resulting
    premise is true, the consequent aspect is licensed; factuality determines
    the activation probability rather than imposing a hard threshold.
    Concession does not cancel the consequent.
    """
    effective_premise = bool(row["premise_truth"]) ^ bool(row["negated"])
    if not effective_premise:
        return "inactive"
    return ASPECT_NAMES[int(row["aspect_stage"])]


def symbolic_distribution(row: dict[str, Any]) -> dict[str, float]:
    """Specify the output-status distribution for the structured semantics."""
    distribution = {label: 0.0 for label in STATUS_NAMES}
    effective_premise = bool(row["premise_truth"]) ^ bool(row["negated"])
    if not effective_premise:
        distribution["inactive"] = 1.0
        return distribution
    activation_probability = float(row["factuality_probability"])
    distribution["inactive"] = 1.0 - activation_probability
    distribution[ASPECT_NAMES[int(row["aspect_stage"])]] = activation_probability
    return distribution


def make_sentence(row: dict[str, Any]) -> str:
    """Produce an illustrative Spanish realization of a structured input."""
    premise = "el sol no sale" if row["negated"] else "el sol sale"
    connector = "Aunque" if row["concession"] else "Si"
    factuality = {
        0: "es poco probable que",
        1: "es posible que",
        2: "es muy probable (80%) que",
    }[int(row["factuality_level"])]
    event = {
        0: "el tomate empiece a enrojecer",
        1: "el tomate esté enrojeciendo",
        2: "el tomate haya enrojecido",
    }[int(row["aspect_stage"])]
    return f"{connector} {premise}, {factuality} {event}."


def make_dataset() -> list[dict[str, Any]]:
    """Exhaustive factorial table: 2×2×3×2×3 = 72 structured cases."""
    rows: list[dict[str, Any]] = []
    combinations = product((0, 1), (0, 1), range(3), (0, 1), range(3))
    for row_id, (premise_truth, negated, fact_level, concession, aspect) in enumerate(combinations):
        row: dict[str, Any] = {
            "example_id": f"sem-{row_id:03d}",
            "premise_truth": premise_truth,
            "negated": negated,
            "factuality_level": fact_level,
            "factuality_probability": FACTUALITY_PROBABILITIES[fact_level],
            "concession": concession,
            "aspect_stage": aspect,
        }
        row["target_label"] = symbolic_oracle(row)
        distribution = symbolic_distribution(row)
        row["target_activation_probability"] = 1.0 - distribution["inactive"]
        row["expected_inactive_probability"] = distribution["inactive"]
        row["expected_stage_probability"] = row["target_activation_probability"]
        row["sentence_es"] = make_sentence(row)
        rows.append(row)
    return rows


def build_task_circuit(row: dict[str, Any]) -> QuantumCircuit:
    """Encode one structured case into an 11-qubit reversible operator circuit.

    q0 premise truth; q1 negation; q2 factuality amplitude; q3 concession;
    q4..q6 one-hot aspect; q7 effective premise; q8 active inference;
    q9..q10 output status bits (00 inactive, 01 onset, 10 ongoing, 11 done).
    """
    qc = QuantumCircuit(N_QUBITS, name=str(row.get("example_id", "case")))
    premise, negation, factuality, concession = 0, 1, 2, 3
    stage0, stage1, stage2 = 4, 5, 6
    effective, active, out0, out1 = 7, 8, 9, 10

    if int(row["premise_truth"]):
        qc.x(premise)
    if int(row["negated"]):
        qc.x(negation)
    qc.ry(probability_to_ry_angle(float(row["factuality_probability"])), factuality)
    if int(row["concession"]):
        qc.x(concession)
    stage = int(row["aspect_stage"])
    qc.x((stage0, stage1, stage2)[stage])

    # Effective premise truth = underlying truth XOR negation.
    qc.cx(premise, effective)
    qc.cx(negation, effective)
    # The factuality qubit has P(1)=p. Activation is conditional on premise.
    qc.ccx(effective, factuality, active)
    # Encode an active stage as 01=onset, 10=ongoing, 11=completed.
    qc.ccx(active, stage0, out0)
    qc.ccx(active, stage1, out1)
    qc.ccx(active, stage2, out0)
    qc.ccx(active, stage2, out1)
    return qc


def run_circuit(row: dict[str, Any]) -> dict[str, float]:
    """Return exact output-status probabilities from the statevector."""
    state = Statevector.from_instruction(build_task_circuit(row))
    basis_probabilities = state.probabilities()
    distribution = {label: 0.0 for label in STATUS_NAMES}
    for basis_index, probability in enumerate(basis_probabilities):
        output_code = ((basis_index >> 9) & 1) + 2 * ((basis_index >> 10) & 1)
        distribution[STATUS_NAMES[output_code]] += float(probability)
    return distribution


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_classical_baselines(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train scikit-learn baselines on one frozen stratified split."""
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Classical ML baselines require the benchmark optional dependency. "
            "Install the project with its benchmark extra."
        ) from exc

    features = [
        [
            int(r["premise_truth"]),
            int(r["negated"]),
            float(r["factuality_probability"]),
            int(r["concession"]),
            int(r["aspect_stage"]),
        ]
        for r in rows
    ]
    labels = [str(r["target_label"]) for r in rows]
    indices = list(range(len(rows)))
    train_ids, test_ids = train_test_split(
        indices, test_size=0.25, random_state=SEED, stratify=labels
    )
    train_set, test_set = set(train_ids), set(test_ids)
    for index, row in enumerate(rows):
        row["split"] = "train" if index in train_set else "test"

    models = [
        ("majority_dummy", DummyClassifier(strategy="most_frequent")),
        ("logistic_regression", LogisticRegression(max_iter=2000, random_state=SEED)),
        ("decision_tree", DecisionTreeClassifier(random_state=SEED)),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=SEED, n_jobs=1
            ),
        ),
    ]
    records: list[dict[str, Any]] = []
    predictions_by_model: dict[str, Any] = {}
    y_test = [labels[i] for i in test_ids]
    x_train = [features[i] for i in train_ids]
    x_test = [features[i] for i in test_ids]

    for name, model in models:
        train_start = time.perf_counter()
        model.fit(x_train, [labels[i] for i in train_ids])
        train_seconds = time.perf_counter() - train_start
        predict_start = time.perf_counter()
        predicted = list(model.predict(x_test))
        predict_seconds = time.perf_counter() - predict_start
        records.append({
            "model": name,
            "test_examples": len(test_ids),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
        })
        predictions_by_model[name] = [
            {
                "example_id": rows[idx]["example_id"],
                "gold": labels[idx],
                "predicted": str(pred),
            }
            for idx, pred in zip(test_ids, predicted)
        ]

    split_meta = {
        "seed": SEED,
        "train_examples": len(train_ids),
        "test_examples": len(test_ids),
        "stratified": True,
        "note": "One fixed split on a small synthetic structured dataset; not a generalization estimate for natural language.",
    }
    return records, {"split": split_meta, "predictions": predictions_by_model}


def run_benchmark(output_dir: Path) -> dict[str, Any]:
    rows = make_dataset()

    oracle_start = time.perf_counter()
    oracle_distributions = [symbolic_distribution(row) for row in rows]
    oracle_seconds = time.perf_counter() - oracle_start

    circuit_start = time.perf_counter()
    circuit_distributions = [run_circuit(row) for row in rows]
    circuit_seconds = time.perf_counter() - circuit_start

    errors = [
        abs(actual[label] - expected[label])
        for actual, expected in zip(circuit_distributions, oracle_distributions)
        for label in STATUS_NAMES
    ]
    mismatches = []
    for row, actual, expected in zip(rows, circuit_distributions, oracle_distributions):
        max_error = max(abs(actual[label] - expected[label]) for label in STATUS_NAMES)
        if max_error > PROBABILITY_TOLERANCE:
            mismatches.append({
                "example_id": row["example_id"],
                "expected": expected,
                "actual": actual,
                "max_abs_error": max_error,
            })

    labels = [r["target_label"] for r in rows]
    label_distribution = {name: labels.count(name) for name in STATUS_NAMES}
    classical_records, classical_details = run_classical_baselines(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "semantic_task_dataset.csv", rows)

    try:
        sklearn_version = version("scikit-learn")
    except PackageNotFoundError:
        sklearn_version = "not-installed"

    report = {
        "benchmark": "Hisab-inspired structured semantic microtask v0.2",
        "benchmark_type": "synthetic exhaustive rule-validation; not free-text NLP",
        "seed": SEED,
        "dataset": {
            "examples": len(rows),
            "factorial_design": "2 premise-truth values × 2 polarity values × 3 factuality probabilities × 2 concession values × 3 aspect stages",
            "label_distribution": label_distribution,
            "input_features": [
                "premise_truth", "negated", "factuality_probability",
                "concession", "aspect_stage"
            ],
            "labels": list(STATUS_NAMES),
            "factuality_probabilities": {"0": 0.20, "1": 0.50, "2": 0.80},
            "target_rule": (
                "Effective premise = premise_truth XOR negated. If effective premise is false, "
                "the output is inactive with probability 1. If true, the selected aspect is "
                "activated with probability p and the output is inactive with probability 1-p. "
                "Concession does not cancel the branch."
            ),
            "important_limit": (
                "Spanish sentences are illustrative realizations. Features are supplied gold "
                "annotations; this experiment does not parse text."
            ),
        },
        "semantic_reference": {
            "implementation": "direct symbolic oracle for output probability distribution",
            "seconds_for_72_examples": oracle_seconds,
            "is_independent_of_quantum_circuit": True,
        },
        "qiskit_exact_circuit": {
            "simulator": "qiskit.quantum_info.Statevector",
            "examples": len(rows),
            "qubits": N_QUBITS,
            "seconds_for_72_examples": circuit_seconds,
            "mean_absolute_output_probability_error": sum(errors) / len(errors),
            "max_absolute_output_probability_error": max(errors, default=0.0),
            "mismatches_over_tolerance": mismatches,
            "hardware_execution": False,
            "full_statevector_exact": True,
        },
        "classical_ml_baselines": classical_records,
        "split": classical_details["split"],
        "test_predictions": classical_details["predictions"],
        "software": {
            "python": platform.python_version(),
            "qiskit": version("qiskit"),
            "scikit_learn": sklearn_version,
        },
        "interpretation": {
            "correctness_check_passed": len(mismatches) == 0,
            "computational_advantage_demonstrated": False,
            "reason": (
                "The task is a small synthetic truth table. The Qiskit circuit ran on a "
                "classical statevector simulator, and the symbolic rule is already known. "
                "Runtime comparisons are not quantum-versus-classical hardware comparisons."
            ),
            "next_required_evidence": (
                "Independently annotated natural-language examples, predeclared task metrics, "
                "stronger classical baselines, repeated splits, and end-to-end cost accounting."
            ),
        },
    }
    (output_dir / "semantic_task_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Structured semantic microtask v0.2",
        "",
        "**Scope:** exhaustive synthetic rule-validation, not free-text NLP and not a quantum-advantage test.",
        "",
        f"- Examples: {len(rows)}",
        f"- Label counts: {json.dumps(label_distribution, sort_keys=True)}",
        f"- Qubits in encoded circuit: {N_QUBITS}",
        f"- Max error in output probabilities: {max(errors, default=0.0):.3e}",
        f"- Statevector simulator runtime for all examples: {circuit_seconds:.6f} s",
        f"- Symbolic reference runtime for all examples: {oracle_seconds:.6f} s",
        "",
        "## Classical machine-learning baselines",
        "",
        "| Model | Test accuracy | Test macro-F1 | Fit seconds | Predict seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in classical_records:
        lines.append(
            f"| {item['model']} | {item['accuracy']:.3f} | {item['macro_f1']:.3f} "
            f"| {item['train_seconds']:.6f} | {item['predict_seconds']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The circuit's output distribution is compared for every input against an independent symbolic probability rule.",
        "The supervised models predict the licensed semantic state on one deterministic stratified split. The corpus is small and synthetic.",
        "",
        "**No quantum computational advantage is demonstrated.** The circuit and its statevector simulation run on a classical CPU; the symbolic rule is an exact classical reference. Do not interpret simulator wall times as quantum speedup.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/semantic_task"))
    args = parser.parse_args()
    report = run_benchmark(args.output_dir)
    print(json.dumps({
        "benchmark": report["benchmark"],
        "examples": report["dataset"]["examples"],
        "max_probability_error": report["qiskit_exact_circuit"]["max_absolute_output_probability_error"],
        "mismatches": len(report["qiskit_exact_circuit"]["mismatches_over_tolerance"]),
        "computational_advantage_demonstrated": report["interpretation"]["computational_advantage_demonstrated"],
        "output_dir": str(args.output_dir),
    }, indent=2))
    if report["qiskit_exact_circuit"]["mismatches_over_tolerance"]:
        raise SystemExit("Circuit disagrees with symbolic probability rule; inspect report.")


if __name__ == "__main__":
    main()
