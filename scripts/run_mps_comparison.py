#!/usr/bin/env python3
"""Matched statevector vs matrix-product-state (MPS) benchmark.

Synthetic circuits only. Both methods evaluate the same deterministic circuit and
same small set of Pauli observables; the MPS is bond-dimension capped, so accuracy
is measured against the statevector baseline rather than assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_scaling_aer import build_circuit  # noqa: E402

FIELDS = [
    "status", "n_qubits", "topology", "logical_layers", "seed",
    "mps_max_bond_dimension", "mps_truncation_threshold",
    "transpile_seconds", "statevector_seconds", "mps_seconds",
    "mps_speedup_vs_statevector", "observable_count",
    "mean_abs_observable_error", "max_abs_observable_error",
    "error",
]
BASIS = ["ry", "rz", "cry", "h"]


def gate_only_circuit(n: int, layers: int, topology: str, seed: int) -> QuantumCircuit:
    """Create the benchmark circuit without Aer-only save instructions."""
    source = build_circuit(n, layers, topology, seed)
    result = QuantumCircuit(n)
    for item in source.data:
        if item.operation.name.startswith("save_"):
            continue
        indices = [source.find_bit(q).index for q in item.qubits]
        result.append(item.operation, [result.qubits[i] for i in indices])
    return result


def observable_circuit(gates: QuantumCircuit) -> tuple[QuantumCircuit, list[str]]:
    """Add local Z observables and selected ZZ correlations without full-state export."""
    circuit = gates.copy()
    labels: list[str] = []
    n = circuit.num_qubits
    for q in range(n):
        label = f"z_{q}"
        circuit.save_expectation_value(Pauli("Z"), [q], label=label)
        labels.append(label)
    pairs = {(q, q + 1) for q in range(n - 1)}
    pairs.add((0, n - 1))
    for a, b in sorted(pairs):
        label = f"zz_{a}_{b}"
        circuit.save_expectation_value(Pauli("ZZ"), [a, b], label=label)
        labels.append(label)
    return circuit, labels


def run_method(circuit: QuantumCircuit, method: str, seed: int,
               max_bond_dimension: int, truncation_threshold: float) -> tuple[dict[str, float], float]:
    options = {
        "method": method,
        "precision": "double",
        "max_parallel_threads": 2,
    }
    if method == "matrix_product_state":
        options.update({
            "matrix_product_state_max_bond_dimension": max_bond_dimension,
            "matrix_product_state_truncation_threshold": truncation_threshold,
            "mps_log_data": True,
        })
    backend = AerSimulator(**options)
    start = time.perf_counter()
    result = backend.run(circuit, shots=1, seed_simulator=seed).result()
    elapsed = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(result.status)
    data = result.data(0)
    # Aer may return numerically complex values for real Hermitian expectations.
    expectations = {key: float(np.real(data[key])) for key in data
                    if key.startswith("z_") or key.startswith("zz_")}
    return expectations, elapsed


def configurations(min_qubits: int, max_qubits: int):
    # Same topology/layer/seed combinations as the scaling frontier wherever possible.
    for n in (20, 22, 24, 26):
        if n < min_qubits or n > max_qubits:
            continue
        for topology in ("local", "nearest"):
            for layers in (1, 2, 4):
                for seed in (11, 29, 47):
                    yield n, topology, layers, seed
        for seed in (11, 29, 47):
            yield n, "all_to_all", 1, seed


def evaluate(n: int, topology: str, layers: int, seed: int,
             max_bond_dimension: int, truncation_threshold: float) -> dict:
    row = {key: "" for key in FIELDS}
    row.update(status="ok", n_qubits=n, topology=topology, logical_layers=layers, seed=seed,
               mps_max_bond_dimension=max_bond_dimension,
               mps_truncation_threshold=truncation_threshold)
    try:
        source = gate_only_circuit(n, layers, topology, seed)
        t0 = time.perf_counter()
        common = transpile(source, basis_gates=BASIS, optimization_level=0,
                           seed_transpiler=seed)
        row["transpile_seconds"] = time.perf_counter() - t0
        circuit, labels = observable_circuit(common)

        sv, sv_seconds = run_method(circuit, "statevector", seed,
                                    max_bond_dimension, truncation_threshold)
        mps, mps_seconds = run_method(circuit, "matrix_product_state", seed,
                                      max_bond_dimension, truncation_threshold)
        if set(sv) != set(mps) or not set(labels).issubset(sv):
            raise RuntimeError("Observable labels differ between simulation methods")
        errors = [abs(sv[label] - mps[label]) for label in labels]
        row.update(
            statevector_seconds=sv_seconds,
            mps_seconds=mps_seconds,
            mps_speedup_vs_statevector=(sv_seconds / mps_seconds if mps_seconds > 0 else None),
            observable_count=len(labels),
            mean_abs_observable_error=float(mean(errors)),
            max_abs_observable_error=float(max(errors, default=0.0)),
        )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-qubits", type=int, default=20)
    parser.add_argument("--max-qubits", type=int, default=26)
    parser.add_argument("--max-bond-dimension", type=int, default=128)
    parser.add_argument("--truncation-threshold", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tensor_network"))
    args = parser.parse_args()
    if args.min_qubits not in (20, 22, 24, 26) or args.max_qubits not in (20, 22, 24, 26):
        parser.error("min/max qubits must be one of 20, 22, 24, 26")
    if args.min_qubits > args.max_qubits:
        parser.error("--min-qubits must be <= --max-qubits")
    if args.max_bond_dimension < 1 or args.truncation_threshold < 0:
        parser.error("bond dimension must be positive and truncation threshold nonnegative")

    configs = list(configurations(args.min_qubits, args.max_qubits))
    rows: list[dict] = []
    started = time.perf_counter()
    for i, (n, topology, layers, seed) in enumerate(configs, 1):
        row = evaluate(n, topology, layers, seed, args.max_bond_dimension,
                       args.truncation_threshold)
        rows.append(row)
        write_csv(args.output_dir / "comparison_results.csv", rows)
        print(json.dumps({
            "completed": i, "total": len(configs), **{
                k: row.get(k) for k in (
                    "status", "n_qubits", "topology", "logical_layers", "seed",
                    "statevector_seconds", "mps_seconds", "max_abs_observable_error", "error"
                )
            }
        }, sort_keys=True), flush=True)

    successful = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    groups: dict[tuple, list[dict]] = {}
    for row in successful:
        groups.setdefault((row["n_qubits"], row["topology"], row["logical_layers"]), []).append(row)
    aggregate = []
    for (n, topology, layers), group in sorted(groups.items()):
        item = {"n_qubits": n, "topology": topology, "logical_layers": layers,
                "replicates": len(group)}
        for metric in ("statevector_seconds", "mps_seconds", "mps_speedup_vs_statevector",
                       "mean_abs_observable_error", "max_abs_observable_error"):
            values = [float(r[metric]) for r in group if r.get(metric) not in ("", None)]
            item[metric + "_mean"] = mean(values) if values else None
            item[metric + "_sd"] = pstdev(values) if values else None
        aggregate.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregated.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    summary = {
        "benchmark": "Matched Qiskit Aer statevector vs matrix-product-state comparison",
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "circuits_requested": len(configs),
        "circuits_succeeded": len(successful),
        "circuits_failed": len(failed),
        "wall_time_seconds": elapsed,
        "mps_max_bond_dimension": args.max_bond_dimension,
        "mps_truncation_threshold": args.truncation_threshold,
        "observable_definition": "all single-qubit Z expectations, all adjacent ZZ expectations, and ZZ(0,n-1)",
        "comparison_basis": "identical transpiled circuit and identical observables; statevector reference",
        "synthetic_circuits_only": True,
        "hardware_execution": False,
        "limitations": [
            "MPS is a classical tensor-network simulator, not quantum hardware",
            "bond-dimension cap and truncation threshold can introduce approximation error",
            "selected Pauli expectations do not establish full-state fidelity or task-level accuracy",
            "synthetic circuits do not establish linguistic validity, classical intractability, or quantum advantage",
            "wall-clock results are specific to this GitHub Actions runner and software stack"
        ],
        "failed_runs": [
            {"n_qubits": r["n_qubits"], "topology": r["topology"],
             "layers": r["logical_layers"], "seed": r["seed"], "error": r["error"]}
            for r in failed
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Statevector vs matrix-product-state comparison", "",
        f"- Qiskit: {qiskit.__version__}; Aer: {qiskit_aer.__version__}",
        f"- Runs: {len(successful)} successful / {len(configs)} requested",
        f"- Wall-clock time: {elapsed:.1f} s",
        f"- MPS bond-dimension cap: {args.max_bond_dimension}; truncation threshold: {args.truncation_threshold:g}",
        "- Both simulators receive the identical transpiled circuit and compute the same local Pauli expectations.",
        "- MPS speedup is defined as statevector runtime divided by MPS runtime; values above 1 mean MPS was faster.",
        "- Error columns compare selected observables against statevector, not the full quantum state.", "",
        "## Results by configuration", "",
        "| Qubits | Topology | Layers | n | Statevector mean (s) | MPS mean (s) | MPS speedup | Max observable error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"
    ]
    for item in aggregate:
        report.append(
            f"| {item['n_qubits']} | {item['topology']} | {item['logical_layers']} | {item['replicates']} "
            f"| {item['statevector_seconds_mean']:.4f} | {item['mps_seconds_mean']:.4f} "
            f"| {item['mps_speedup_vs_statevector_mean']:.3f} | {item['max_abs_observable_error_mean']:.3e} |"
        )
    if failed:
        report.extend(["", "## Failed configurations", ""])
        report.extend([f"- {r['n_qubits']}q {r['topology']} L{r['layers']} seed {r['seed']}: {r['error']}"
                       for r in failed])
    report.extend(["", "## Interpretation", "",
        "This is a controlled comparison of two classical simulation methods for one synthetic circuit family. "
        "MPS can benefit when entanglement across its chosen ordering remains modest; dense all-to-all interactions "
        "can increase the required bond dimension. Faster MPS timings or small errors on the selected observables "
        "do not prove that every classical tensor-network algorithm will scale similarly, nor establish quantum advantage.",
        ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
