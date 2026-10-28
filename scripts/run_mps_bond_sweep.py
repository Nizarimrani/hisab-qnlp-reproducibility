#!/usr/bin/env python3
"""One matched all-to-all circuit comparison at a chosen MPS bond-dimension cap."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import transpile

from scripts.run_mps_comparison import (
    gate_only_circuit,
    observable_circuit,
    run_method,
)

BASIS = ["ry", "rz", "cx", "h"]
FIELDS = [
    "status", "n_qubits", "topology", "logical_layers", "seed",
    "mps_max_bond_dimension", "mps_truncation_threshold",
    "transpile_seconds", "statevector_seconds", "mps_seconds",
    "mps_speedup_vs_statevector", "observable_count",
    "mean_abs_observable_error", "max_abs_observable_error", "error",
]


def run_one(n_qubits: int, seed: int, bond_dimension: int,
            truncation_threshold: float = 1e-12) -> dict:
    """Reuse one common circuit and statevector reference for one MPS cap."""
    row = {key: "" for key in FIELDS}
    row.update(
        status="ok", n_qubits=n_qubits, topology="all_to_all", logical_layers=1,
        seed=seed, mps_max_bond_dimension=bond_dimension,
        mps_truncation_threshold=truncation_threshold,
    )
    try:
        source = gate_only_circuit(n_qubits, 1, "all_to_all", seed)
        t0 = time.perf_counter()
        common = transpile(
            source, basis_gates=BASIS, optimization_level=0, seed_transpiler=seed
        )
        row["transpile_seconds"] = time.perf_counter() - t0
        circuit, labels = observable_circuit(common)

        reference, statevector_seconds = run_method(
            circuit, "statevector", seed, bond_dimension, truncation_threshold
        )
        approximation, mps_seconds = run_method(
            circuit, "matrix_product_state", seed, bond_dimension,
            truncation_threshold,
        )
        if set(reference) != set(approximation) or not set(labels).issubset(reference):
            raise RuntimeError("Observable labels differ between simulation methods")
        errors = [abs(reference[label] - approximation[label]) for label in labels]
        row.update(
            statevector_seconds=statevector_seconds,
            mps_seconds=mps_seconds,
            mps_speedup_vs_statevector=(
                statevector_seconds / mps_seconds if mps_seconds > 0 else None
            ),
            observable_count=len(labels),
            mean_abs_observable_error=float(sum(errors) / len(errors)),
            max_abs_observable_error=float(max(errors, default=0.0)),
        )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, required=True, choices=(24, 26))
    parser.add_argument("--seed", type=int, required=True, choices=(11, 29, 47))
    parser.add_argument("--bond-dimension", type=int, required=True, choices=(128, 256, 512))
    parser.add_argument("--truncation-threshold", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mps_bond_sweep"))
    args = parser.parse_args()
    if args.n_qubits == 26 and args.bond_dimension == 512:
        parser.error("26-qubit MPS at bond dimension 512 is not in the initial resource-bounded sweep")

    row = run_one(args.n_qubits, args.seed, args.bond_dimension,
                  args.truncation_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "result.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    summary = {
        "benchmark": "MPS bond-dimension sensitivity, dense all-to-all synthetic circuit",
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "python": platform.python_version(),
        "n_qubits": args.n_qubits,
        "topology": "all_to_all",
        "logical_layers": 1,
        "seed": args.seed,
        "mps_max_bond_dimension": args.bond_dimension,
        "mps_truncation_threshold": args.truncation_threshold,
        "row": row,
        "synthetic_circuit_only": True,
        "hardware_execution": False,
        "interpretation_limits": [
            "Both methods are classical simulators; statevector is the reference for selected Pauli expectations",
            "Agreement on selected local Z and ZZ expectations is not full-state fidelity",
            "One deterministic seed is a diagnostic replicate, not a statistical conclusion",
            "Runtime depends on the hosted runner and software version",
            "No linguistic validity, classical intractability, or quantum advantage is established",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# MPS bond-dimension sensitivity run", "",
        f"- Status: {row['status']}",
        f"- Qubits/topology/layers: {args.n_qubits} / all-to-all / 1",
        f"- Seed: {args.seed}",
        f"- MPS maximum bond dimension: {args.bond_dimension}",
        f"- Truncation threshold: {args.truncation_threshold:g}",
        f"- Statevector runtime: {row['statevector_seconds']}",
        f"- MPS runtime: {row['mps_seconds']}",
        f"- MPS speedup (statevector/MPS): {row['mps_speedup_vs_statevector']}",
        f"- Mean absolute observable error: {row['mean_abs_observable_error']}",
        f"- Maximum absolute observable error: {row['max_abs_observable_error']}",
        f"- Error: {row['error']}", "",
        "The run compares selected Pauli expectations on an identical transpiled circuit. "
        "It is a simulator sensitivity measurement, not a quantum-advantage test.", ""
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if row["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
