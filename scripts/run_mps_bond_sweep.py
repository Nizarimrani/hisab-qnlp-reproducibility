#!/usr/bin/env python3
"""Timeout-safe matched statevector/MPS comparison with durable checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import qiskit
import qiskit_aer
from qiskit import transpile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mps_comparison import (  # noqa: E402
    gate_only_circuit,
    observable_circuit,
    run_method,
)

BASIS = ["ry", "rz", "cx", "h"]
FIELDS = [
    "status", "checkpoint_stage", "n_qubits", "topology", "logical_layers", "seed",
    "mps_max_bond_dimension", "mps_truncation_threshold", "mps_timeout_seconds",
    "transpile_seconds", "statevector_seconds", "mps_seconds", "mps_wall_seconds",
    "mps_speedup_vs_statevector", "observable_count",
    "mean_abs_observable_error", "max_abs_observable_error", "error",
]


def prepare_circuit(n_qubits: int, seed: int):
    """Construct and transpile the common circuit deterministically."""
    source = gate_only_circuit(n_qubits, 1, "all_to_all", seed)
    start = time.perf_counter()
    common = transpile(
        source, basis_gates=BASIS, optimization_level=0, seed_transpiler=seed
    )
    transpile_seconds = time.perf_counter() - start
    circuit, labels = observable_circuit(common)
    return circuit, labels, transpile_seconds


def write_checkpoint(output_dir: Path, row: dict[str, Any],
                     reference: dict[str, float] | None = None) -> None:
    """Persist every reached stage; never wait for MPS to finish before writing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    checkpoint = {
        "benchmark": "Timeout-safe MPS bond-dimension sensitivity",
        "checkpoint_stage": row.get("checkpoint_stage"),
        "status": row.get("status"),
        "row": row,
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "python": platform.python_version(),
        "synthetic_circuit_only": True,
        "hardware_execution": False,
        "interpretation_limits": [
            "Statevector and MPS are both classical simulators.",
            "Errors are computed only over local Z and selected ZZ expectations, not full-state fidelity.",
            "A timeout means MPS did not return within the configured budget; it does not provide an approximation-error estimate.",
            "No linguistic validity, classical intractability, or quantum advantage is established.",
        ],
    }
    (output_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
    )
    if reference is not None:
        (output_dir / "statevector_expectations.json").write_text(
            json.dumps({
                "source": "Qiskit Aer statevector reference",
                "n_observables": len(reference),
                "expectations": reference,
            }, indent=2) + "\n", encoding="utf-8"
        )
    report = [
        "# MPS bond-dimension sensitivity run", "",
        f"- Status: {row.get('status')}",
        f"- Checkpoint stage: {row.get('checkpoint_stage')}",
        f"- Qubits/topology/layers: {row.get('n_qubits')} / all-to-all / 1",
        f"- Seed: {row.get('seed')}",
        f"- MPS maximum bond dimension: {row.get('mps_max_bond_dimension')}",
        f"- MPS timeout budget (s): {row.get('mps_timeout_seconds')}",
        f"- Statevector runtime (s): {row.get('statevector_seconds')}",
        f"- MPS runtime (s; blank if not completed): {row.get('mps_seconds')}",
        f"- MPS wall time (s): {row.get('mps_wall_seconds')}",
        f"- MPS speedup (statevector/MPS): {row.get('mps_speedup_vs_statevector')}",
        f"- Mean absolute observable error: {row.get('mean_abs_observable_error')}",
        f"- Maximum absolute observable error: {row.get('max_abs_observable_error')}",
        f"- Error/detail: {row.get('error')}", "",
        "The reference observables are checkpointed before MPS starts. A timed-out MPS stage "
        "leaves these reference results available, but does not imply an MPS error estimate.", ""
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_mps_worker(config_path: Path) -> None:
    """Child process: run MPS only and save result JSON immediately on completion."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(config["worker_output"])
    try:
        circuit, labels, _ = prepare_circuit(config["n_qubits"], config["seed"])
        expectations, mps_seconds = run_method(
            circuit,
            "matrix_product_state",
            config["seed"],
            config["bond_dimension"],
            config["truncation_threshold"],
        )
        if not set(labels).issubset(expectations):
            raise RuntimeError("MPS output does not contain every expected observable")
        payload = {"status": "ok", "expectations": expectations, "mps_seconds": mps_seconds}
    except Exception as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000]}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] != "ok":
        raise SystemExit(1)


def run_one(
    n_qubits: int,
    seed: int,
    bond_dimension: int,
    truncation_threshold: float = 1e-12,
    output_dir: Path | None = None,
    mps_timeout_seconds: float = 2400,
) -> dict:
    """Compute a statevector reference, persist it, then run MPS in a capped subprocess."""
    if mps_timeout_seconds <= 0:
        raise ValueError("mps_timeout_seconds must be positive")
    row: dict[str, Any] = {key: "" for key in FIELDS}
    row.update(
        status="starting",
        checkpoint_stage="initial",
        n_qubits=n_qubits,
        topology="all_to_all",
        logical_layers=1,
        seed=seed,
        mps_max_bond_dimension=bond_dimension,
        mps_truncation_threshold=truncation_threshold,
        mps_timeout_seconds=mps_timeout_seconds,
    )
    reference = None

    def persist() -> None:
        if output_dir is not None:
            write_checkpoint(output_dir, row, reference)

    try:
        circuit, labels, transpile_seconds = prepare_circuit(n_qubits, seed)
        row["transpile_seconds"] = transpile_seconds
        row["status"] = "circuit_ready"
        row["checkpoint_stage"] = "circuit_transpiled"
        persist()

        reference, statevector_seconds = run_method(
            circuit, "statevector", seed, bond_dimension, truncation_threshold
        )
        if not set(labels).issubset(reference):
            raise RuntimeError("Statevector output does not contain every expected observable")
        row.update(
            status="statevector_complete",
            checkpoint_stage="statevector_reference_saved",
            statevector_seconds=statevector_seconds,
            observable_count=len(labels),
        )
        persist()

        if output_dir is None:
            import tempfile
            temp_context = tempfile.TemporaryDirectory(prefix="mps-bond-checkpoint-")
            working_dir = Path(temp_context.name)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            working_dir = output_dir
            temp_context = None

        config_path = working_dir / "mps_worker_input.json"
        worker_output = working_dir / "mps_worker_output.json"
        config_path.write_text(json.dumps({
            "n_qubits": n_qubits,
            "seed": seed,
            "bond_dimension": bond_dimension,
            "truncation_threshold": truncation_threshold,
            "worker_output": str(worker_output),
        }, indent=2) + "\n", encoding="utf-8")

        row["status"] = "mps_running"
        row["checkpoint_stage"] = "mps_subprocess_started"
        persist()
        mps_start = time.perf_counter()
        try:
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--mps-worker", str(config_path)],
                capture_output=True,
                text=True,
                timeout=mps_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            row.update(
                status="mps_timeout",
                checkpoint_stage="mps_timeout_reference_preserved",
                mps_wall_seconds=time.perf_counter() - mps_start,
                error=f"MPS subprocess exceeded configured budget of {mps_timeout_seconds:g} seconds; statevector checkpoint preserved.",
            )
            persist()
            if temp_context is not None:
                temp_context.cleanup()
            return row

        row["mps_wall_seconds"] = time.perf_counter() - mps_start
        if not worker_output.exists():
            detail = (child.stderr or child.stdout or f"worker exit code {child.returncode}").strip()
            raise RuntimeError(f"MPS worker produced no output: {detail[-700:]}")
        payload = json.loads(worker_output.read_text(encoding="utf-8"))
        if payload.get("status") != "ok" or child.returncode != 0:
            raise RuntimeError(payload.get("error") or f"MPS worker exit code {child.returncode}")

        approximation = payload["expectations"]
        if set(reference) != set(approximation) or not set(labels).issubset(reference):
            raise RuntimeError("Observable labels differ between simulation methods")
        errors = [abs(float(reference[label]) - float(approximation[label])) for label in labels]
        mps_seconds = float(payload["mps_seconds"])
        row.update(
            status="ok",
            checkpoint_stage="mps_complete",
            mps_seconds=mps_seconds,
            mps_speedup_vs_statevector=(
                float(statevector_seconds) / mps_seconds if mps_seconds > 0 else None
            ),
            mean_abs_observable_error=float(sum(errors) / len(errors)),
            max_abs_observable_error=float(max(errors, default=0.0)),
            error="",
        )
        persist()
        if temp_context is not None:
            temp_context.cleanup()
        return row
    except Exception as exc:
        row.update(status="error", checkpoint_stage="error", error=f"{type(exc).__name__}: {exc}"[:1000])
        persist()
        return row


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--mps-worker":
        run_mps_worker(Path(sys.argv[2]))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, required=True, choices=(24, 26))
    parser.add_argument("--seed", type=int, required=True, choices=(11, 29, 47))
    parser.add_argument("--bond-dimension", type=int, required=True, choices=(128, 256, 512))
    parser.add_argument("--truncation-threshold", type=float, default=1e-12)
    parser.add_argument("--mps-timeout-seconds", type=float, default=2400,
                        help="Maximum time allowed for the MPS subprocess; checkpoint is retained on timeout.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mps_bond_sweep"))
    args = parser.parse_args()
    if args.n_qubits == 26 and args.bond_dimension == 512:
        parser.error("26-qubit MPS at bond dimension 512 is not in the resource-bounded sweep")

    row = run_one(
        args.n_qubits, args.seed, args.bond_dimension, args.truncation_threshold,
        output_dir=args.output_dir, mps_timeout_seconds=args.mps_timeout_seconds,
    )
    print(json.dumps(row, indent=2), flush=True)
    # A bounded timeout is a valid partial result with a durable reference checkpoint.
    if row["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
