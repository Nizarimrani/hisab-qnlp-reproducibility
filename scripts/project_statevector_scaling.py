#!/usr/bin/env python3
"""Fit and extrapolate the statevector cost of the dense Hisab circuit family.

This projects exact classical statevector simulation only. It is not a lower
bound on every classical algorithm and does not claim quantum speedup.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def linear_regression(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Need at least two paired observations")
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    if sxx == 0:
        raise ValueError("Regression x values must not all be equal")
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sxx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * value for value in x]
    ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, fitted))
    ss_tot = sum((value - y_mean) ** 2 for value in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "intercept": intercept,
        "slope": slope,
        "r_squared_log_space": r_squared,
    }


def load_dense_layer_measurements(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[int, list[float]] = {}
    diagnostics: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            if row["status"] != "ok":
                continue
            if row["topology"] != "all_to_all" or int(row["logical_layers"]) != 1:
                continue
            n = int(row["n_qubits"])
            seconds = float(row["elapsed_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if n < 16 or not math.isfinite(seconds) or seconds <= 0:
            continue
        groups.setdefault(n, []).append(seconds)
        try:
            rank_text = row.get("central_cut_schmidt_rank_numeric", "")
            entropy_text = row.get("central_cut_entropy_bits", "")
            if rank_text not in ("", None) and entropy_text not in ("", None):
                diagnostics.setdefault(n, []).append((float(rank_text), float(entropy_text)))
        except (TypeError, ValueError):
            pass

    measurements = []
    for n, times in sorted(groups.items()):
        ds = diagnostics.get(n, [])
        measurements.append({
            "n_qubits": n,
            "replicates": len(times),
            "runtime_seconds_median": statistics.median(times),
            "runtime_seconds_mean": statistics.mean(times),
            "runtime_seconds_min": min(times),
            "runtime_seconds_max": max(times),
            "minimum_statevector_memory_bytes": 16 * (1 << n),
            "logical_cry_gates_per_dense_layer": n * (n - 1) // 2,
            "approx_basis_cx_count_per_dense_layer": n * (n - 1),
            "numerical_schmidt_rank": int(statistics.median(x[0] for x in ds)) if ds else None,
            "central_cut_entropy_bits": statistics.mean(x[1] for x in ds) if ds else None,
        })
    return measurements


def project(input_csv: Path, output_dir: Path,
            projection_qubits: list[int] | None = None) -> dict[str, Any]:
    projection_qubits = projection_qubits or [28, 30, 32, 34, 36, 40, 50]
    if any(n < 2 or n % 2 for n in projection_qubits):
        raise ValueError("Projection sizes must be even integers >= 2")
    measurements = load_dense_layer_measurements(input_csv)
    fit_points = [item for item in measurements if item["n_qubits"] >= 20]
    if len({item["n_qubits"] for item in fit_points}) < 4:
        raise ValueError(
            "Need measurements at four or more distinct sizes >=20 qubits "
            "for an empirical extrapolation"
        )

    fit = linear_regression(
        [float(item["n_qubits"]) for item in fit_points],
        [math.log(float(item["runtime_seconds_median"])) for item in fit_points],
    )
    empirical_per_qubit_factor = math.exp(fit["slope"])
    anchor = max(fit_points, key=lambda item: item["n_qubits"])
    anchor_n = int(anchor["n_qubits"])
    anchor_seconds = float(anchor["runtime_seconds_median"])
    projections: list[dict[str, Any]] = []

    for n in sorted(set(projection_qubits)):
        empirical_seconds = math.exp(fit["intercept"] + fit["slope"] * n)
        structural_seconds = anchor_seconds * ((n / anchor_n) ** 2) * (2.0 ** (n - anchor_n))
        raw_bytes = 16 * (1 << n)
        projections.append({
            "n_qubits": n,
            "measured": False,
            "empirical_log_linear_runtime_seconds": empirical_seconds,
            "statevector_work_model_n2_2n_seconds": structural_seconds,
            "minimum_complex128_statevector_bytes": raw_bytes,
            "minimum_complex128_statevector_gib": raw_bytes / (1024 ** 3),
            "statevector_amplitudes": 1 << n,
            "full_output_distribution_entries": 1 << n,
            "logical_cry_gates_per_dense_layer": n * (n - 1) // 2,
            "approx_basis_cx_count_per_dense_layer": n * (n - 1),
            "exact_mps_central_bond_dimension_upper_bound": 1 << (n // 2),
        })

    observed_growth = []
    by_n = {item["n_qubits"]: item for item in measurements}
    for left, right in zip(sorted(by_n), sorted(by_n)[1:]):
        if left >= 20 and right - left == 2:
            observed_growth.append({
                "from_qubits": left,
                "to_qubits": right,
                "runtime_ratio_median": by_n[right]["runtime_seconds_median"] / by_n[left]["runtime_seconds_median"],
                "raw_statevector_memory_ratio": 2 ** (right - left),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark": "Empirical and structural projection of dense Hisab statevector simulation",
        "input_csv": str(input_csv),
        "circuit_family": "existing RY/RZ/CRY signed-rotation synthetic all-to-all family, one logical layer",
        "backend_scope": "exact classical statevector simulation",
        "observed_measurements": measurements,
        "empirical_fit": {
            "model": "log(runtime_seconds) = intercept + slope * n_qubits",
            **fit,
            "per_added_qubit_runtime_factor": empirical_per_qubit_factor,
            "fit_sizes_qubits": [item["n_qubits"] for item in fit_points],
            "fit_anchor_qubits": anchor_n,
            "interpretation": "Finite-range fit only; not a universal runtime law and not an optimal-classical-algorithm lower bound.",
        },
        "structural_work_model": {
            "assumed_statevector_work": "Theta(G * 2^n), with G=Theta(n^2) two-qubit gates for one all-to-all layer",
            "time_model": "T(n) = T(anchor) * (n/anchor)^2 * 2^(n-anchor)",
            "memory_model": "16 * 2^n bytes for one complex128 statevector, excluding simulator workspace",
            "gate_counts": "n(n-1)/2 logical CRY gates and approximately n(n-1) basis CX gates per dense layer",
            "important_limit": "Algorithm-specific exact statevector model, not a lower bound on all classical simulators and not proof that this gate family satisfies a published random-circuit-sampling theorem.",
        },
        "projections": projections,
        "observed_two_qubit_size_growth": observed_growth,
        "classical_tensor_network_evidence": {
            "tested_full_numerical_schmidt_rank": [
                item for item in measurements if item["numerical_schmidt_rank"] is not None
            ],
            "interpretation": (
                "For the tested state, numerical full Schmidt rank across a balanced cut requires "
                "that bond dimension for exact MPS representation. Approximate MPS can use smaller "
                "bond dimensions but incur observable error. Sampled rank alone is not a theorem "
                "for every circuit or approximation."
            ),
        },
        "theoretical_hardness_status": {
            "current_hisab_circuit_family_formally_proved_hard": False,
            "exact_statevector_scaling_extrapolated": True,
            "approximate_quantum_sampling_advantage_proved": False,
            "conditional_hard_family_route": (
                "Validate an efficient semantic-operator encoding of a known universal or random-circuit-sampling family, "
                "then state the applicable complexity assumptions separately from NLP utility."
            ),
        },
    }
    (output_dir / "asymptotic_projection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Dense Hisab circuit: observed scaling and extrapolation",
        "",
        "## Observed one-layer all-to-all statevector runs",
        "",
        "| Qubits | Replicates | Median runtime (s) | Statevector minimum memory (GiB) | Logical CRY gates | Numerical central Schmidt rank | Central-cut entropy (bits) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in measurements:
        rank = str(item["numerical_schmidt_rank"]) if item["numerical_schmidt_rank"] is not None else "not measured"
        entropy = f"{item['central_cut_entropy_bits']:.3f}" if item["central_cut_entropy_bits"] is not None else "not measured"
        lines.append(
            f"| {item['n_qubits']} | {item['replicates']} | {item['runtime_seconds_median']:.4f} "
            f"| {item['minimum_statevector_memory_bytes'] / (1024**3):.4f} "
            f"| {item['logical_cry_gates_per_dense_layer']} | {rank} | {entropy} |"
        )
    lines.extend([
        "",
        "## Observed growth per two added qubits",
        "",
        "| From → to | Runtime ratio | Statevector memory ratio |",
        "|---|---:|---:|",
    ])
    for item in observed_growth:
        lines.append(
            f"| {item['from_qubits']} → {item['to_qubits']} | {item['runtime_ratio_median']:.3f}× "
            f"| {item['raw_statevector_memory_ratio']:.0f}× |"
        )
    lines.extend([
        "",
        "## Extrapolation (not measurement)",
        "",
        f"- Log-linear fit R² in log-time space: {fit['r_squared_log_space']:.4f}",
        f"- Fitted runtime multiplier per added qubit: {empirical_per_qubit_factor:.3f}×",
        f"- Fit sizes: {', '.join(str(item['n_qubits']) for item in fit_points)} qubits.",
        "- The structural estimate uses the statevector work model Θ(n²·2ⁿ), anchored at the largest measured size.",
        "",
        "| Qubits | Empirical fit time | Structural n²·2ⁿ time | Minimum statevector memory | Logical CRY gates | Basis CX estimate |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for item in projections:
        lines.append(
            f"| {item['n_qubits']} | {item['empirical_log_linear_runtime_seconds'] / 60:.2f} min "
            f"| {item['statevector_work_model_n2_2n_seconds'] / 60:.2f} min "
            f"| {item['minimum_complex128_statevector_gib']:.2f} GiB "
            f"| {item['logical_cry_gates_per_dense_layer']} "
            f"| {item['approx_basis_cx_count_per_dense_layer']} |"
        )
    lines.extend([
        "",
        "Both time columns are algorithm-specific projections, not measured runtimes and not lower bounds on the best classical algorithm. Runtime assumes the same machine/software profile and should not be extrapolated indefinitely from a short range.",
        "",
        "## What complexity theory does and does not give us",
        "",
        "- For the measured current Hisab synthetic family, we have empirical exponential statevector growth and circuit-specific entanglement diagnostics. We do not have a hardness theorem for this exact gate ensemble.",
        "- Exact simulation of sufficiently general universal quantum circuits can encode #P-hard probability/amplitude calculations. Approximate-sampling hardness for random-circuit families relies on stronger ensemble-specific average-case hardness and anti-concentration assumptions.",
        "- A theoretical candidate is an extended operator grammar that efficiently maps semantic superposition, non-Clifford phase/interference, and compositional dependency to a universal gate set. This would establish a hard worst-case circuit subfamily, not that ordinary language inputs are hard or that the semantic task is useful.",
        "- To claim a computational advantage for Hisab itself, the same well-defined inference task must be solved by a quantum circuit and strong classical competitors, with input encoding, output sampling, error, circuit depth, shots, and hardware noise included.",
        "",
        "## Reproduction",
        "",
        "Run python scripts/run_scaling_aer.py --max-qubits 26 --output-dir results/scaling_aer, then python scripts/project_statevector_scaling.py --input-csv results/scaling_aer/scaling_results.csv --output-dir results/asymptotic_projection.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=Path("results/scaling_aer/scaling_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/asymptotic_projection"))
    parser.add_argument("--projection-qubits", type=str, default="28,30,32,34,36,40,50")
    args = parser.parse_args()
    sizes = [int(item.strip()) for item in args.projection_qubits.split(",") if item.strip()]
    report = project(args.input_csv, args.output_dir, sizes)
    print(json.dumps({
        "benchmark": report["benchmark"],
        "observed_sizes": [item["n_qubits"] for item in report["observed_measurements"]],
        "empirical_fit": report["empirical_fit"],
        "projections": report["projections"],
        "current_family_hardness_proved": False,
    }, indent=2))


if __name__ == "__main__":
    main()
