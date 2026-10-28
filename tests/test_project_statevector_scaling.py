from pathlib import Path
import csv
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_statevector_scaling import project  # noqa: E402


def test_projection_models_memory_and_gate_counts(tmp_path: Path):
    csv_path = tmp_path / "scaling.csv"
    fields = [
        "status", "n_qubits", "topology", "logical_layers", "seed",
        "elapsed_seconds", "central_cut_entropy_bits",
        "central_cut_schmidt_rank_numeric",
    ]
    rows = []
    # Runtime follows a dense statevector-like growth pattern for fixture purposes.
    for n, seconds in [(16, 0.03), (20, 0.5), (22, 2.5), (24, 13.0), (26, 65.0)]:
        rows.append({
            "status": "ok", "n_qubits": n, "topology": "all_to_all",
            "logical_layers": 1, "seed": 11, "elapsed_seconds": seconds,
            "central_cut_entropy_bits": min(n / 4, 5),
            "central_cut_schmidt_rank_numeric": 1 << (n // 2),
        })
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = project(csv_path, tmp_path / "out", [28, 30])
    by_n = {item["n_qubits"]: item for item in result["projections"]}
    assert by_n[28]["minimum_complex128_statevector_bytes"] == 16 * (1 << 28)
    assert by_n[28]["logical_cry_gates_per_dense_layer"] == 28 * 27 // 2
    assert by_n[28]["approx_basis_cx_count_per_dense_layer"] == 28 * 27
    assert by_n[30]["minimum_complex128_statevector_bytes"] == 4 * by_n[28]["minimum_complex128_statevector_bytes"]
    assert result["empirical_fit"]["r_squared_log_space"] > 0.99
    assert (tmp_path / "out" / "REPORT.md").exists()
    assert (tmp_path / "out" / "asymptotic_projection.json").exists()


def test_projection_requires_four_distinct_large_sizes(tmp_path: Path):
    csv_path = tmp_path / "small.csv"
    fields = ["status", "n_qubits", "topology", "logical_layers", "seed", "elapsed_seconds"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for n, seconds in [(20, 0.5), (22, 2.5), (24, 13.0)]:
            writer.writerow({
                "status": "ok", "n_qubits": n, "topology": "all_to_all",
                "logical_layers": 1, "seed": 11, "elapsed_seconds": seconds,
            })
    with pytest.raises(ValueError, match="four or more"):
        project(csv_path, tmp_path / "out", [28])


def test_projection_rejects_odd_projection_sizes(tmp_path: Path):
    with pytest.raises(ValueError, match="even integers"):
        project(tmp_path / "missing.csv", tmp_path / "out", [29])
