import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_mps_bond_sweep import run_one  # noqa: E402


def test_small_matched_all_to_all_circuit_reproduces_expectations():
    row = run_one(4, 11, 16, 1e-12)
    assert row["status"] == "ok", row.get("error")
    assert row["observable_count"] == 8
    assert row["mean_abs_observable_error"] < 1e-9
    assert row["max_abs_observable_error"] < 1e-9
