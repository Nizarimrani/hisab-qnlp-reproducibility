import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_mps_bond_sweep import run_one  # noqa: E402


def test_small_matched_all_to_all_circuit_reproduces_expectations():
    row = run_one(4, 11, 16, 1e-12, mps_timeout_seconds=60)
    assert row["status"] == "ok", row.get("error")
    assert row["checkpoint_stage"] == "mps_complete"
    assert row["observable_count"] == 8
    assert row["mean_abs_observable_error"] < 1e-9
    assert row["max_abs_observable_error"] < 1e-9


def test_mps_timeout_keeps_statevector_checkpoint(tmp_path):
    # A 1 ms deadline deterministically exercises the timeout/recovery path.
    row = run_one(
        4, 11, 16, 1e-12,
        output_dir=tmp_path,
        mps_timeout_seconds=0.001,
    )
    assert row["status"] == "mps_timeout", row
    assert row["checkpoint_stage"] == "mps_timeout_reference_preserved"
    assert row["statevector_seconds"] != ""
    assert row["mps_seconds"] == ""
    assert (tmp_path / "result.csv").exists()
    assert (tmp_path / "checkpoint.json").exists()
    assert (tmp_path / "statevector_expectations.json").exists()
    saved = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert saved["status"] == "mps_timeout"
    reference = json.loads((tmp_path / "statevector_expectations.json").read_text(encoding="utf-8"))
    assert reference["n_observables"] == 8
