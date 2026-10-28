from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_semantic_task_benchmark import (  # noqa: E402
    ASPECT_NAMES,
    make_dataset,
    run_circuit,
    symbolic_oracle,
)


def test_dataset_is_complete_factorial_design():
    rows = make_dataset()
    assert len(rows) == 72
    signatures = {
        (
            row["premise_truth"],
            row["negated"],
            row["factuality_level"],
            row["concession"],
            row["aspect_stage"],
        )
        for row in rows
    }
    assert len(signatures) == 72
    assert {row["aspect_stage"] for row in rows} == {0, 1, 2}


def test_concession_does_not_cancel_the_inference_branch():
    rows = make_dataset()
    by_signature = {
        (
            row["premise_truth"],
            row["negated"],
            row["factuality_level"],
            row["aspect_stage"],
            row["concession"],
        ): row
        for row in rows
    }
    for row in rows:
        opposite_concession = 1 - row["concession"]
        key = (
            row["premise_truth"],
            row["negated"],
            row["factuality_level"],
            row["aspect_stage"],
            opposite_concession,
        )
        assert symbolic_oracle(row) == symbolic_oracle(by_signature[key])


def test_qiskit_circuit_matches_symbolic_rule_for_all_examples():
    rows = make_dataset()
    for row in rows:
        assert run_circuit(row) == symbolic_oracle(row)


def test_all_four_output_labels_are_defined():
    assert len(ASPECT_NAMES) == 3
    assert set(row["target_label"] for row in make_dataset()) == {
        "inactive", "not_started", "ongoing", "completed"
    }
