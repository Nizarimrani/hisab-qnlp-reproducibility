from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_semantic_task_benchmark import (  # noqa: E402
    ASPECT_NAMES,
    STATUS_NAMES,
    make_dataset,
    run_circuit,
    symbolic_distribution,
    symbolic_oracle,
)


def test_dataset_is_complete_factorial_design_and_balanced_by_label():
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
    assert {label: sum(r["target_label"] == label for r in rows) for label in STATUS_NAMES} == {
        "inactive": 36,
        "not_started": 12,
        "ongoing": 12,
        "completed": 12,
    }


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
        assert symbolic_distribution(row) == symbolic_distribution(by_signature[key])


def test_qiskit_output_distribution_matches_symbolic_rule_for_all_examples():
    rows = make_dataset()
    for row in rows:
        actual = run_circuit(row)
        expected = symbolic_distribution(row)
        assert set(actual) == set(expected)
        assert sum(actual.values()) == pytest.approx(1.0, abs=1e-9)
        for label in STATUS_NAMES:
            assert actual[label] == pytest.approx(expected[label], abs=1e-9)


def test_factuality_controls_activation_probability():
    rows = make_dataset()
    eligible = [
        row for row in rows
        if (bool(row["premise_truth"]) ^ bool(row["negated"]))
        and row["aspect_stage"] == 1
        and row["concession"] == 0
    ]
    assert sorted(row["factuality_probability"] for row in eligible) == [0.2, 0.2, 0.5, 0.5, 0.8, 0.8]
    for row in eligible:
        distribution = run_circuit(row)
        assert distribution["ongoing"] == pytest.approx(row["factuality_probability"], abs=1e-9)
        assert distribution["inactive"] == pytest.approx(1 - row["factuality_probability"], abs=1e-9)


def test_all_semantic_output_labels_are_defined():
    assert len(ASPECT_NAMES) == 3
    assert set(row["target_label"] for row in make_dataset()) == set(STATUS_NAMES)
