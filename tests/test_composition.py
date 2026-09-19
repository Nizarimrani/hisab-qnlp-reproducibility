import math

import pytest

from hisab_qnlp.composition import (
    ExperimentConfig,
    bit_at,
    probability_to_ry_angle,
    run_experiment,
    summarize_counts,
)


def test_probability_angle_boundaries():
    assert probability_to_ry_angle(0.0) == 0.0
    assert probability_to_ry_angle(1.0) == pytest.approx(math.pi)
    assert math.sin(probability_to_ry_angle(0.8) / 2) ** 2 == pytest.approx(0.8)


def test_bit_ordering():
    assert bit_at("100000000", 8) == 1
    assert bit_at("000000001", 0) == 1
    assert bit_at("010000000", 7) == 1


def test_summary_inference_gate_invariant():
    counts = {
        "000000000": 75,
        "101000000": 25,
    }
    summary = summarize_counts(counts)
    assert summary["shots"] == 100
    assert summary["p_red_given_no_inference"] == 0.0


def test_invalid_config():
    with pytest.raises(ValueError):
        ExperimentConfig(factuality_probability=1.1)


def test_seeded_run_is_reproducible():
    config = ExperimentConfig(shots=256)
    first = run_experiment(config)
    second = run_experiment(config)
    assert first["counts"] == second["counts"]
    assert first["metrics"]["shots"] == 256
    assert first["metrics"]["p_red_given_no_inference"] == 0.0

