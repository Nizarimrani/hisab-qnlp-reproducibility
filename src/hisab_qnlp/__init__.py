"""Hisab Labs compositional quantum-semantics reference package."""

from .composition import ExperimentConfig, build_circuit, run_experiment, summarize_counts
from .operator_semantics import (
    OperatorType,
    SemanticOperator,
    build_operator_circuit,
    compose_symbolically,
    evaluate_operator_circuit,
    operators_from_token_tags,
    validate_composition,
)

__all__ = [
    "ExperimentConfig",
    "OperatorType",
    "SemanticOperator",
    "build_circuit",
    "build_operator_circuit",
    "compose_symbolically",
    "evaluate_operator_circuit",
    "operators_from_token_tags",
    "run_experiment",
    "summarize_counts",
    "validate_composition",
]
