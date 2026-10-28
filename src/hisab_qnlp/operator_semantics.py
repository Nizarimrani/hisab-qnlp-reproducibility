"""Typed operator schema and variable-length compositional Qiskit encoding.

The schema separates linguistic extraction (cue/scope spans), semantic
composition (reference rule), and circuit realization. It makes no claim that
the resulting deterministic X-gate circuit offers quantum advantage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class OperatorType(str, Enum):
    NEGATION = "negation"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class SemanticOperator:
    """One typed cue and its target scope in token-index coordinates."""

    operator_id: str
    kind: OperatorType
    cue_token_indices: tuple[int, ...]
    scope_token_indices: tuple[int, ...]
    source: str = "predicted"

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty")
        if not self.cue_token_indices:
            raise ValueError("cue_token_indices must not be empty")
        if any(index < 0 for index in (*self.cue_token_indices, *self.scope_token_indices)):
            raise ValueError("token indices must be non-negative")
        if tuple(sorted(set(self.cue_token_indices))) != self.cue_token_indices:
            raise ValueError("cue token indices must be sorted and unique")
        if tuple(sorted(set(self.scope_token_indices))) != self.scope_token_indices:
            raise ValueError("scope token indices must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


def operators_from_token_tags(
    tokens: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    source: str = "predicted",
) -> list[SemanticOperator]:
    """Convert cue/scope token tags into a typed, auditable operator list.

    Only lexical tokens and *predicted* tags are consumed. This conversion does
    not read token['label'] or any other reference annotation. Scope is limited
    to target tokens tagged with the matching scope label after a cue, stopping
    at punctuation and line boundaries. A token tagged MIXED_SCOPE is attached
    to both operator families, including when another cue intervenes. This
    supports explicitly annotated overlap but cannot infer missing scope structure.
    """
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length")

    cue_to_type = {
        "NEG_CUE": OperatorType.NEGATION,
        "UNCERT_CUE": OperatorType.UNCERTAINTY,
        "BOTH_CUE": None,
    }
    scope_label = {
        OperatorType.NEGATION: "NEG_SCOPE",
        OperatorType.UNCERTAINTY: "UNCERT_SCOPE",
    }
    punctuation = {".", ";", ":", "!", "?", "…"}
    cue_runs: list[tuple[OperatorType, tuple[int, ...]]] = []

    index = 0
    while index < len(labels):
        tag = labels[index]
        if tag not in cue_to_type:
            index += 1
            continue

        kinds = (
            (OperatorType.NEGATION, OperatorType.UNCERTAINTY)
            if tag == "BOTH_CUE"
            else (cue_to_type[tag],)
        )
        assert all(kind is not None for kind in kinds)
        line = int(tokens[index].get("line", 0))
        end = index + 1
        while (
            end < len(labels)
            and labels[end] == tag
            and int(tokens[end].get("line", 0)) == line
        ):
            end += 1
        cue_indices = tuple(range(index, end))
        for kind in kinds:
            cue_runs.append((kind, cue_indices))  # type: ignore[arg-type]
        index = end

    operators: list[SemanticOperator] = []
    for ordinal, (kind, cue_indices) in enumerate(cue_runs):
        cue_end = max(cue_indices)
        cue_line = int(tokens[cue_end].get("line", 0))
        scope_indices: list[int] = []
        cursor = cue_end + 1
        crossed_another_cue = False
        while cursor < len(tokens):
            if int(tokens[cursor].get("line", 0)) != cue_line:
                break
            token_text = str(tokens[cursor].get("token", ""))
            if token_text in punctuation:
                break
            if labels[cursor] in {"NEG_CUE", "UNCERT_CUE", "BOTH_CUE"}:
                crossed_another_cue = True
                cursor += 1
                continue
            if labels[cursor] == "MIXED_SCOPE":
                scope_indices.append(cursor)
            elif not crossed_another_cue and labels[cursor] == scope_label[kind]:
                scope_indices.append(cursor)
            cursor += 1

        operators.append(SemanticOperator(
            operator_id=f"op-{ordinal:04d}",
            kind=kind,
            cue_token_indices=cue_indices,
            scope_token_indices=tuple(scope_indices),
            source=source,
        ))
    return operators


def compose_symbolically(
    base_truth: bool,
    operators: Iterable[SemanticOperator],
    target_token_index: int,
) -> dict[str, Any]:
    """Reference composition rule for one target token.

    Each scoped negation toggles truth; double negation cancels by parity.
    Any scoped uncertainty marks the result as uncertain without inventing a
    numeric probability. It is a minimal operational definition, not a full
    linguistic theory of negation or modality.
    """
    if target_token_index < 0:
        raise ValueError("target_token_index must be non-negative")
    truth = bool(base_truth)
    applicable: list[SemanticOperator] = []
    uncertainty_count = 0
    negation_count = 0
    for operator in operators:
        if target_token_index not in operator.scope_token_indices:
            continue
        applicable.append(operator)
        if operator.kind == OperatorType.NEGATION:
            truth = not truth
            negation_count += 1
        elif operator.kind == OperatorType.UNCERTAINTY:
            uncertainty_count += 1
    return {
        "base_truth": bool(base_truth),
        "truth": truth,
        "uncertain": uncertainty_count > 0,
        "negation_count": negation_count,
        "uncertainty_count": uncertainty_count,
        "applicable_operator_ids": [op.operator_id for op in applicable],
    }


def build_operator_circuit(
    base_truth: bool,
    operators: Iterable[SemanticOperator],
    target_token_index: int,
) -> QuantumCircuit:
    """Build a variable-length X-gate circuit implementing the rule above.

    q0 stores truth. Each applicable uncertainty operator has its own flag
    qubit, preserving provenance rather than XOR-canceling repeated operators.
    This is a dynamically constructed, deterministic circuit, not a quantum
    algorithm or a demonstration of quantum computational advantage.
    """
    applicable = [
        op for op in operators if target_token_index in op.scope_token_indices
    ]
    negations = [op for op in applicable if op.kind == OperatorType.NEGATION]
    uncertainties = [op for op in applicable if op.kind == OperatorType.UNCERTAINTY]
    qc = QuantumCircuit(1 + len(uncertainties), name=f"semantic-token-{target_token_index}")
    truth_qubit = 0
    if bool(base_truth):
        qc.x(truth_qubit)
    for _operator in negations:
        qc.x(truth_qubit)
    for flag_index, _operator in enumerate(uncertainties, start=1):
        qc.x(flag_index)
    return qc


def evaluate_operator_circuit(
    base_truth: bool,
    operators: Iterable[SemanticOperator],
    target_token_index: int,
) -> dict[str, Any]:
    """Run the exact statevector and decode the output bits."""
    operator_list = list(operators)
    circuit = build_operator_circuit(base_truth, operator_list, target_token_index)
    state = Statevector.from_instruction(circuit)
    probabilities = state.probabilities()
    basis_index = max(range(len(probabilities)), key=lambda i: float(probabilities[i]))
    truth = bool(basis_index & 1)
    uncertainty_flags = [
        bool((basis_index >> qubit_index) & 1)
        for qubit_index in range(1, circuit.num_qubits)
    ]
    return {
        "truth": truth,
        "uncertain": any(uncertainty_flags),
        "uncertainty_flags": uncertainty_flags,
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_count": circuit.size(),
        "operation_counts": {str(k): int(v) for k, v in circuit.count_ops().items()},
        "statevector": True,
        "hardware_execution": False,
    }


def validate_composition(
    base_truth: bool,
    operators: Sequence[SemanticOperator],
    target_token_index: int,
) -> dict[str, Any]:
    """Compare circuit outputs with the independent symbolic composition."""
    symbolic = compose_symbolically(base_truth, operators, target_token_index)
    quantum_encoding = evaluate_operator_circuit(base_truth, operators, target_token_index)
    return {
        "symbolic": symbolic,
        "circuit": quantum_encoding,
        "agreement": (
            symbolic["truth"] == quantum_encoding["truth"]
            and symbolic["uncertain"] == quantum_encoding["uncertain"]
        ),
    }
