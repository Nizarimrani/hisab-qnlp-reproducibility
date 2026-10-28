from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hisab_qnlp.operator_semantics import (  # noqa: E402
    OperatorType,
    SemanticOperator,
    build_operator_circuit,
    compose_symbolically,
    evaluate_operator_circuit,
    operators_from_token_tags,
    validate_composition,
)


def op(operator_id, kind, scope, cue=(0,)):
    return SemanticOperator(
        operator_id=operator_id,
        kind=kind,
        cue_token_indices=tuple(cue),
        scope_token_indices=tuple(scope),
    )


@pytest.mark.parametrize(
    "base_truth,operators,target,expected_truth,expected_uncertain",
    [
        (True, [], 2, True, False),
        (True, [op("neg1", OperatorType.NEGATION, (2,))], 2, False, False),
        (False, [op("neg1", OperatorType.NEGATION, (2,))], 2, True, False),
        (
            True,
            [
                op("neg1", OperatorType.NEGATION, (2,)),
                op("neg2", OperatorType.NEGATION, (2,)),
            ],
            2,
            True,
            False,
        ),
        (True, [op("unc1", OperatorType.UNCERTAINTY, (2,))], 2, True, True),
        (
            False,
            [
                op("neg1", OperatorType.NEGATION, (2,)),
                op("unc1", OperatorType.UNCERTAINTY, (2,)),
            ],
            2,
            True,
            True,
        ),
        (True, [op("neg1", OperatorType.NEGATION, (1,))], 2, True, False),
    ],
)
def test_symbolic_and_circuit_composition_agree(
    base_truth, operators, target, expected_truth, expected_uncertain
):
    result = validate_composition(base_truth, operators, target)
    assert result["agreement"]
    assert result["symbolic"]["truth"] is expected_truth
    assert result["symbolic"]["uncertain"] is expected_uncertain
    assert result["circuit"]["truth"] is expected_truth
    assert result["circuit"]["uncertain"] is expected_uncertain


def test_variable_length_circuit_preserves_multiple_uncertainty_sources():
    operators = [
        op("u1", OperatorType.UNCERTAINTY, (3,), cue=(0,)),
        op("u2", OperatorType.UNCERTAINTY, (3,), cue=(1,)),
        op("n1", OperatorType.NEGATION, (3,), cue=(2,)),
    ]
    circuit = build_operator_circuit(True, operators, 3)
    output = evaluate_operator_circuit(True, operators, 3)
    assert circuit.num_qubits == 3
    assert output["truth"] is False
    assert output["uncertain"] is True
    assert output["uncertainty_flags"] == [True, True]


def test_typed_schema_rejects_duplicate_or_unsorted_token_indices():
    with pytest.raises(ValueError):
        SemanticOperator("bad", OperatorType.NEGATION, (1, 0), (2,))
    with pytest.raises(ValueError):
        SemanticOperator("bad", OperatorType.NEGATION, (0,), (2, 2))


def test_token_tags_convert_to_typed_operators_without_reading_gold_labels():
    tokens = [
        {"token": "No", "line": 0, "label": "O"},
        {"token": "presenta", "line": 0, "label": "NEG_SCOPE"},
        {"token": "fiebre", "line": 0, "label": "NEG_SCOPE"},
        {"token": ".", "line": 0, "label": "O"},
        {"token": "Puede", "line": 1, "label": "O"},
        {"token": "mejorar", "line": 1, "label": "O"},
        {"token": ".", "line": 1, "label": "O"},
    ]
    tags = ["NEG_CUE", "NEG_SCOPE", "NEG_SCOPE", "O", "UNCERT_CUE", "UNCERT_SCOPE", "O"]
    first = operators_from_token_tags(tokens, tags, source="predicted")
    altered_tokens = [dict(token, label="WRONG_GOLD_LABEL") for token in tokens]
    second = operators_from_token_tags(altered_tokens, tags, source="predicted")
    assert [op.to_dict() for op in first] == [op.to_dict() for op in second]
    assert len(first) == 2
    assert first[0].kind == OperatorType.NEGATION
    assert first[0].scope_token_indices == (1, 2)
    assert first[1].kind == OperatorType.UNCERTAINTY
    assert first[1].scope_token_indices == (5,)
