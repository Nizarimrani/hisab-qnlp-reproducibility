from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_nubes_repeated_splits import run_repeated_benchmark  # noqa: E402
from scripts.run_nubes_operator_scope_benchmark import (  # noqa: E402
    brat_linked_scope_metrics,
    hisab_operator_pipeline,
    load_document,
    parse_brat,
    parse_brat_detailed,
    split_by_document,
    scope_from_cue_tags,
    token_features,
)


def test_brat_scope_relations_assign_negation_and_uncertainty():
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        text_path = root / "example.txt"
        ann_path = root / "example.ann"
        text_path.write_text("No fiebre alta.\nPuede mejorar.\n", encoding="utf-8")
        ann_path.write_text(
            "T1\tNegSynMarker 0 2\tNo\n"
            "T2\tPhrase 3 14\tfiebre alta\n"
            "T3\tUncertSynMarker 16 21\tPuede\n"
            "T4\tPhrase 22 29\tmejorar\n"
            "R1\tScope Arg1:T1 Arg2:T2\n"
            "R2\tScope Arg1:T3 Arg2:T4\n",
            encoding="utf-8",
        )

        cues, scopes = parse_brat(ann_path)
        assert len(cues) == 2
        assert ("NEG", 3, 14, "Scope") in scopes
        assert ("UNCERT", 22, 29, "Scope") in scopes

        document = load_document(text_path)
        by_token = {item["token"]: item["label"] for item in document["tokens"]}
        assert by_token["No"] == "NEG_CUE"
        assert by_token["fiebre"] == "NEG_SCOPE"
        assert by_token["alta"] == "NEG_SCOPE"
        assert by_token["Puede"] == "UNCERT_CUE"
        assert by_token["mejorar"] == "UNCERT_SCOPE"


def test_features_use_local_context_without_gold_label():
    tokens = [
        {"token": "No", "lower": "no", "line": 0, "label": "NEG_CUE"},
        {"token": "presenta", "lower": "presenta", "line": 0, "label": "NEG_SCOPE"},
        {"token": "fiebre", "lower": "fiebre", "line": 0, "label": "NEG_SCOPE"},
        {"token": "Puede", "lower": "puede", "line": 1, "label": "UNCERT_CUE"},
    ]
    features = token_features(tokens, 1)
    assert features["w"] == "presenta"
    assert features["prev"] == "no"
    assert features["next"] == "fiebre"
    assert "next2" not in features


def test_document_split_has_no_file_overlap_and_is_seeded():
    docs = [
        {"doc_id": f"doc-{i:02d}", "tokens": [{"token": "x", "lower": "x", "line": 0, "label": "O"}]}
        for i in range(20)
    ]
    train, test = split_by_document(docs, 0.2, seed=101)
    train_repeat, test_repeat = split_by_document(docs, 0.2, seed=101)
    _, other_test = split_by_document(docs, 0.2, seed=202)
    assert {d["doc_id"] for d in train}.isdisjoint({d["doc_id"] for d in test})
    assert len(test) == 4
    assert [d["doc_id"] for d in test] == [d["doc_id"] for d in test_repeat]
    assert [d["doc_id"] for d in train] == [d["doc_id"] for d in train_repeat]
    assert {d["doc_id"] for d in other_test} != {d["doc_id"] for d in test}


def test_hisab_inspired_pipeline_detects_simple_cues_and_local_scope():
    tokens = [
        {"token": "No", "lower": "no", "line": 0, "label": "O"},
        {"token": "presenta", "lower": "presenta", "line": 0, "label": "O"},
        {"token": "fiebre", "lower": "fiebre", "line": 0, "label": "O"},
        {"token": ".", "lower": ".", "line": 0, "label": "O"},
        {"token": "Puede", "lower": "puede", "line": 1, "label": "O"},
        {"token": "mejorar", "lower": "mejorar", "line": 1, "label": "O"},
        {"token": ".", "lower": ".", "line": 1, "label": "O"},
    ]
    predicted = hisab_operator_pipeline(tokens)
    assert predicted == [
        "NEG_CUE", "NEG_SCOPE", "NEG_SCOPE", "O",
        "UNCERT_CUE", "UNCERT_SCOPE", "O",
    ]


def test_hisab_inspired_pipeline_never_reads_gold_labels():
    tokens = [
        {"token": "No", "lower": "no", "line": 0, "label": "NEG_CUE"},
        {"token": "fiebre", "lower": "fiebre", "line": 0, "label": "NEG_SCOPE"},
        {"token": ".", "lower": ".", "line": 0, "label": "O"},
    ]
    first = hisab_operator_pipeline(tokens)
    altered = [dict(token) for token in tokens]
    for token in altered:
        token["label"] = "UNCERT_CUE" if token["label"] == "O" else "O"
    assert hisab_operator_pipeline(altered) == first

def test_scope_from_predicted_cues_uses_punctuation_boundaries_only():
    tokens = [
        {"token": "No", "line": 0},
        {"token": "presenta", "line": 0},
        {"token": "fiebre", "line": 0},
        {"token": ".", "line": 0},
        {"token": "quizá", "line": 0},
        {"token": "mejore", "line": 0},
        {"token": ".", "line": 0},
    ]
    cues = ["NEG_CUE", "O", "O", "O", "UNCERT_CUE", "O", "O"]
    assert scope_from_cue_tags(tokens, cues) == [
        "NEG_CUE", "NEG_SCOPE", "NEG_SCOPE", "O",
        "UNCERT_CUE", "UNCERT_SCOPE", "O",
    ]


def test_both_cue_generates_mixed_scope_without_gold_labels():
    tokens = [
        {"token": "No", "line": 0, "label": "WRONG"},
        {"token": "descartado", "line": 0, "label": "WRONG"},
        {"token": ".", "line": 0, "label": "WRONG"},
    ]
    cues = ["BOTH_CUE", "O", "O"]
    assert scope_from_cue_tags(tokens, cues) == ["BOTH_CUE", "MIXED_SCOPE", "O"]


def test_scope_from_cue_tags_rejects_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        scope_from_cue_tags([{"token": "No", "line": 0}], [])

def test_repeated_split_runner_aggregates_seeded_results(tmp_path, monkeypatch):
    import scripts.run_nubes_repeated_splits as repeated

    def fake_run_benchmark(corpus_dir, output_dir, test_fraction, seed):
        model = {
            "model": "hybrid",
            "accuracy": seed / 100.0,
            "macro_f1_all_labels": seed / 100.0,
            "macro_f1_operator_labels_present_in_test": seed / 100.0,
            "exact_span_metrics": {
                "cue_spans": {"f1": seed / 100.0},
                "scope_spans": {"f1": (seed + 10) / 100.0},
            },
        }
        return {
            "data": {
                "train_documents": 8,
                "test_documents": 2,
                "train_tokens": 100,
                "test_tokens": 20,
            },
            "models": [model],
            "operator_composition_validation": {
                "typed_operators_extracted_from_test_text": 3,
                "annotated_scope_tokens_evaluated": 4,
                "circuit_symbolic_checks": 4,
                "circuit_symbolic_agreements": 4,
            },
        }

    monkeypatch.setattr(repeated, "run_benchmark", fake_run_benchmark)
    report = run_repeated_benchmark(
        corpus_dir=tmp_path / "unused-corpus",
        output_dir=tmp_path / "repeated",
        seeds=[10, 20, 30],
    )
    assert report["seed_values"] == [10, 20, 30]
    assert report["aggregate_metrics"]["hybrid"]["accuracy"]["mean"] == 0.2
    assert report["aggregate_metrics"]["hybrid"]["accuracy"]["sample_standard_deviation"] == 0.1
    assert (tmp_path / "repeated" / "REPORT.md").exists()
    assert (tmp_path / "repeated" / "repeated_split_metrics.json").exists()


def test_repeated_split_runner_rejects_duplicate_seeds(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unique"):
        run_repeated_benchmark(
            corpus_dir=tmp_path / "unused-corpus",
            output_dir=tmp_path / "repeated",
            seeds=[7, 7],
        )

def test_brat_parser_preserves_linked_discontinuous_scope_segments(tmp_path):
    text_path = tmp_path / "discontinuous.txt"
    ann_path = tmp_path / "discontinuous.ann"
    text_path.write_text("No fiebre de alta.\n", encoding="utf-8")
    ann_path.write_text(
        "T1\tNegSynMarker 0 2\tNo\n"
        "T2\tPhrase 3 9;13 17\tfiebre alta\n"
        "R1\tScope Arg1:T1 Arg2:T2\n",
        encoding="utf-8",
    )

    cues, linked_targets = parse_brat_detailed(ann_path)
    assert len(cues) == 1
    assert cues[0]["segments"] == [(0, 2)]
    assert len(linked_targets) == 1
    assert linked_targets[0]["target_segments"] == [(3, 9), (13, 17)]

    document = load_document(text_path)
    labels = [token["label"] for token in document["tokens"]]
    assert labels == ["NEG_CUE", "NEG_SCOPE", "O", "NEG_SCOPE", "O"]

    # The two predicted contiguous runs must not count as an exact match for
    # the one original BRAT target entity containing two linked segments.
    score = brat_linked_scope_metrics([document], {document["doc_id"]: labels})
    assert score["gold_discontinuous_targets"] == 1
    assert score["overall"]["gold_targets"] == 1
    assert score["overall"]["predicted_targets"] == 2
    assert score["overall"]["exact_matches"] == 0
    assert score["overall"]["f1"] == 0.0

