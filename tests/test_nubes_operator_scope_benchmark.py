from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_nubes_operator_scope_benchmark import (  # noqa: E402
    load_document,
    parse_brat,
    split_by_document,
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


def test_document_split_has_no_file_overlap():
    docs = [
        {"doc_id": f"doc-{i:02d}", "tokens": [{"token": "x", "lower": "x", "line": 0, "label": "O"}]}
        for i in range(10)
    ]
    train, test = split_by_document(docs, 0.2)
    assert {d["doc_id"] for d in train}.isdisjoint({d["doc_id"] for d in test})
    assert len(test) == 2
