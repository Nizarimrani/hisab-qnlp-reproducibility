#!/usr/bin/env python3
"""NUBes SAMPLE-001 benchmark for Spanish negation/uncertainty cue and scope tagging.

Reads the upstream BRAT files at runtime; it does not copy or publish source
sentences or token predictions. Both tested models are classical CPU baselines.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

SEED = 20261028
UPSTREAM_REPOSITORY = "https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus"
LICENSE = "Creative Commons Attribution-ShareAlike 3.0 Spain"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
NEG_CUE_TYPES = {"NegSynMarker", "NegLexMarker", "NegMorMarker"}
UNCERT_CUE_TYPES = {"UncertSynMarker", "UncertLexMarker"}
SCOPE_RELATIONS = {"Scope", "DiscScope"}
LABELS = ["O", "NEG_CUE", "UNCERT_CUE", "BOTH_CUE", "NEG_SCOPE", "UNCERT_SCOPE", "MIXED_SCOPE"]


def parse_brat(ann_path: Path) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, str]]]:
    """Return cue entities and scope spans with their semantic category."""
    entities: dict[str, dict[str, Any]] = {}
    relations: list[tuple[str, str, str]] = []
    with ann_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("T"):
                try:
                    annotation_id, specification, surface = line.split("\t", 2)
                    bits = specification.split()
                    if len(bits) < 3:
                        continue
                    entities[annotation_id] = {
                        "id": annotation_id,
                        "type": bits[0],
                        "start": int(bits[1]),
                        "end": int(bits[2]),
                        "surface": surface,
                    }
                except (ValueError, IndexError):
                    continue
            elif line.startswith("R"):
                pieces = line.split("\t", 2)
                if len(pieces) < 2:
                    continue
                bits = pieces[1].split()
                if len(bits) >= 3:
                    relations.append((bits[0], bits[1].split(":", 1)[1],
                                      bits[2].split(":", 1)[1]))

    cues = [e for e in entities.values() if e["type"] in NEG_CUE_TYPES | UNCERT_CUE_TYPES]
    scope_spans: list[tuple[str, int, int, str]] = []
    for relation_type, arg1, arg2 in relations:
        if relation_type not in SCOPE_RELATIONS:
            continue
        cue, target = entities.get(arg1), entities.get(arg2)
        if not cue or not target:
            continue
        if cue["type"] in NEG_CUE_TYPES:
            category = "NEG"
        elif cue["type"] in UNCERT_CUE_TYPES:
            category = "UNCERT"
        else:
            continue
        scope_spans.append((category, int(target["start"]), int(target["end"]), relation_type))
    return cues, scope_spans


def load_document(text_path: Path) -> dict[str, Any]:
    ann_path = text_path.with_suffix(".ann")
    text = text_path.read_text(encoding="utf-8")
    cues, scopes = parse_brat(ann_path) if ann_path.exists() else ([], [])
    tokens: list[dict[str, Any]] = []
    line_offset = 0

    for line_number, line in enumerate(text.splitlines(keepends=True)):
        line_content = line.rstrip("\r\n")
        for match in TOKEN_RE.finditer(line_content):
            start, end = line_offset + match.start(), line_offset + match.end()
            cue_categories = {
                ("NEG" if entity["type"] in NEG_CUE_TYPES else "UNCERT")
                for entity in cues
                if start < int(entity["end"]) and end > int(entity["start"])
            }
            scope_categories = {
                category for category, scope_start, scope_end, _ in scopes
                if start < scope_end and end > scope_start
            }
            if cue_categories:
                label = "BOTH_CUE" if len(cue_categories) > 1 else (
                    "NEG_CUE" if "NEG" in cue_categories else "UNCERT_CUE"
                )
            elif scope_categories:
                label = "MIXED_SCOPE" if len(scope_categories) > 1 else (
                    "NEG_SCOPE" if "NEG" in scope_categories else "UNCERT_SCOPE"
                )
            else:
                label = "O"
            tokens.append({
                "token": match.group(0),
                "lower": match.group(0).lower(),
                "start": start,
                "end": end,
                "line": line_number,
                "label": label,
            })
        line_offset += len(line)
    return {"doc_id": text_path.stem, "tokens": tokens}


def token_features(tokens: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Classical lexical/context features using only text, never gold labels."""
    current = tokens[index]["token"]
    lower = current.lower()
    features: dict[str, Any] = {
        "w": lower,
        "shape": (
            "NUM" if lower.isdigit()
            else "CAP" if current[:1].isupper()
            else "PUNCT" if re.fullmatch(r"[^\w]+", current)
            else "OTHER"
        ),
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "is_digit": lower.isdigit(),
        "is_upper_initial": current[:1].isupper(),
        "is_punct": bool(re.fullmatch(r"[^\w]+", current)),
    }
    for delta, tag in ((-2, "prev2"), (-1, "prev"), (1, "next"), (2, "next2")):
        neighbor = index + delta
        if 0 <= neighbor < len(tokens) and tokens[neighbor]["line"] == tokens[index]["line"]:
            features[tag] = tokens[neighbor]["lower"]
    return features


def hisab_operator_pipeline(tokens: list[dict[str, Any]]) -> list[str]:
    """Transparent rule-based operator extractor plus scope propagation.

    Inference uses only token strings and sentence boundaries. Gold annotation
    labels are deliberately ignored. This is a classical Hisab-inspired
    operator baseline, not the fixed nine-qubit Qiskit circuit.
    """
    uncertainty_patterns = [
        ("no", "se", "puede", "descartar"),
        ("no", "sabe", "precisar"),
        ("no", "se", "descarta"),
        ("al", "parecer"),
        ("sin", "claro"),
        ("no", "claro"),
        ("puede",),
        ("podría",),
        ("podria",),
        ("podrían",),
        ("podrian",),
        ("probable",),
        ("probablemente",),
        ("posible",),
        ("posiblemente",),
        ("quizás",),
        ("quizas",),
        ("aparentemente",),
        ("aparente",),
        ("presunto",),
        ("presunta",),
        ("sospecha",),
        ("parece",),
    ]
    negation_patterns = [
        ("ausencia", "de"),
        ("ninguno",),
        ("ninguna",),
        ("ningún",),
        ("ningun",),
        ("ningunas",),
        ("ningunos",),
        ("nunca",),
        ("jamás",),
        ("jamas",),
        ("asintomático",),
        ("asintomatico",),
        ("asintomática",),
        ("asintomatica",),
        ("tampoco",),
        ("sin",),
        ("no",),
        ("ni",),
    ]
    patterns = sorted(
        [(p, "UNCERT") for p in uncertainty_patterns]
        + [(p, "NEG") for p in negation_patterns],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    words = [str(token["token"]).casefold() for token in tokens]
    lines = [int(token["line"]) for token in tokens]
    cue_categories: dict[int, set[str]] = {}
    cue_starts: list[tuple[int, int, str]] = []

    # Longest patterns first preserve multi-token cues such as "no sabe precisar"
    # rather than independently labeling each embedded "no" as negation.
    for pattern, category in patterns:
        width = len(pattern)
        for start in range(0, len(tokens) - width + 1):
            positions = range(start, start + width)
            if any(lines[i] != lines[start] for i in positions):
                continue
            if tuple(words[start:start + width]) != pattern:
                continue
            if any(i in cue_categories for i in positions):
                continue
            cue_starts.append((start, start + width, category))
            for i in positions:
                cue_categories.setdefault(i, set()).add(category)

    labels = ["O"] * len(tokens)
    for index, categories in cue_categories.items():
        labels[index] = "BOTH_CUE" if len(categories) > 1 else (
            "NEG_CUE" if "NEG" in categories else "UNCERT_CUE"
        )

    # Scope heuristic: from immediately after each cue until punctuation,
    # sentence end, or the next cue. This deliberately simple scope rule is
    # auditable and will not reproduce discontinuous BRAT relations.
    cue_starts.sort()
    punctuation = {".", ";", ":", "!", "?", "…"}
    scopes: list[tuple[int, int, str]] = []
    for cue_start, cue_end, category in cue_starts:
        cursor = cue_end
        while cursor < len(tokens) and lines[cursor] == lines[cue_start]:
            if str(tokens[cursor]["token"]) in punctuation:
                break
            if cursor in cue_categories:
                break
            cursor += 1
        if cursor > cue_end:
            scopes.append((cue_end, cursor, category))

    scope_categories: dict[int, set[str]] = {}
    for start, end, category in scopes:
        for index in range(start, end):
            if not cue_categories.get(index):
                scope_categories.setdefault(index, set()).add(category)
    for index, categories in scope_categories.items():
        labels[index] = "MIXED_SCOPE" if len(categories) > 1 else (
            "NEG_SCOPE" if "NEG" in categories else "UNCERT_SCOPE"
        )
    return labels


def split_by_document(
    documents: list[dict[str, Any]], test_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stable held-out-document split; no token from one file crosses splits."""
    ordered = sorted(documents, key=lambda item: item["doc_id"])
    random.Random(SEED).shuffle(ordered)
    test_count = max(1, round(len(ordered) * test_fraction))
    train, test = ordered[test_count:], ordered[:test_count]
    if not train or not test:
        raise ValueError("Need at least two annotated text documents for a grouped split")
    return train, test


def flatten(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    features: list[dict[str, Any]] = []
    labels: list[str] = []
    for doc in documents:
        toks = doc["tokens"]
        for index, token in enumerate(toks):
            features.append(token_features(toks, index))
            labels.append(token["label"])
    return features, labels


def run_benchmark(corpus_dir: Path, output_dir: Path, test_fraction: float = 0.20) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}. Clone the upstream repository "
            "or pass --corpus-dir pointing at NUBes/SAMPLE-001."
        )
    text_files = sorted(corpus_dir.glob("*.txt"))
    documents = [
        load_document(path) for path in text_files
        if path.with_suffix(".ann").exists()
    ]
    documents = [doc for doc in documents if doc["tokens"]]
    if len(documents) < 5:
        raise RuntimeError(
            f"Only {len(documents)} annotated documents found in {corpus_dir}; "
            "expected multiple NUBes SAMPLE-001 text/annotation pairs."
        )

    train_docs, test_docs = split_by_document(documents, test_fraction)
    x_train, y_train = flatten(train_docs)
    x_test, y_test = flatten(test_docs)
    train_counts, test_counts = Counter(y_train), Counter(y_test)

    models = [
        ("majority_dummy", make_pipeline(
            DictVectorizer(sparse=True), DummyClassifier(strategy="most_frequent")
        )),
        ("contextual_linear_svc", make_pipeline(
            DictVectorizer(sparse=True),
            LinearSVC(class_weight="balanced", random_state=SEED, max_iter=10000)
        )),
    ]

    model_metrics: list[dict[str, Any]] = []
    for name, model in models:
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted = list(model.predict(x_test))
        predict_seconds = time.perf_counter() - start
        class_report = classification_report(
            y_test, predicted, labels=LABELS, output_dict=True, zero_division=0
        )
        operator_labels = [label for label in LABELS if label != "O" and test_counts[label] > 0]
        model_metrics.append({
            "model": name,
            "test_tokens": len(y_test),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1_all_labels": float(f1_score(
                y_test, predicted, labels=LABELS, average="macro", zero_division=0
            )),
            "macro_f1_operator_labels_present_in_test": float(f1_score(
                y_test, predicted, labels=operator_labels, average="macro", zero_division=0
            )) if operator_labels else None,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "classification_report": {
                label: {
                    "precision": float(class_report[label]["precision"]),
                    "recall": float(class_report[label]["recall"]),
                    "f1-score": float(class_report[label]["f1-score"]),
                    "support": int(class_report[label]["support"]),
                } for label in LABELS
            },
            "predictions_saved": False,
        })

    # Rule-based Hisab-inspired pipeline: same held-out tokens, no gold labels
    # passed into inference, no training fit required.
    rule_start = time.perf_counter()
    rule_predicted = [
        prediction
        for doc in test_docs
        for prediction in hisab_operator_pipeline(doc["tokens"])
    ]
    rule_predict_seconds = time.perf_counter() - rule_start
    if len(rule_predicted) != len(y_test):
        raise RuntimeError(
            f"Rule pipeline produced {len(rule_predicted)} predictions for {len(y_test)} test tokens"
        )
    rule_report = classification_report(
        y_test, rule_predicted, labels=LABELS, output_dict=True, zero_division=0
    )
    operator_labels = [
        label for label in LABELS if label != "O" and test_counts[label] > 0
    ]
    model_metrics.append({
        "model": "hisab_inspired_lexicon_scope_rules",
        "model_type": "transparent rule-based operator pipeline; classical execution",
        "test_tokens": len(y_test),
        "accuracy": float(accuracy_score(y_test, rule_predicted)),
        "macro_f1_all_labels": float(f1_score(
            y_test, rule_predicted, labels=LABELS, average="macro", zero_division=0
        )),
        "macro_f1_operator_labels_present_in_test": float(f1_score(
            y_test, rule_predicted, labels=operator_labels,
            average="macro", zero_division=0
        )) if operator_labels else None,
        "fit_seconds": 0.0,
        "predict_seconds": rule_predict_seconds,
        "classification_report": {
            label: {
                "precision": float(rule_report[label]["precision"]),
                "recall": float(rule_report[label]["recall"]),
                "f1-score": float(rule_report[label]["f1-score"]),
                "support": int(rule_report[label]["support"]),
            } for label in LABELS
        },
        "predictions_saved": False,
        "uses_gold_annotations_at_inference": False,
        "uses_quantum_circuit": False,
    })

    corpus_counts = Counter(token["label"] for doc in documents for token in doc["tokens"])
    report = {
        "benchmark": "NUBes SAMPLE-001 operator/scope benchmark: symbolic rules vs contextual LinearSVC",
        "task": "identify negation/uncertainty cues and annotation-defined scope tokens",
        "corpus": {
            "name": "NUBes",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "subcorpus": "SAMPLE-001",
            "annotation_protocol_note": (
                "Upstream README reports that SAMPLE-001 was annotated by two annotators and "
                "a referee; other samples were annotated by one annotator."
            ),
            "license": LICENSE,
            "attribution": (
                "Lima Lopez, S.; Perez, N.; Cuadros, M.; Rigau, G. (2020). "
                "NUBes: A Corpus of Negation and Uncertainty in Spanish Clinical Texts. LREC 2020."
            ),
            "source_texts_redistributed": False,
            "source_texts_or_predictions_in_artifact": False,
        },
        "data": {
            "annotated_documents": len(documents),
            "train_documents": len(train_docs),
            "test_documents": len(test_docs),
            "train_tokens": len(y_train),
            "test_tokens": len(y_test),
            "label_order": LABELS,
            "all_corpus_label_counts": {label: corpus_counts[label] for label in LABELS},
            "train_label_counts": {label: train_counts[label] for label in LABELS},
            "test_label_counts": {label: test_counts[label] for label in LABELS},
            "split": "deterministic held-out documents; token-level random splitting avoided",
            "test_fraction_target": test_fraction,
            "seed": SEED,
        },
        "models": model_metrics,
        "software": {"python": platform.python_version(), "scikit_learn": version("scikit-learn")},
        "interpretation": {
            "does_not_test_quantum_advantage": True,
            "does_not_execute_hisab_quantum_circuit": True,
            "includes_hisab_inspired_rule_pipeline": True,
            "scope_definition": (
                "Scope or DiscScope target span from a Neg* cue is NEG_SCOPE; from an Uncert* "
                "cue is UNCERT_SCOPE. Cue tokens are labeled separately."
            ),
            "limitations": [
                "This evaluates one language-processing component, not end-to-end semantic inference.",
                "NUBes is biomedical Spanish; transfer to general Spanish must be tested separately.",
                "This is not a full competitive NLP benchmark.",
                "The CC BY-SA 3.0 Spain source license must be respected; source text is not copied to this repository or artifacts.",
                "The rule-based pipeline is an initial classical Hisab-inspired operator mapping; it is not the fixed Qiskit circuit and does not yet validate the full Hisab architecture.",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nubes_scope_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# NUBes SAMPLE-001: operator and scope tagging",
        "",
        "Scope: real annotated Spanish biomedical text; token-level detection of negation/uncertainty cues and annotated spans.",
        "Not a quantum-advantage experiment. No source sentences or token predictions are included in the report artifact.",
        "",
        f"- Annotated documents: {len(documents)}",
        f"- Held-out train/test documents: {len(train_docs)} / {len(test_docs)}",
        f"- Train/test tokens: {len(y_train)} / {len(y_test)}",
        f"- Test label distribution: {json.dumps({label: test_counts[label] for label in LABELS}, ensure_ascii=False)}",
        "",
        "## Results",
        "",
        "| Model | Accuracy | Macro-F1 (all labels) | Macro-F1 (operator labels present in test) | Fit (s) | Predict (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in model_metrics:
        score = item["macro_f1_operator_labels_present_in_test"]
        score_text = f"{score:.4f}" if score is not None else "n/a"
        lines.append(
            f"| {item['model']} | {item['accuracy']:.4f} | {item['macro_f1_all_labels']:.4f} "
            f"| {score_text} | {item['fit_seconds']:.3f} | {item['predict_seconds']:.3f} |"
        )
    lines.extend([
        "",
        "## Attribution and limitations",
        "",
        f"- Source: {UPSTREAM_REPOSITORY}",
        f"- License: {LICENSE}",
        "- Upstream documentation reports two annotators and a referee for SAMPLE-001.",
        "- The corpus is clinical Spanish; general-domain transfer is untested.",
        "- This benchmark does not execute the Hisab quantum circuit and does not establish quantum computational advantage.",
        "- Source text and token-level predictions are not redistributed by this workflow.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("external/NUBes-negation-uncertainty-biomedical-corpus/NUBes/SAMPLE-001"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/nubes_operator_scope"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be between 0.05 and 0.5")
    report = run_benchmark(args.corpus_dir, args.output_dir, args.test_fraction)
    print(json.dumps({
        "benchmark": report["benchmark"],
        "annotated_documents": report["data"]["annotated_documents"],
        "train_tokens": report["data"]["train_tokens"],
        "test_tokens": report["data"]["test_tokens"],
        "models": [
            {
                "model": model["model"],
                "accuracy": model["accuracy"],
                "macro_f1_all_labels": model["macro_f1_all_labels"],
                "macro_f1_operator_labels_present_in_test": model["macro_f1_operator_labels_present_in_test"],
            } for model in report["models"]
        ],
        "does_not_test_quantum_advantage": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
