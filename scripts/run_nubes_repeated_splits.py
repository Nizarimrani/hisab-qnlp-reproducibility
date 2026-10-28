#!/usr/bin/env python3
"""Repeated document-grouped evaluation for the NUBes operator/scope benchmark.

Runs the existing benchmark on multiple seeded document-level splits and writes
aggregate statistics only; it does not save source sentences or token predictions.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from scripts.run_nubes_operator_scope_benchmark import SEED, run_benchmark
except ModuleNotFoundError:
    from run_nubes_operator_scope_benchmark import SEED, run_benchmark


METRIC_PATHS = {
    "accuracy": ("accuracy",),
    "macro_f1_all_labels": ("macro_f1_all_labels",),
    "macro_f1_operator_labels_present_in_test": (
        "macro_f1_operator_labels_present_in_test",
    ),
    "cue_exact_span_f1": ("exact_span_metrics", "cue_spans", "f1"),
    "scope_exact_span_f1": ("exact_span_metrics", "scope_spans", "f1"),
}


def _nested_get(item: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = item
    for key in path:
        value = value[key]
    return value


def run_repeated_benchmark(
    corpus_dir: Path,
    output_dir: Path,
    seeds: list[int],
    test_fraction: float = 0.20,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for repeated-split evaluation")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        report = run_benchmark(
            corpus_dir=corpus_dir,
            output_dir=seed_dir,
            test_fraction=test_fraction,
            seed=seed,
        )
        by_model: dict[str, Any] = {}
        for model in report["models"]:
            by_model[model["model"]] = {
                key: _nested_get(model, path)
                for key, path in METRIC_PATHS.items()
            }
        run_records.append({
            "seed": seed,
            "train_documents": report["data"]["train_documents"],
            "test_documents": report["data"]["test_documents"],
            "train_tokens": report["data"]["train_tokens"],
            "test_tokens": report["data"]["test_tokens"],
            "models": by_model,
            "operator_composition_validation": {
                "typed_operators": report["operator_composition_validation"][
                    "typed_operators_extracted_from_test_text"
                ],
                "scoped_target_tokens": report["operator_composition_validation"][
                    "annotated_scope_tokens_evaluated"
                ],
                "circuit_checks": report["operator_composition_validation"][
                    "circuit_symbolic_checks"
                ],
                "circuit_agreements": report["operator_composition_validation"][
                    "circuit_symbolic_agreements"
                ],
            },
        })

    model_names = sorted(run_records[0]["models"])
    summary: dict[str, Any] = {}
    for model_name in model_names:
        metric_summary: dict[str, Any] = {}
        for metric_name in METRIC_PATHS:
            values = [
                record["models"][model_name][metric_name]
                for record in run_records
                if record["models"][model_name][metric_name] is not None
            ]
            metric_summary[metric_name] = {
                "mean": statistics.mean(values) if values else None,
                "sample_standard_deviation": (
                    statistics.stdev(values) if len(values) >= 2 else None
                ),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "n": len(values),
            }
        summary[model_name] = metric_summary

    report: dict[str, Any] = {
        "benchmark": "NUBes SAMPLE-001 repeated document-grouped operator/scope evaluation",
        "seed_values": seeds,
        "number_of_splits": len(seeds),
        "test_fraction_target": test_fraction,
        "split_unit": "whole annotated documents; no document crosses train/test within a split",
        "corpus": {
            "name": "NUBes",
            "subcorpus": "SAMPLE-001",
            "upstream_repository": "https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus",
            "source_texts_redistributed": False,
            "per_token_predictions_redistributed": False,
        },
        "individual_splits": run_records,
        "aggregate_metrics": summary,
        "elapsed_seconds": time.perf_counter() - overall_start,
        "interpretation": {
            "repeated_random_grouped_holdouts_not_independent_cross_validation": True,
            "seeds_are_distinct_but_test_documents_may_overlap_between_splits": True,
            "quantum_advantage_tested": False,
            "scope_f1_definition": (
                "Strict exact F1 over maximal contiguous token-tag runs; discontinuous BRAT target spans "
                "are not yet evaluated as linked multi-part units."
            ),
            "note": (
                "Means and sample standard deviations quantify sensitivity to these sampled document "
                "splits only. With a small number of annotated documents and overlapping holdouts, they "
                "must not be interpreted as independent-sample confidence intervals."
            ),
        },
    }
    (output_dir / "repeated_split_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# NUBes: repeated document-grouped operator/scope evaluation",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Number of splits: {len(seeds)}",
        f"- Target held-out fraction per split: {test_fraction:.2f}",
        f"- Elapsed time: {report['elapsed_seconds']:.2f} seconds",
        "",
        "Documents, not tokens, are assigned to train/test within each split. Different random holdouts can overlap across seeds; this is repeated random holdout evaluation, not independent cross-validation.",
        "",
        "## Mean and sample standard deviation",
        "",
        "| System | Metric | Mean | Sample SD | Min | Max | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in model_names:
        for metric_name in METRIC_PATHS:
            values = summary[model_name][metric_name]
            def fmt(value: Any) -> str:
                return "n/a" if value is None else f"{value:.4f}"
            lines.append(
                f"| {model_name} | {metric_name} | {fmt(values['mean'])} "
                f"| {fmt(values['sample_standard_deviation'])} "
                f"| {fmt(values['minimum'])} | {fmt(values['maximum'])} | {values['n']} |"
            )
    lines.extend([
        "",
        "## Split audit",
        "",
        "| Seed | Train docs | Test docs | Train tokens | Test tokens | Typed operators | Scope target tokens | Circuit agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in run_records:
        comp = record["operator_composition_validation"]
        agreement = f"{comp['circuit_agreements']}/{comp['circuit_checks']}"
        lines.append(
            f"| {record['seed']} | {record['train_documents']} | {record['test_documents']} "
            f"| {record['train_tokens']} | {record['test_tokens']} "
            f"| {comp['typed_operators']} | {comp['scoped_target_tokens']} | {agreement} |"
        )
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- Means and sample SDs quantify how this benchmark changes under these sampled document splits. The held-out documents may overlap between seeds, so the splits are not independent and the SD is not a confidence interval.",
        "- NUBes SAMPLE-001 is clinical Spanish; this is not a general-domain Spanish evaluation.",
        "- Exact-scope metrics currently treat contiguous token-tag runs as spans; linked discontinuous BRAT scope segments are not yet scored as one entity.",
        "- The Qiskit circuit uses deterministic X gates and classical statevector simulation. Symbolic/circuit agreement checks implementation consistency, not linguistic correctness or quantum advantage.",
        "- The report contains aggregate metrics only; it does not redistribute source text or per-token predictions.",
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/nubes_repeated_splits"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--seeds",
        type=str,
        default="20261028,20261029,20261030,20261031,20261032",
        help="Comma-separated distinct integer seeds (at least two).",
    )
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be between 0.05 and 0.5")
    report = run_repeated_benchmark(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        seeds=seeds,
        test_fraction=args.test_fraction,
    )
    print(json.dumps({
        "benchmark": report["benchmark"],
        "seed_values": report["seed_values"],
        "aggregate_metrics": report["aggregate_metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
        "quantum_advantage_tested": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
