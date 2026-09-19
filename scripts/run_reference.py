#!/usr/bin/env python3
"""Run the canonical Aer experiment and write an auditable JSON record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hisab_qnlp import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/reference/aer_reference.json"))
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args()

    record = run_experiment(ExperimentConfig(shots=args.shots))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

