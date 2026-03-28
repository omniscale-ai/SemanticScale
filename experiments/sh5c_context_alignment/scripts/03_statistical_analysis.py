#!/usr/bin/env python
"""Stage C: Run all statistical analyses."""

import sys
from pathlib import Path
import yaml

WORKING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKING_DIR))

from src.statistics import run_all_statistics

INPUT = WORKING_DIR / "../../data/sh5c" / "alignment_features.jsonl"
OUTPUT_DIR = WORKING_DIR / "../../data/sh5c" / "results"


def main():
    force = "--force" in sys.argv

    if not INPUT.exists():
        print(f"ERROR: Input not found: {INPUT}")
        print("Run 02_alignment_features.py first.")
        sys.exit(1)

    # Check if all outputs exist
    outputs = ["correlations.json", "wilcoxon.json", "logistic.json", "partial_correlations.json"]
    if all((OUTPUT_DIR / f).exists() for f in outputs) and not force:
        print(f"All outputs exist in {OUTPUT_DIR} (use --force to rerun)")
        return

    with open(WORKING_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    run_all_statistics(INPUT, OUTPUT_DIR, cfg)


if __name__ == "__main__":
    main()
