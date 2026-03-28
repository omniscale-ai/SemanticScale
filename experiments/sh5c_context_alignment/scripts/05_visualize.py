#!/usr/bin/env python
"""Stage E: Generate all visualization figures."""

import sys
from pathlib import Path
import yaml

WORKING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKING_DIR))

from src.visualization import generate_all_plots

INPUT = WORKING_DIR / "../../data/sh5c" / "alignment_features.jsonl"
RESULTS_DIR = WORKING_DIR / "../../data/sh5c" / "results"
FIGURES_DIR = WORKING_DIR / "reports" / "figures"


def main():
    if not INPUT.exists():
        print(f"ERROR: Input not found: {INPUT}")
        print("Run 02_alignment_features.py first.")
        sys.exit(1)

    with open(WORKING_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    generate_all_plots(INPUT, RESULTS_DIR, FIGURES_DIR, cfg)


if __name__ == "__main__":
    main()
