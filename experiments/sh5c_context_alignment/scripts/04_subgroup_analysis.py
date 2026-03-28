#!/usr/bin/env python
"""Stage D: Run subgroup analyses."""

import sys
from pathlib import Path

WORKING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKING_DIR))

from src.subgroups import run_subgroup_analysis

INPUT = WORKING_DIR / "data" / "alignment_features.jsonl"
OUTPUT_DIR = WORKING_DIR / "data" / "results"


def main():
    force = "--force" in sys.argv

    if not INPUT.exists():
        print(f"ERROR: Input not found: {INPUT}")
        print("Run 02_alignment_features.py first.")
        sys.exit(1)

    output_file = OUTPUT_DIR / "subgroups.json"
    if output_file.exists() and not force:
        print(f"Output exists: {output_file} (use --force to rerun)")
        return

    run_subgroup_analysis(INPUT, OUTPUT_DIR)


if __name__ == "__main__":
    main()
