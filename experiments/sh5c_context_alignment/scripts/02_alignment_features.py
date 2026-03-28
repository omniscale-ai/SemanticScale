#!/usr/bin/env python
"""Stage B: Compute alignment features from merged traces."""

import sys
from pathlib import Path
import yaml

WORKING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKING_DIR))

from src.alignment import process_all_traces

INPUT = WORKING_DIR / "../../data/sh5c" / "merged_traces.jsonl"
OUTPUT = WORKING_DIR / "../../data/sh5c" / "alignment_features.jsonl"


def main():
    force = "--force" in sys.argv

    if OUTPUT.exists() and not force:
        print(f"Output exists: {OUTPUT} (use --force to rerun)")
        return

    if not INPUT.exists():
        print(f"ERROR: Input not found: {INPUT}")
        print("Run 01_load_and_extract.py first.")
        sys.exit(1)

    with open(WORKING_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    cap = cfg.get("alignment", {}).get("diversity_ratio_cap", 100.0)
    process_all_traces(INPUT, OUTPUT, diversity_ratio_cap=cap)


if __name__ == "__main__":
    main()
