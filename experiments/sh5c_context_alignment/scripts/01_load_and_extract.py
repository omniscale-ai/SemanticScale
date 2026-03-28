#!/usr/bin/env python
"""Stage A: Load SH5 + SH3 data and build merged traces."""

import sys
from pathlib import Path

WORKING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKING_DIR))

from src.data_loader import build_merged_traces, save_merged_traces

OUTPUT = WORKING_DIR / "../../data/sh5c" / "merged_traces.jsonl"


def main():
    force = "--force" in sys.argv

    if OUTPUT.exists() and not force:
        print(f"Output exists: {OUTPUT} (use --force to rerun)")
        return

    df = build_merged_traces(WORKING_DIR)
    save_merged_traces(df, OUTPUT)


if __name__ == "__main__":
    main()
