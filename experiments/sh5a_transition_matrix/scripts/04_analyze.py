#!/usr/bin/env python
"""Script 04: Run statistical analyses."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import run_full_analysis

if __name__ == "__main__":
    force = "--force" in sys.argv
    output_path = "data/results/summary.json"

    if os.path.exists(output_path) and not force:
        print(f"{output_path} already exists. Use --force to rerun.")
        sys.exit(0)

    summary = run_full_analysis()
    print("\nStage D complete.")
