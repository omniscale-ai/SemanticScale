#!/usr/bin/env python
"""Script 01: Load and merge SH5 data into unified trace records."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all

if __name__ == "__main__":
    force = "--force" in sys.argv
    output_path = "data/traces.jsonl"

    if os.path.exists(output_path) and not force:
        print(f"{output_path} already exists. Use --force to rerun.")
        sys.exit(0)

    traces = load_all()
    print("Stage A complete.")
