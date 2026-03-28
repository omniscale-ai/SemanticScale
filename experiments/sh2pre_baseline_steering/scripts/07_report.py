#!/usr/bin/env python3
"""Stage G: Generate visualization and auto-report.

Produces 6 figures and a markdown report with verdict.

Output: reports/SH2_results.md, reports/figures/*.png
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

# TODO: Engineer implements this
# See PLAN.md Stage G and src/visualization.py
#
# Key steps:
# 1. Load evaluation results from data/results/
# 2. Generate all 6 figures
# 3. Generate markdown report with verdict
# 4. Save everything to reports/


def main():
    parser = argparse.ArgumentParser(description="Stage G: Report")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["reports_dir"]) / "SH2_results.md"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    Path(config["figures_dir"]).mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Engineer agent should implement this script. "
        "See PLAN.md Stage G for the specification."
    )


if __name__ == "__main__":
    main()
