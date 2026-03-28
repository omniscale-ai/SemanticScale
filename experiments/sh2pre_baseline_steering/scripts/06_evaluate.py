#!/usr/bin/env python3
"""Stage F: Evaluate baseline vs steered answers.

Measures SLoD shift, surface metrics, and factuality preservation.

Output: data/results/evaluation_results.json
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

# TODO: Engineer implements this
# See PLAN.md Stage F and src/evaluate.py
#
# Key steps:
# 1. Load baseline and steered answers
# 2. Embed all answers with SciBERT
# 3. Project onto SLoD axis -> compute shift (H1)
# 4. Compute surface metrics for all answers (H2)
# 5. Compute token-F1 vs gold answers (H3)
# 6. Compile results and determine verdict
# 7. Save to data/results/evaluation_results.json


def main():
    parser = argparse.ArgumentParser(description="Stage F: Evaluate")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "results" / "evaluation_results.json"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Engineer agent should implement this script. "
        "See PLAN.md Stage F for the specification."
    )


if __name__ == "__main__":
    main()
