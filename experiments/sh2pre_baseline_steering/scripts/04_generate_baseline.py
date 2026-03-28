#!/usr/bin/env python3
"""Stage D: Generate baseline answers (no steering).

Generates answers to 500 QASPER questions using the generative model
without any activation steering. These serve as the control condition.

GPU required. Expected runtime: ~15 minutes on A100.

Output: data/baseline_answers.jsonl
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

# TODO: Engineer implements this
# See PLAN.md Stage D and src/steering.py generate_with_steering()
#
# Key steps:
# 1. Load questions from data/sh5/selected_questions.jsonl
# 2. Load generative model
# 3. For each question: format prompt (question + gold_evidence as context)
# 4. Generate answer (temperature=0.0, no steering)
# 5. Save to data/baseline_answers.jsonl


def main():
    parser = argparse.ArgumentParser(description="Stage D: Generate baseline answers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "baseline_answers.jsonl"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Engineer agent should implement this script. "
        "See PLAN.md Stage D for the specification."
    )


if __name__ == "__main__":
    main()
