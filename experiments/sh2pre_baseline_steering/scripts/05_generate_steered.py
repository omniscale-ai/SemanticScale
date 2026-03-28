#!/usr/bin/env python3
"""Stage E: Generate steered answers (micro and macro directions).

Uses the steering vector from Stage C to generate answers at different
abstraction levels. Includes alpha sweep for strength selection.

GPU required. Expected runtime: 30-60 minutes on A100.

Output: data/steered_answers.jsonl, data/results/alpha_sweep.json
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

# TODO: Engineer implements this
# See PLAN.md Stage E and src/steering.py
#
# Key steps:
# 1. Load steering vectors from data/steering_vectors.npz
# 2. Load questions
# 3. Alpha sweep on validation subset (first 50 questions)
# 4. Select best alpha
# 5. Generate steered answers for all 500 questions x 2 directions
# 6. Save outputs


def main():
    parser = argparse.ArgumentParser(description="Stage E: Generate steered answers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "steered_answers.jsonl"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Engineer agent should implement this script. "
        "See PLAN.md Stage E for the specification."
    )


if __name__ == "__main__":
    main()
