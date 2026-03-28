#!/usr/bin/env python3
"""Stage C: Compute steering vector in the generative model's representation space.

This is the core SH2 stage. Processes ~17,400 spans through the generative model,
extracts per-layer hidden states, and computes per-layer steering vectors as
micro_centroid - macro_centroid.

GPU required. Expected runtime: 30-60 minutes on A100.

Output: data/steering_vectors.npz
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

# TODO: Engineer implements the full pipeline here
# See PLAN.md Stage C and src/steering.py for details
#
# Key steps:
# 1. Load SH0 spans + SH1 splits -> filter to train split macro/micro spans
# 2. Load generative model (with quantization if configured)
# 3. Extract hidden states at candidate layers for all spans
# 4. Compute per-layer steering vectors
# 5. Select best layer via validation
# 6. Save to data/steering_vectors.npz


def main():
    parser = argparse.ArgumentParser(description="Stage C: Compute steering vector")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "steering_vectors.npz"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Engineer agent should implement this script. "
        "See PLAN.md Stage C and src/steering.py for the specification."
    )


if __name__ == "__main__":
    main()
