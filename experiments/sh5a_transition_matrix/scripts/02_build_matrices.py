#!/usr/bin/env python
"""Script 02: Build per-trace transition matrices (hard + soft)."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_jsonl
from src.transition_matrix import build_all_matrices, aggregate_per_question, save_matrices

if __name__ == "__main__":
    force = "--force" in sys.argv
    output_path = "data/transition_matrices.npz"

    if os.path.exists(output_path) and not force:
        print(f"{output_path} already exists. Use --force to rerun.")
        sys.exit(0)

    print("Loading traces...")
    traces = load_jsonl("data/traces.jsonl")
    print(f"  {len(traces)} traces loaded")

    print("Building transition matrices...")
    result = build_all_matrices(traces)
    print(f"  Hard matrices shape: {result['hard_matrices'].shape}")
    print(f"  Soft matrices shape: {result['soft_matrices'].shape}")

    print("Aggregating per question...")
    agg_hard, qids, agg_hard_meta = aggregate_per_question(result["hard_matrices"], result["metadata"])
    agg_soft, _, agg_soft_meta = aggregate_per_question(result["soft_matrices"], result["metadata"])
    print(f"  {len(qids)} questions, aggregated matrices shape: {agg_hard.shape}")

    print("Saving...")
    save_matrices(result, agg_hard, agg_soft, qids)

    # Also save aggregated metadata
    with open("data/agg_metadata.jsonl", "w") as f:
        for m in agg_hard_meta:
            f.write(json.dumps(m) + "\n")
    print("  Saved aggregated metadata to data/agg_metadata.jsonl")

    print("Stage B complete.")
