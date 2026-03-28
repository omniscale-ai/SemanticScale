#!/usr/bin/env python
"""Script 03: Extract features from transition matrices."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data_loader import load_jsonl
from src.features import extract_all_features, extract_aggregated_features, save_features

if __name__ == "__main__":
    force = "--force" in sys.argv
    output_path = "data/features.jsonl"

    if os.path.exists(output_path) and not force:
        print(f"{output_path} already exists. Use --force to rerun.")
        sys.exit(0)

    print("Loading matrices...")
    data = np.load("data/transition_matrices.npz")
    hard_matrices = data["hard_matrices"]
    soft_matrices = data["soft_matrices"]
    agg_hard = data["agg_hard_matrices"]
    agg_soft = data["agg_soft_matrices"]
    print(f"  Hard: {hard_matrices.shape}, Soft: {soft_matrices.shape}")
    print(f"  Agg Hard: {agg_hard.shape}, Agg Soft: {agg_soft.shape}")

    print("Loading metadata...")
    metadata = load_jsonl("data/transition_matrices_meta.jsonl")
    agg_metadata = load_jsonl("data/agg_metadata.jsonl")
    print(f"  {len(metadata)} trace records, {len(agg_metadata)} question records")

    print("Extracting per-trace features...")
    features = extract_all_features(hard_matrices, soft_matrices, metadata)
    save_features(features, "data/features.jsonl")

    print("Extracting per-question features...")
    features_agg = extract_aggregated_features(agg_hard, agg_soft, agg_metadata)
    save_features(features_agg, "data/features_agg.jsonl")

    # Quick stats
    print(f"\nFeature summary:")
    print(f"  Per-trace features: {len(features)} records, {len(features[0])} fields each")
    print(f"  Per-question features: {len(features_agg)} records, {len(features_agg[0])} fields each")

    print("Stage C complete.")
