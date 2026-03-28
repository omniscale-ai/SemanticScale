#!/usr/bin/env python3
"""Stage B: Compute SLoD evaluation axis from SH1 data.

Reuses proven SH5d code. This axis is for EVALUATION only (measuring SLoD shift
in generated outputs via SciBERT embedding), NOT for steering.

Output: data/eval_slod_axis.npz, reports/figures/slod_axis_validation.png
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, save_json
from src.slod_axis import compute_centroid_axis, compute_lda_axis, validate_axis


def main():
    parser = argparse.ArgumentParser(description="Stage B: Compute SLoD evaluation axis")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "eval_slod_axis.npz"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load SH1 data
    print("Loading SH1 embeddings and splits...")
    sh1_data = np.load(config["sh1_embeddings"])
    embeddings = sh1_data["embeddings"]
    labels = sh1_data["labels"]
    with open(config["sh1_splits"]) as f:
        splits = json.load(f)

    train_idx = np.array(splits["train"])
    test_idx = np.array(splits["test"])

    train_emb = embeddings[train_idx]
    train_labels = labels[train_idx]
    test_emb = embeddings[test_idx]
    test_labels = labels[test_idx]

    print(f"Train: {len(train_idx)} spans, Test: {len(test_idx)} spans")

    # Compute axes from train split only
    print("\nComputing centroid axis...")
    centroid_axis = compute_centroid_axis(train_emb, train_labels)

    print("\nComputing LDA axis...")
    lda_axis = compute_lda_axis(train_emb, train_labels)

    # Validate on test split
    print("\nValidating centroid axis on test set...")
    centroid_stats = validate_axis(test_emb, test_labels, centroid_axis)

    print("\nValidating LDA axis on test set...")
    lda_stats = validate_axis(test_emb, test_labels, lda_axis)

    # Save
    np.savez(
        str(output_path),
        centroid_axis=centroid_axis,
        lda_axis=lda_axis,
        centroid_validation=json.dumps(centroid_stats),
        lda_validation=json.dumps(lda_stats),
    )
    print(f"\nSaved to {output_path}")

    # Save validation stats as JSON too
    stats_path = Path(config["data_dir"]) / "results" / "slod_axis_validation.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    save_json({"centroid": centroid_stats, "lda": lda_stats}, str(stats_path))

    print(f"\nCentroid axis: Cohen's d = {centroid_stats['cohens_d']:.3f}")
    print(f"LDA axis: Cohen's d = {lda_stats['cohens_d']:.3f}")
    print(f"Ordering correct (centroid): {centroid_stats['separation_correct']}")


if __name__ == "__main__":
    main()
