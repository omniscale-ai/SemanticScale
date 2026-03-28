#!/usr/bin/env python3
"""Stage C: Compute SLoD axis from SH1 training data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from src.slod_axis import compute_centroid_axis, compute_lda_axis, project_onto_axis, validate_axis
from src.visualization import plot_slod_axis_validation

def main():
    parser = argparse.ArgumentParser(description="Stage C: SLoD axis")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = Path('data')
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / 'slod_axis.npz'

    if output_path.exists() and not args.force:
        print("Stage C output exists, skipping.")
        return

    # Load SH1 data
    sh1 = np.load('data_sh1/embeddings/scibert_length_matched.npz')
    embeddings = sh1['embeddings']
    labels = sh1['labels']
    with open('data_sh1/splits.json') as f:
        splits = json.load(f)

    train_idx = np.array(splits['train'])
    test_idx = np.array(splits['test'])

    train_emb = embeddings[train_idx]
    train_labels = labels[train_idx]
    test_emb = embeddings[test_idx]
    test_labels = labels[test_idx]

    # Compute axes from TRAIN split only
    print("Computing centroid axis from train split...")
    centroid_axis = compute_centroid_axis(train_emb, train_labels)
    print("Computing LDA axis from train split...")
    lda_axis = compute_lda_axis(train_emb, train_labels)

    # Validate on TEST split
    print("\n--- Centroid axis validation (test set) ---")
    centroid_stats = validate_axis(test_emb, test_labels, centroid_axis)
    print("\n--- LDA axis validation (test set) ---")
    lda_stats = validate_axis(test_emb, test_labels, lda_axis)

    # Cosine similarity between the two axes
    cos_sim = float(np.dot(centroid_axis, lda_axis))
    print(f"\nCosine similarity between centroid and LDA axes: {cos_sim:.4f}")

    # Save
    np.savez(
        output_path,
        centroid_axis=centroid_axis,
        lda_axis=lda_axis,
        centroid_stats=json.dumps(centroid_stats),
        lda_stats=json.dumps(lda_stats),
        axis_cosine_similarity=cos_sim
    )

    # Validation plot
    test_projections = project_onto_axis(test_emb, centroid_axis)
    plot_slod_axis_validation(
        test_projections, test_labels,
        str(figures_dir / 'slod_axis_validation.png')
    )

    print(f"\nSaved: {output_path}")
    print("Stage C: DONE")

if __name__ == '__main__':
    main()
