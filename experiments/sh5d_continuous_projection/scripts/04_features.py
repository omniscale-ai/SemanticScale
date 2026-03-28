#!/usr/bin/env python3
"""Stage D: Feature engineering — compute per-trace embedding features."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.features import compute_all_trace_features

def main():
    parser = argparse.ArgumentParser(description="Stage D: Features")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = Path('data')
    output_path = data_dir / 'trace_features.parquet'

    if output_path.exists() and not args.force:
        print("Stage D output exists, skipping.")
        return

    # Load inputs
    steps = pd.read_parquet(data_dir / 'cot_steps.parquet')
    step_embs = np.load(data_dir / 'step_embeddings.npz')
    embeddings = step_embs['embeddings']
    slod_data = np.load(data_dir / 'slod_axis.npz', allow_pickle=True)
    slod_axis = slod_data['centroid_axis']
    scores = pd.read_parquet(data_dir / 'answer_scores.parquet')

    print(f"Steps: {len(steps)}, Embeddings: {embeddings.shape}, Scores: {len(scores)}")

    # Ensure steps index aligns with embeddings
    assert len(steps) == embeddings.shape[0], "Steps and embeddings count mismatch!"

    # Reset index to ensure alignment
    steps = steps.reset_index(drop=True)

    features_df = compute_all_trace_features(steps, embeddings, slod_axis, scores)
    features_df.to_parquet(output_path, index=False)

    # Summary
    print(f"\nFeature summary:")
    feature_cols = [c for c in features_df.columns if c not in
                    ['question_id', 'condition', 'n_steps', 'answer_token_f1', 'attribution_f1']]
    for col in feature_cols:
        valid = features_df[col].dropna()
        print(f"  {col}: mean={valid.mean():.4f}, std={valid.std():.4f}, n_valid={len(valid)}")

    print(f"\nSaved: {output_path}")
    print("Stage D: DONE")

if __name__ == '__main__':
    main()
