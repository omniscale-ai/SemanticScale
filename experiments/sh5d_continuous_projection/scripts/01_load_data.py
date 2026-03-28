#!/usr/bin/env python3
"""Stage A: Load and validate all input data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from src.data_loading import load_cot_steps, load_answer_scores, load_jump_metrics, load_sh1_data

def main():
    parser = argparse.ArgumentParser(description="Stage A: Load data")
    parser.add_argument('--force', action='store_true', help='Force recompute')
    args = parser.parse_args()

    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    output_steps = data_dir / 'cot_steps.parquet'
    output_scores = data_dir / 'answer_scores.parquet'

    if output_steps.exists() and output_scores.exists() and not args.force:
        print("Stage A outputs exist, skipping. Use --force to rerun.")
        return

    # Load all data
    steps = load_cot_steps('data_sh5/cot_slod_tags.jsonl')
    scores = load_answer_scores('data_sh5/answer_scores.jsonl')
    jumps = load_jump_metrics('data_sh5/jump_metrics.jsonl')
    embeddings, labels, splits = load_sh1_data(
        'data_sh1/embeddings/scibert_length_matched.npz',
        'data_sh1/splits.json'
    )

    # Validate
    assert len(steps) == 6101, f"Expected 6101 steps, got {len(steps)}"
    assert len(scores) == 2000, f"Expected 2000 scores, got {len(scores)}"
    assert embeddings.shape == (37278, 768), f"Unexpected SH1 shape: {embeddings.shape}"

    # Check merge: steps should map to answer scores
    trace_keys = steps.groupby(['question_id', 'condition']).size().reset_index()
    merged = trace_keys.merge(scores, on=['question_id', 'condition'], how='inner')
    print(f"Traces with answer scores: {len(merged)} / {len(trace_keys)}")

    # Save
    steps.to_parquet(output_steps, index=False)
    scores.to_parquet(output_scores, index=False)
    jumps.to_parquet(data_dir / 'jump_metrics.parquet', index=False)
    print(f"Saved: {output_steps}, {output_scores}")
    print("Stage A: DONE")

if __name__ == '__main__':
    main()
