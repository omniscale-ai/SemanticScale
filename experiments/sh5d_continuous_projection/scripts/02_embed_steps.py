#!/usr/bin/env python3
"""Stage B: Embed all CoT steps with SciBERT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.embedding import embed_texts

def main():
    parser = argparse.ArgumentParser(description="Stage B: SciBERT embedding")
    parser.add_argument('--force', action='store_true', help='Force recompute')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()

    data_dir = Path('data')
    output_path = data_dir / 'step_embeddings.npz'

    if output_path.exists() and not args.force:
        print("Stage B output exists, skipping. Use --force to rerun.")
        d = np.load(output_path, allow_pickle=True)
        print(f"  Existing embeddings shape: {d['embeddings'].shape}")
        return

    # Load steps
    steps = pd.read_parquet(data_dir / 'cot_steps.parquet')
    texts = steps['step_text'].tolist()
    print(f"Embedding {len(texts)} step texts...")

    embeddings = embed_texts(
        texts,
        model_name='allenai/scibert_scivocab_uncased',
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    assert embeddings.shape[0] == len(texts), f"Shape mismatch: {embeddings.shape[0]} vs {len(texts)}"
    assert embeddings.shape[1] == 768, f"Unexpected dim: {embeddings.shape[1]}"

    # Save with metadata
    np.savez(
        output_path,
        embeddings=embeddings,
        question_ids=steps['question_id'].values,
        conditions=steps['condition'].values,
        step_indices=steps['step_index'].values
    )
    print(f"Saved: {output_path} ({embeddings.shape})")
    print("Stage B: DONE")

if __name__ == '__main__':
    main()
