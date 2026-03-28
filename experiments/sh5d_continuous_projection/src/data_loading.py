"""Data loading utilities for SH5d pipeline."""
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_cot_steps(path: str) -> pd.DataFrame:
    """Load CoT step tags from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} CoT steps from {path}")
    print(f"  Unique traces: {df.groupby(['question_id','condition']).ngroups}")
    return df


def load_answer_scores(path: str) -> pd.DataFrame:
    """Load answer quality scores from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} answer scores from {path}")
    return df


def load_jump_metrics(path: str) -> pd.DataFrame:
    """Load SH5 jump metrics from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} jump metrics from {path}")
    return df


def load_sh1_data(emb_path: str, splits_path: str):
    """Load SH1 embeddings and splits.

    Returns:
        embeddings: (N, 768) array
        labels: (N,) array (0=macro, 1=meso, 2=micro)
        splits: dict with train/val/test index lists
    """
    data = np.load(emb_path)
    embeddings = data['embeddings']
    labels = data['labels']
    with open(splits_path) as f:
        splits = json.load(f)
    print(f"Loaded SH1: {embeddings.shape[0]} embeddings, {embeddings.shape[1]}-dim")
    print(f"  Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  Splits: { {k: len(v) for k, v in splits.items()} }")
    return embeddings, labels, splits
