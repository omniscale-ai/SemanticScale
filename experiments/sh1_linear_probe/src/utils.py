"""Shared utilities for SH1: config loading, data I/O, splits, logging."""

import json
import logging
from pathlib import Path

import jsonlines
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


LABEL_MAP = {"macro": 0, "meso": 1, "micro": 2}
LABEL_NAMES = ["macro", "meso", "micro"]


def load_config(path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_spans(jsonl_path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of span dicts."""
    spans = []
    with jsonlines.open(jsonl_path, mode="r") as reader:
        for obj in reader:
            spans.append(obj)
    logging.info(f"Loaded {len(spans)} spans from {jsonl_path}")
    return spans


def create_splits(
    n_samples: int,
    labels: list | np.ndarray,
    ratios: list[float],
    seed: int,
    save_path: str | Path | None = None,
) -> dict[str, list[int]]:
    """Create stratified train/val/test splits and optionally save to JSON.

    Args:
        n_samples: Total number of samples.
        labels: Array-like of labels for stratification.
        ratios: [train_ratio, val_ratio, test_ratio] summing to 1.0.
        seed: Random seed for reproducibility.
        save_path: If provided, save split indices to this JSON file.

    Returns:
        Dict with keys 'train', 'val', 'test', each a list of integer indices.
    """
    assert len(ratios) == 3, "Expected [train, val, test] ratios"
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    indices = np.arange(n_samples)
    labels = np.array(labels)

    train_ratio, val_ratio, test_ratio = ratios
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, valtest_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=labels[indices],
        random_state=seed,
    )
    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=relative_test_ratio,
        stratify=labels[valtest_idx],
        random_state=seed,
    )

    splits = {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }

    logging.info(
        f"Splits created: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(splits, f)
        logging.info(f"Splits saved to {save_path}")

    return splits


def load_splits(path: str | Path) -> dict[str, list[int]]:
    """Load saved split indices from JSON."""
    with open(path, "r") as f:
        splits = json.load(f)
    logging.info(
        f"Loaded splits: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return splits


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
