"""Stage C: Extract features from 3x3 transition matrices."""

import json
from pathlib import Path

import numpy as np


LEVEL_NAMES = ["macro", "meso", "micro"]
CELL_NAMES = [
    f"{LEVEL_NAMES[i]}->{LEVEL_NAMES[j]}"
    for i in range(3) for j in range(3)
]


def transition_entropy(mat: np.ndarray) -> float:
    """Shannon entropy of the flattened transition matrix."""
    flat = mat.flatten()
    flat = flat[flat > 0]  # exclude zeros
    if len(flat) == 0:
        return 0.0
    return float(-np.sum(flat * np.log2(flat)))


def self_loop_ratio(mat: np.ndarray) -> float:
    """Sum of diagonal elements (proportion of self-transitions)."""
    total = mat.sum()
    if total == 0:
        return 0.0
    return float(np.trace(mat) / total)


def oscillation_index(mat: np.ndarray) -> float:
    """Std of off-diagonal elements — measures transition variability."""
    mask = ~np.eye(3, dtype=bool)
    off_diag = mat[mask]
    if len(off_diag) == 0:
        return 0.0
    return float(np.std(off_diag))


def macro_micro_shuttle(mat: np.ndarray) -> float:
    """Proportion of macro<->micro transitions (long-range jumps)."""
    total = mat.sum()
    if total == 0:
        return 0.0
    return float((mat[0, 2] + mat[2, 0]) / total)


def upward_ratio(mat: np.ndarray) -> float:
    """Proportion of off-diagonal transitions going coarser->finer (increasing index)."""
    off_diag_sum = mat.sum() - np.trace(mat)
    if off_diag_sum == 0:
        return 0.0
    upward = mat[0, 1] + mat[0, 2] + mat[1, 2]  # macro->meso, macro->micro, meso->micro
    return float(upward / off_diag_sum)


def downward_ratio(mat: np.ndarray) -> float:
    """Proportion of off-diagonal transitions going finer->coarser (decreasing index)."""
    off_diag_sum = mat.sum() - np.trace(mat)
    if off_diag_sum == 0:
        return 0.0
    downward = mat[1, 0] + mat[2, 0] + mat[2, 1]
    return float(downward / off_diag_sum)


def dominant_transition(mat: np.ndarray) -> int:
    """Index of the dominant transition cell (argmax of flattened matrix)."""
    return int(np.argmax(mat.flatten()))


def extract_features_from_matrix(mat: np.ndarray, prefix: str = "") -> dict:
    """Extract all features from a single 3x3 transition matrix.

    Args:
        mat: (3, 3) transition matrix
        prefix: feature name prefix (e.g., "hard_" or "soft_")
    """
    features = {}

    # All 9 cells
    flat = mat.flatten()
    for i, name in enumerate(CELL_NAMES):
        features[f"{prefix}{name}"] = float(flat[i])

    # Derived features
    features[f"{prefix}self_loop_ratio"] = self_loop_ratio(mat)
    features[f"{prefix}entropy"] = transition_entropy(mat)
    features[f"{prefix}oscillation_index"] = oscillation_index(mat)
    features[f"{prefix}macro_micro_shuttle"] = macro_micro_shuttle(mat)
    features[f"{prefix}upward_ratio"] = upward_ratio(mat)
    features[f"{prefix}downward_ratio"] = downward_ratio(mat)
    features[f"{prefix}dominant_transition"] = dominant_transition(mat)

    return features


def extract_all_features(
    hard_matrices: np.ndarray,
    soft_matrices: np.ndarray,
    metadata: list[dict],
) -> list[dict]:
    """Extract features from all trace matrices.

    Args:
        hard_matrices: (N, 3, 3)
        soft_matrices: (N, 3, 3)
        metadata: list of per-trace metadata dicts
    """
    features_list = []
    for idx in range(len(metadata)):
        feats = {}
        # Identifiers
        feats["question_id"] = metadata[idx]["question_id"]
        feats["condition"] = metadata[idx]["condition"]
        feats["n_steps"] = metadata[idx]["n_steps"]
        feats["n_transitions"] = metadata[idx]["n_transitions"]
        feats["answer_token_f1"] = metadata[idx]["answer_token_f1"]
        feats["attribution_f1"] = metadata[idx]["attribution_f1"]
        feats["jump_count"] = metadata[idx]["jump_count"]
        feats["normalized_jump_rate"] = metadata[idx]["normalized_jump_rate"]

        # Hard features
        feats.update(extract_features_from_matrix(hard_matrices[idx], prefix="hard_"))
        # Soft features
        feats.update(extract_features_from_matrix(soft_matrices[idx], prefix="soft_"))

        features_list.append(feats)

    return features_list


def extract_aggregated_features(
    agg_hard: np.ndarray,
    agg_soft: np.ndarray,
    agg_metadata: list[dict],
) -> list[dict]:
    """Extract features from per-question aggregated matrices."""
    features_list = []
    for idx in range(len(agg_metadata)):
        feats = {}
        feats["question_id"] = agg_metadata[idx]["question_id"]
        feats["n_conditions"] = agg_metadata[idx]["n_conditions"]
        feats["mean_token_f1"] = agg_metadata[idx]["mean_token_f1"]
        feats["mean_attribution_f1"] = agg_metadata[idx]["mean_attribution_f1"]

        feats.update(extract_features_from_matrix(agg_hard[idx], prefix="hard_"))
        feats.update(extract_features_from_matrix(agg_soft[idx], prefix="soft_"))

        features_list.append(feats)

    return features_list


def save_features(features: list[dict], path: str) -> None:
    """Save feature records to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for feat in features:
            f.write(json.dumps(feat) + "\n")
    print(f"Saved {len(features)} feature records to {path}")


def get_numeric_feature_names(prefix: str = "") -> list[str]:
    """Get list of numeric feature names for a given prefix."""
    names = [f"{prefix}{cn}" for cn in CELL_NAMES]
    names.extend([
        f"{prefix}self_loop_ratio",
        f"{prefix}entropy",
        f"{prefix}oscillation_index",
        f"{prefix}macro_micro_shuttle",
        f"{prefix}upward_ratio",
        f"{prefix}downward_ratio",
    ])
    return names
