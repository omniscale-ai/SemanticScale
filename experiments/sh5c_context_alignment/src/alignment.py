"""Stage B: Alignment Feature Engineering.

Computes alignment features between context SLoD profile (from retrieved chunks)
and reasoning SLoD profile (from CoT step labels).
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy


EPSILON = 1e-10


def compute_alignment_features(trace: dict, diversity_ratio_cap: float = 100.0) -> dict:
    """Compute all alignment features for a single trace.

    Args:
        trace: dict with context and reasoning profiles (from merged_traces)
        diversity_ratio_cap: cap for diversity_ratio when context_diversity ~ 0

    Returns:
        dict with all alignment feature values
    """
    context_mean = trace.get("context_slod_mean")
    context_dist = np.array(trace.get("context_slod_distribution", [0, 0, 0]), dtype=float)
    reasoning_dist = np.array(trace.get("reasoning_slod_distribution", [0, 0, 0]), dtype=float)
    reasoning_soft_dist = np.array(trace.get("reasoning_slod_soft_distribution", [0, 0, 0]), dtype=float)
    step_slods = trace.get("reasoning_step_slods", [])
    step_confidences = trace.get("reasoning_step_confidences", [])
    n_steps = trace.get("reasoning_n_steps", 0)
    context_n_docs = trace.get("context_n_docs", 0)

    features = {}

    # Handle degenerate cases
    if n_steps == 0 or context_n_docs == 0 or np.isnan(context_mean):
        return _nan_features()

    step_slods_arr = np.array(step_slods, dtype=float)

    # 1. mean_alignment_gap: mean |context_slod_mean - step_slod| across steps
    gaps = np.abs(context_mean - step_slods_arr)
    features["mean_alignment_gap"] = float(np.mean(gaps))

    # 2. max_alignment_gap: max |context_slod_mean - step_slod|
    features["max_alignment_gap"] = float(np.max(gaps))

    # 3. jsd: Jensen-Shannon divergence between reasoning and context distributions
    # Add epsilon to avoid zero entries
    c_dist = context_dist + EPSILON
    r_dist = reasoning_dist + EPSILON
    c_dist = c_dist / c_dist.sum()
    r_dist = r_dist / r_dist.sum()
    # scipy jensenshannon returns sqrt(JSD), square it for actual JSD
    jsd_sqrt = jensenshannon(r_dist, c_dist)
    features["jsd"] = float(jsd_sqrt ** 2)

    # 4. dominant_level_match: 1 if mode(reasoning) == mode(context)
    reasoning_mode = int(np.argmax(reasoning_dist))
    context_mode = int(np.argmax(context_dist))
    features["dominant_level_match"] = 1 if reasoning_mode == context_mode else 0

    # 5. context_reasoning_correlation: Pearson r between distributions
    features["context_reasoning_correlation"] = _safe_pearson(context_dist, reasoning_dist)

    # 6. weighted_alignment_gap: confidence-weighted alignment gap
    if step_confidences and len(step_confidences) == n_steps:
        confs = np.array(step_confidences, dtype=float)
        total_conf = confs.sum()
        if total_conf > 0:
            features["weighted_alignment_gap"] = float(np.sum(confs * gaps) / total_conf)
        else:
            features["weighted_alignment_gap"] = features["mean_alignment_gap"]
    else:
        features["weighted_alignment_gap"] = features["mean_alignment_gap"]

    # 7. context_diversity: Shannon entropy of context distribution (base 2)
    features["context_diversity"] = float(shannon_entropy(c_dist, base=2))

    # 8. reasoning_diversity: Shannon entropy of reasoning distribution (base 2)
    features["reasoning_diversity"] = float(shannon_entropy(r_dist, base=2))

    # 9. diversity_ratio: reasoning_diversity / (context_diversity + epsilon)
    cd = features["context_diversity"]
    rd = features["reasoning_diversity"]
    if cd > EPSILON:
        features["diversity_ratio"] = min(float(rd / cd), diversity_ratio_cap)
    else:
        features["diversity_ratio"] = diversity_ratio_cap if rd > EPSILON else 1.0

    # 10. soft_mean_gap: L1 distance between context_dist and reasoning_soft_dist
    rs_dist = reasoning_soft_dist + EPSILON
    rs_dist = rs_dist / rs_dist.sum()
    features["soft_mean_gap"] = float(np.mean(np.abs(c_dist - rs_dist)))

    # 11. soft_jsd: JSD between reasoning_soft_dist and context_dist
    jsd_soft_sqrt = jensenshannon(rs_dist, c_dist)
    features["soft_jsd"] = float(jsd_soft_sqrt ** 2)

    return features


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning NaN for degenerate cases."""
    if len(x) < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < EPSILON or y_std < EPSILON:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def _nan_features() -> dict:
    """Return dict with NaN for all features."""
    return {
        "mean_alignment_gap": float("nan"),
        "max_alignment_gap": float("nan"),
        "jsd": float("nan"),
        "dominant_level_match": float("nan"),
        "context_reasoning_correlation": float("nan"),
        "weighted_alignment_gap": float("nan"),
        "context_diversity": float("nan"),
        "reasoning_diversity": float("nan"),
        "diversity_ratio": float("nan"),
        "soft_mean_gap": float("nan"),
        "soft_jsd": float("nan"),
    }


def process_all_traces(merged_path: Path, output_path: Path,
                       diversity_ratio_cap: float = 100.0):
    """Process all merged traces and compute alignment features.

    Reads merged_traces.jsonl, computes features, writes alignment_features.jsonl.
    """
    traces = []
    with open(merged_path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    n_nan = 0

    with open(output_path, "w") as out:
        for trace in traces:
            features = compute_alignment_features(trace, diversity_ratio_cap)

            row = {
                "question_id": trace["question_id"],
                "condition": trace["condition"],
                "answer_token_f1": trace["answer_token_f1"],
                "attribution_f1": trace["attribution_f1"],
                "answer_type": trace.get("answer_type", "unknown"),
                # Context summary
                "context_slod_mean": trace.get("context_slod_mean"),
                "context_n_docs": trace.get("context_n_docs", 0),
                "reasoning_n_steps": trace.get("reasoning_n_steps", 0),
                # Jump metrics (for comparison)
                "n_steps": trace.get("n_steps"),
                "normalized_jump_rate": trace.get("normalized_jump_rate"),
                "slod_variance": trace.get("slod_variance"),
                "mean_abs_delta": trace.get("mean_abs_delta"),
                "max_jump": trace.get("max_jump"),
                "direction_changes": trace.get("direction_changes"),
                # Alignment features
                **features,
            }
            out.write(json.dumps(row, default=_json_default) + "\n")

            if np.isnan(features.get("mean_alignment_gap", float("nan"))):
                n_nan += 1
            else:
                n_valid += 1

    print(f"Alignment features computed: {n_valid} valid, {n_nan} NaN (no context), "
          f"{n_valid + n_nan} total")
    print(f"Saved to {output_path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")
