"""Stage D: Compute jump metrics from SLoD-tagged CoT traces."""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import load_jsonl, resolve_path, save_jsonl

logger = logging.getLogger(__name__)


def compute_jump_metrics(slod_sequence: list[int]) -> dict:
    """Compute all 6 jump metrics from a list of SLoD level integers.

    Args:
        slod_sequence: List of SLoD levels (0=macro, 1=meso, 2=micro).

    Returns:
        Dict with jump_count, mean_abs_delta, max_jump, slod_variance,
        direction_changes, normalized_jump_rate.
    """
    n = len(slod_sequence)

    if n <= 1:
        return {
            "jump_count": 0,
            "mean_abs_delta": 0.0,
            "max_jump": 0,
            "slod_variance": 0.0,
            "direction_changes": 0,
            "normalized_jump_rate": 0.0,
        }

    s = slod_sequence

    # Deltas between consecutive steps
    deltas = [s[i] - s[i - 1] for i in range(1, n)]
    abs_deltas = [abs(d) for d in deltas]

    # jump_count: number of step transitions where SLoD changes
    jump_count = sum(1 for d in deltas if d != 0)

    # mean_abs_delta: average absolute SLoD change
    mean_abs_delta = float(np.mean(abs_deltas))

    # max_jump: largest single-step SLoD change
    max_jump = max(abs_deltas)

    # slod_variance: variance of SLoD levels
    slod_variance = float(np.var(s))

    # direction_changes: sign changes in non-zero deltas
    nonzero_deltas = [d for d in deltas if d != 0]
    direction_changes = 0
    if len(nonzero_deltas) >= 2:
        for i in range(1, len(nonzero_deltas)):
            if (nonzero_deltas[i] > 0) != (nonzero_deltas[i - 1] > 0):
                direction_changes += 1

    # normalized_jump_rate: jump_count / (N-1)
    normalized_jump_rate = jump_count / (n - 1)

    return {
        "jump_count": jump_count,
        "mean_abs_delta": round(mean_abs_delta, 6),
        "max_jump": max_jump,
        "slod_variance": round(slod_variance, 6),
        "direction_changes": direction_changes,
        "normalized_jump_rate": round(normalized_jump_rate, 6),
    }


def process_all_traces(config: dict) -> Path:
    """Load SLoD tags, group by (qid, condition), compute jump metrics, save.

    Returns path to saved output.
    """
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    output_path = data_dir / "jump_metrics.jsonl"

    if output_path.exists():
        existing = load_jsonl(output_path)
        logger.info(
            f"Jump metrics already exist at {output_path} ({len(existing)} records), skipping."
        )
        return output_path

    # Load SLoD tags
    tags_path = data_dir / "cot_slod_tags.jsonl"
    tags = load_jsonl(tags_path)
    logger.info(f"Loaded {len(tags)} SLoD tags")

    # Group by (question_id, condition)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for tag in tags:
        key = (tag["question_id"], tag["condition"])
        grouped[key].append(tag)

    # Sort steps within each group by step_index
    for key in grouped:
        grouped[key].sort(key=lambda x: x["step_index"])

    # Compute metrics
    results = []
    for (qid, condition), step_tags in grouped.items():
        slod_sequence = [t["predicted_slod"] for t in step_tags]
        metrics = compute_jump_metrics(slod_sequence)

        results.append({
            "question_id": qid,
            "condition": condition,
            "n_steps": len(slod_sequence),
            "slod_sequence": slod_sequence,
            **metrics,
        })

    save_jsonl(results, output_path)
    logger.info(f"Saved {len(results)} jump metric records to {output_path}")

    # Report summary statistics
    if results:
        import pandas as pd

        df = pd.DataFrame(results)
        logger.info(f"\nJump metrics summary:\n{df.describe().to_string()}")

        # Per-condition summary
        for cond in config["retrieval"]["conditions"]:
            cond_df = df[df["condition"] == cond]
            if len(cond_df) > 0:
                logger.info(
                    f"\n[{cond}] n={len(cond_df)}, "
                    f"mean_njr={cond_df['normalized_jump_rate'].mean():.4f}, "
                    f"mean_steps={cond_df['n_steps'].mean():.1f}"
                )

    return output_path
