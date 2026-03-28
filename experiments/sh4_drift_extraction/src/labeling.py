"""Stage C: Silver label matching — fuzzy-match LLM extractions against PwC ground truth."""

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.utils import (
    get_project_root,
    load_config,
    normalize_text,
    parse_numeric_value,
    read_jsonl,
    write_jsonl,
)


def string_similarity(a: str, b: str) -> float:
    """Compute string similarity using SequenceMatcher ratio.

    Returns value in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def is_substring_match(a: str, b: str) -> bool:
    """Check if one string is a substring of the other (case-insensitive)."""
    if not a or not b:
        return False
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    return a_norm in b_norm or b_norm in a_norm


def dataset_matches(extracted: str, ground_truth: str, threshold: float) -> bool:
    """Check if extracted dataset name matches ground truth."""
    if is_substring_match(extracted, ground_truth):
        return True
    return string_similarity(extracted, ground_truth) >= threshold


def metric_matches(extracted: str, ground_truth: str, threshold: float) -> bool:
    """Check if extracted metric name matches ground truth.

    Handles common metric name variations.
    """
    # Normalize common metric name variations
    metric_aliases = {
        "f1": ["f1-score", "f1 score", "f-measure", "f-score"],
        "accuracy": ["acc", "top-1 accuracy", "top1 accuracy"],
        "bleu": ["bleu-4", "bleu4"],
        "rouge-l": ["rougel", "rouge l"],
        "rouge-1": ["rouge1", "rouge 1"],
        "rouge-2": ["rouge2", "rouge 2"],
        "map": ["mean average precision", "map@"],
        "em": ["exact match", "exact-match"],
        "auc": ["auroc", "auc-roc"],
        "ppl": ["perplexity"],
    }

    ext_norm = normalize_text(extracted)
    gt_norm = normalize_text(ground_truth)

    if ext_norm == gt_norm:
        return True

    # Check aliases
    for canonical, aliases in metric_aliases.items():
        all_names = [canonical] + aliases
        ext_match = any(ext_norm == a or ext_norm.startswith(a) for a in all_names)
        gt_match = any(gt_norm == a or gt_norm.startswith(a) for a in all_names)
        if ext_match and gt_match:
            return True

    if is_substring_match(extracted, ground_truth):
        return True

    return string_similarity(extracted, ground_truth) >= threshold


def value_matches(extracted_str: str, ground_truth_str: str, tolerance: float) -> bool:
    """Check if extracted numeric value matches ground truth within tolerance.

    Handles percentage normalization (e.g., 0.874 vs 87.4).
    """
    ext_val = parse_numeric_value(extracted_str)
    gt_val = parse_numeric_value(ground_truth_str)

    if ext_val is None or gt_val is None:
        return False

    # Handle both values being zero (or very close to zero)
    if abs(ext_val) < 1e-9 and abs(gt_val) < 1e-9:
        return True

    # Direct comparison
    if gt_val != 0:
        if abs(ext_val - gt_val) / abs(gt_val) <= tolerance:
            return True

    # Handle percentage vs decimal mismatch
    # e.g., extracted 87.4 vs ground_truth 0.874, or vice versa
    for multiplier in [100.0, 0.01]:
        adjusted = ext_val * multiplier
        if gt_val != 0 and abs(adjusted - gt_val) / abs(gt_val) <= tolerance:
            return True

    return False


def match_extraction_to_ground_truth(
    extraction: dict,
    paper_gt_rows: pd.DataFrame,
    labeling_cfg: dict,
) -> tuple[bool, dict | None]:
    """Match a single extraction against ground truth rows for the same paper.

    Returns (is_correct, best_match_info).
    """
    dataset_threshold = labeling_cfg["dataset_match_ratio"]
    metric_threshold = labeling_cfg["metric_match_ratio"]
    value_tol = labeling_cfg["value_tolerance"]

    ext_dataset = extraction.get("dataset", "")
    ext_metric = extraction.get("metric", "")
    ext_value = str(extraction.get("value", ""))

    best_score = 0.0
    best_match = None

    # Identify dataset and metric columns in ground truth
    # PwC eval tables may use different column names
    gt_dataset_col = None
    for col in ["dataset", "dataset_name", "Dataset"]:
        if col in paper_gt_rows.columns:
            gt_dataset_col = col
            break

    gt_metric_col = None
    for col in ["metric_name", "metric", "Metric"]:
        if col in paper_gt_rows.columns:
            gt_metric_col = col
            break

    gt_value_col = None
    for col in ["metric_value", "value", "Value", "metric_result"]:
        if col in paper_gt_rows.columns:
            gt_value_col = col
            break

    if gt_dataset_col is None or gt_metric_col is None or gt_value_col is None:
        logging.warning(
            f"Missing ground truth columns. Available: {list(paper_gt_rows.columns)}"
        )
        return False, None

    for _, gt_row in paper_gt_rows.iterrows():
        gt_dataset = str(gt_row[gt_dataset_col]) if pd.notna(gt_row[gt_dataset_col]) else ""
        gt_metric = str(gt_row[gt_metric_col]) if pd.notna(gt_row[gt_metric_col]) else ""
        gt_value = str(gt_row[gt_value_col]) if pd.notna(gt_row[gt_value_col]) else ""

        d_match = dataset_matches(ext_dataset, gt_dataset, dataset_threshold)
        m_match = metric_matches(ext_metric, gt_metric, metric_threshold)
        v_match = value_matches(ext_value, gt_value, value_tol)

        # Score: how many of the 3 criteria match
        score = sum([d_match, m_match, v_match])

        if score > best_score:
            best_score = score
            best_match = {
                "gt_dataset": gt_dataset,
                "gt_metric": gt_metric,
                "gt_value": gt_value,
                "dataset_match": d_match,
                "metric_match": m_match,
                "value_match": v_match,
            }

    # All three must match for "correct"
    is_correct = best_score >= 3

    return is_correct, best_match


def run_labeling(config: dict = None, project_root: Path = None) -> Path:
    """Run silver label matching pipeline.

    Returns path to labeled extractions JSONL file.
    """
    if config is None:
        config = load_config()
    if project_root is None:
        project_root = get_project_root()

    lab_cfg = config["labeling"]
    data_dir = project_root / config["data_dir"]

    # Output path
    output_path = data_dir / lab_cfg["output"]
    if output_path.exists():
        logging.info(f"Labeled extractions already exist at {output_path}, skipping.")
        return output_path

    # Load extractions
    ext_path = data_dir / config["extraction"]["output"]
    if not ext_path.exists():
        raise FileNotFoundError(f"Extractions not found at {ext_path}. Run Stage B first.")
    extractions = read_jsonl(ext_path)
    logging.info(f"Loaded {len(extractions)} extractions")

    # Load ground truth (joined parquet)
    joined_path = data_dir / config["data_acquisition"]["output"]["joined"]
    if not joined_path.exists():
        raise FileNotFoundError(f"Joined data not found at {joined_path}. Run Stage A first.")
    gt_df = pd.read_parquet(joined_path)
    logging.info(f"Loaded {len(gt_df)} ground truth rows")

    # Group ground truth by paper
    gt_by_paper = {
        arxiv_id: group for arxiv_id, group in gt_df.groupby("arxiv_id")
    }

    # Match each extraction
    n_correct = 0
    n_total = 0
    labeled_extractions = []

    for ext in extractions:
        paper_id = ext.get("paper_id", "")
        paper_gt = gt_by_paper.get(paper_id)

        if paper_gt is None or len(paper_gt) == 0:
            # No ground truth for this paper
            ext["correct"] = 0
            ext["match_info"] = None
            labeled_extractions.append(ext)
            n_total += 1
            continue

        is_correct, match_info = match_extraction_to_ground_truth(
            ext, paper_gt, lab_cfg
        )

        ext["correct"] = int(is_correct)
        ext["match_info"] = match_info
        labeled_extractions.append(ext)

        if is_correct:
            n_correct += 1
        n_total += 1

    # Save
    write_jsonl(labeled_extractions, output_path)

    # Summary
    match_rate = n_correct / n_total if n_total > 0 else 0.0
    logging.info(f"=== Labeling complete ===")
    logging.info(f"Total extractions: {n_total}")
    logging.info(f"Correct: {n_correct} ({match_rate:.1%})")
    logging.info(f"Incorrect: {n_total - n_correct} ({1 - match_rate:.1%})")

    # Warn if match rate is problematic
    if match_rate < 0.2:
        logging.warning(
            f"Match rate ({match_rate:.1%}) is very low. Consider loosening matching criteria."
        )
    elif match_rate > 0.8:
        logging.warning(
            f"Match rate ({match_rate:.1%}) is very high. Consider tightening matching criteria."
        )

    return output_path
