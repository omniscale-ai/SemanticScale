"""Evaluation for SH3: attribution F1, recall@k, MRR, paired bootstrap tests."""

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import load_config, load_json, load_jsonl, normalize_text, save_json


def token_f1(pred_text: str, gold_text: str) -> float:
    """Compute token-level F1 between two passages.

    Tokenizes by splitting on whitespace after lowering+stripping.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = normalize_text(pred_text).split()
    gold_tokens = normalize_text(gold_text).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    # Use multiset overlap (count-based) for more precise matching
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts if t in gold_counts)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def match_retrieved_to_gold(
    retrieved_texts: list[str],
    gold_texts: list[str],
    threshold: float = 0.5,
) -> tuple[set[int], set[int]]:
    """Find matches between retrieved passages and gold evidence.

    A retrieved passage matches a gold paragraph if token-F1 >= threshold.

    Returns:
        Tuple of (matched_retrieved_indices, matched_gold_indices).
    """
    matched_retrieved = set()
    matched_gold = set()

    for r_idx, r_text in enumerate(retrieved_texts):
        for g_idx, g_text in enumerate(gold_texts):
            f1 = token_f1(r_text, g_text)
            if f1 >= threshold:
                matched_retrieved.add(r_idx)
                matched_gold.add(g_idx)

    return matched_retrieved, matched_gold


def compute_attribution_f1(
    retrieved_texts: list[str],
    gold_texts: list[str],
    threshold: float = 0.5,
) -> dict:
    """Compute attribution precision, recall, and F1 for one question.

    Returns:
        Dict with precision, recall, f1.
    """
    if not retrieved_texts or not gold_texts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_retrieved, matched_gold = match_retrieved_to_gold(
        retrieved_texts, gold_texts, threshold
    )

    precision = len(matched_retrieved) / len(retrieved_texts) if retrieved_texts else 0.0
    recall = len(matched_gold) / len(gold_texts) if gold_texts else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_recall_at_k(
    retrieved_texts: list[str],
    gold_texts: list[str],
    threshold: float = 0.5,
) -> float:
    """Compute recall@k: fraction of gold evidence covered by retrieved passages.

    Returns:
        Recall value in [0, 1].
    """
    if not gold_texts:
        return 0.0
    if not retrieved_texts:
        return 0.0

    _, matched_gold = match_retrieved_to_gold(retrieved_texts, gold_texts, threshold)
    return len(matched_gold) / len(gold_texts)


def compute_mrr(
    retrieved_texts: list[str],
    gold_texts: list[str],
    threshold: float = 0.5,
) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first matching retrieved passage.

    Returns:
        MRR value in [0, 1], 0 if no match.
    """
    if not retrieved_texts or not gold_texts:
        return 0.0

    for r_idx, r_text in enumerate(retrieved_texts):
        for g_text in gold_texts:
            if token_f1(r_text, g_text) >= threshold:
                return 1.0 / (r_idx + 1)

    return 0.0


def compute_token_cost(retrieved_texts: list[str]) -> int:
    """Compute total token count of retrieved passages."""
    return sum(len(t.split()) for t in retrieved_texts)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """Paired bootstrap test for significance of difference between two systems.

    Tests whether system A is significantly different from system B.

    Args:
        scores_a: Per-question scores for system A (e.g., SLoD-routed).
        scores_b: Per-question scores for system B (e.g., baseline).
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: Confidence level for CI.
        random_seed: Random seed.

    Returns:
        Dict with observed_diff, p_value, ci_lower, ci_upper, significant.
    """
    rng = np.random.RandomState(random_seed)
    n = len(scores_a)
    assert len(scores_b) == n, f"Score arrays must have same length: {len(scores_a)} vs {len(scores_b)}"

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    # Bootstrap
    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        diffs[i] = np.mean(scores_a[indices]) - np.mean(scores_b[indices])

    # Two-sided p-value: fraction of bootstrap diffs that have opposite sign
    # or magnitude >= observed in opposite direction
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0))
    else:
        p_value = float(np.mean(diffs >= 0))

    # Confidence interval
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "observed_diff": observed_diff,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def compute_soft_attribution_f1(
    retrieved_texts: list[str],
    gold_texts: list[str],
) -> dict:
    """Compute soft (partial-credit) attribution precision, recall, and F1.

    Unlike binary attribution F1 which uses a threshold to determine match/no-match,
    this metric gives proportional credit based on token-F1 overlap.

    For each retrieved passage: max_overlap = max(token_f1(retrieved, gold_j) for all j)
    Soft precision = mean of max_overlaps across retrieved passages.

    For each gold paragraph: max_coverage = max(token_f1(retrieved_i, gold) for all i)
    Soft recall = mean of max_coverages across gold paragraphs.

    Soft F1 = harmonic mean of soft precision and soft recall.

    Returns:
        Dict with soft_precision, soft_recall, soft_f1.
    """
    if not retrieved_texts or not gold_texts:
        return {"soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0}

    # Build the full overlap matrix once
    overlap_matrix = []
    for r_text in retrieved_texts:
        row = [token_f1(r_text, g_text) for g_text in gold_texts]
        overlap_matrix.append(row)

    # Soft precision: for each retrieved, max overlap with any gold
    max_overlaps = [max(row) for row in overlap_matrix]
    soft_precision = sum(max_overlaps) / len(max_overlaps)

    # Soft recall: for each gold, max overlap with any retrieved
    max_coverages = []
    for g_idx in range(len(gold_texts)):
        max_cov = max(overlap_matrix[r_idx][g_idx] for r_idx in range(len(retrieved_texts)))
        max_coverages.append(max_cov)
    soft_recall = sum(max_coverages) / len(max_coverages)

    # Harmonic mean
    if soft_precision + soft_recall == 0:
        soft_f1 = 0.0
    else:
        soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall)

    return {
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "soft_f1": soft_f1,
    }


def sweep_thresholds(
    retrieved_texts: list[str],
    gold_texts: list[str],
    thresholds: list[float],
) -> dict:
    """Compute attribution P, R, F1 at multiple matching thresholds.

    Returns:
        Dict of threshold (str) -> {"precision": float, "recall": float, "f1": float}.
    """
    results = {}
    for t in thresholds:
        attr = compute_attribution_f1(retrieved_texts, gold_texts, t)
        results[str(t)] = attr
    return results


def evaluate_all(config: dict, project_root: Path | None = None) -> None:
    """Compute all metrics for all conditions and k values, sweeping thresholds.

    Produces:
    - data/results/evaluation_metrics.json (includes threshold dimension)
    - data/results/bootstrap_tests.json
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    results_dir = data_dir / "results"

    metrics_path = results_dir / "evaluation_metrics.json"
    bootstrap_path = results_dir / "bootstrap_tests.json"

    if metrics_path.exists() and bootstrap_path.exists():
        logging.info("Evaluation results already exist, skipping.")
        return

    # Load retrieval results
    retrieval_results = load_json(results_dir / "retrieval_results.json")

    # Load questions for gold evidence
    questions = load_jsonl(data_dir / "questions.jsonl")
    qid_to_question = {q["question_id"]: q for q in questions}

    # Threshold sweep configuration
    thresholds = config["evaluation"].get("matching_thresholds", [0.3, 0.4, 0.5, 0.6])
    primary_threshold = config["evaluation"].get("primary_threshold", 0.5)
    compute_soft = config["evaluation"].get("compute_soft_metrics", False)
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]
    eval_split = config["qasper"]["eval_split"]

    # Compute metrics for each condition x k x threshold
    all_metrics = {}
    # per_question_f1: condition -> k_str -> threshold_str -> [f1 scores]
    per_question_f1 = {}
    # per_question_soft_f1: condition -> k_str -> [soft f1 scores]
    per_question_soft_f1 = {}

    for condition in conditions:
        logging.info(f"Evaluating condition: {condition}")
        condition_metrics = {}
        per_question_f1[condition] = {}
        per_question_soft_f1[condition] = {}

        for k in top_k_values:
            k_str = str(k)
            k_results = retrieval_results.get(condition, {}).get(k_str, [])

            # Per-threshold accumulators
            threshold_f1 = {str(t): [] for t in thresholds}
            threshold_precision = {str(t): [] for t in thresholds}
            threshold_recall = {str(t): [] for t in thresholds}

            recall_at_k_scores = {str(t): [] for t in thresholds}
            mrr_scores = {str(t): [] for t in thresholds}
            token_costs = []

            # Soft metric accumulators
            soft_f1_scores = []
            soft_precision_scores = []
            soft_recall_scores = []

            for entry in k_results:
                qid = entry["question_id"]
                q = qid_to_question.get(qid)
                if not q:
                    continue
                # Only evaluate on the eval split
                if q.get("split") != eval_split:
                    continue

                retrieved_texts = [r["text"] for r in entry.get("retrieved", [])]
                gold_texts = q["gold_evidence"]

                # Sweep thresholds for attribution F1
                sweep = sweep_thresholds(retrieved_texts, gold_texts, thresholds)
                for t_str, attr in sweep.items():
                    threshold_f1[t_str].append(attr["f1"])
                    threshold_precision[t_str].append(attr["precision"])
                    threshold_recall[t_str].append(attr["recall"])

                # Recall@k and MRR at each threshold
                for t in thresholds:
                    t_str = str(t)
                    rec = compute_recall_at_k(retrieved_texts, gold_texts, t)
                    recall_at_k_scores[t_str].append(rec)
                    mrr_val = compute_mrr(retrieved_texts, gold_texts, t)
                    mrr_scores[t_str].append(mrr_val)

                # Token cost (threshold-independent)
                tc = compute_token_cost(retrieved_texts)
                token_costs.append(tc)

                # Soft attribution metrics (threshold-independent)
                if compute_soft:
                    soft = compute_soft_attribution_f1(retrieved_texts, gold_texts)
                    soft_f1_scores.append(soft["soft_f1"])
                    soft_precision_scores.append(soft["soft_precision"])
                    soft_recall_scores.append(soft["soft_recall"])

            # Check we have data
            primary_t_str = str(primary_threshold)
            n_questions = len(threshold_f1.get(primary_t_str, []))
            if n_questions == 0:
                logging.warning(f"No questions evaluated for {condition} k={k}")
                continue

            # Build metrics dict with threshold dimension
            k_metrics = {
                "n_questions": n_questions,
                "mean_token_cost": float(np.mean(token_costs)) if token_costs else 0.0,
            }

            # Add per-threshold metrics
            by_threshold = {}
            for t in thresholds:
                t_str = str(t)
                f1s = threshold_f1[t_str]
                by_threshold[t_str] = {
                    "attribution_f1": float(np.mean(f1s)) if f1s else 0.0,
                    "attribution_precision": float(np.mean(threshold_precision[t_str])) if f1s else 0.0,
                    "attribution_recall": float(np.mean(threshold_recall[t_str])) if f1s else 0.0,
                    "recall_at_k": float(np.mean(recall_at_k_scores[t_str])) if recall_at_k_scores[t_str] else 0.0,
                    "mrr": float(np.mean(mrr_scores[t_str])) if mrr_scores[t_str] else 0.0,
                    "std_attribution_f1": float(np.std(f1s)) if f1s else 0.0,
                }
            k_metrics["by_threshold"] = by_threshold

            # Also store primary threshold metrics at the top level for backward compat
            primary_data = by_threshold.get(primary_t_str, {})
            k_metrics["attribution_f1"] = primary_data.get("attribution_f1", 0.0)
            k_metrics["attribution_precision"] = primary_data.get("attribution_precision", 0.0)
            k_metrics["attribution_recall"] = primary_data.get("attribution_recall", 0.0)
            k_metrics["recall_at_k"] = primary_data.get("recall_at_k", 0.0)
            k_metrics["mrr"] = primary_data.get("mrr", 0.0)
            k_metrics["std_attribution_f1"] = primary_data.get("std_attribution_f1", 0.0)

            # Add soft metrics at top level
            if compute_soft and soft_f1_scores:
                k_metrics["soft_attribution_f1"] = float(np.mean(soft_f1_scores))
                k_metrics["soft_precision"] = float(np.mean(soft_precision_scores))
                k_metrics["soft_recall"] = float(np.mean(soft_recall_scores))
                k_metrics["std_soft_attribution_f1"] = float(np.std(soft_f1_scores))

            condition_metrics[k_str] = k_metrics

            # Store per-question F1 for bootstrap (primary threshold)
            per_question_f1[condition][k_str] = {}
            for t in thresholds:
                t_str = str(t)
                per_question_f1[condition][k_str][t_str] = np.array(threshold_f1[t_str])

            # Store per-question soft F1 for bootstrap
            if compute_soft and soft_f1_scores:
                per_question_soft_f1[condition][k_str] = np.array(soft_f1_scores)

        all_metrics[condition] = condition_metrics

    save_json(all_metrics, metrics_path)
    logging.info(f"Evaluation metrics saved to {metrics_path}")

    # Bootstrap tests: SLoD-routed vs each baseline (at primary threshold)
    bootstrap_cfg = config["evaluation"]["bootstrap"]
    bootstrap_results = {}

    slod_condition = "slod_routed"
    baselines = [c for c in conditions if c != slod_condition]

    primary_t_str = str(primary_threshold)

    for k in top_k_values:
        k_str = str(k)
        slod_scores = per_question_f1.get(slod_condition, {}).get(k_str, {}).get(primary_t_str)
        if slod_scores is None or len(slod_scores) == 0:
            continue

        k_tests = {}
        for baseline in baselines:
            baseline_scores = per_question_f1.get(baseline, {}).get(k_str, {}).get(primary_t_str)
            if baseline_scores is None or len(baseline_scores) == 0:
                continue
            if len(baseline_scores) != len(slod_scores):
                logging.warning(
                    f"Score length mismatch for k={k}: {slod_condition}={len(slod_scores)}, "
                    f"{baseline}={len(baseline_scores)}"
                )
                continue

            test_result = paired_bootstrap_test(
                slod_scores,
                baseline_scores,
                n_bootstrap=bootstrap_cfg["n_resamples"],
                confidence_level=bootstrap_cfg["confidence_level"],
                random_seed=bootstrap_cfg["random_seed"],
            )
            k_tests[baseline] = test_result

            logging.info(
                f"  k={k} {slod_condition} vs {baseline}: "
                f"diff={test_result['observed_diff']:.4f}, "
                f"p={test_result['p_value']:.4f}, "
                f"CI=[{test_result['ci_lower']:.4f}, {test_result['ci_upper']:.4f}], "
                f"sig={test_result['significant']}"
            )

        bootstrap_results[k_str] = k_tests

    # Bootstrap tests for soft F1: slod_weighted_parent vs baselines
    if compute_soft:
        soft_bootstrap_results = {}
        # Use slod_weighted_parent as primary if available, else slod_routed
        soft_primary = "slod_weighted_parent" if "slod_weighted_parent" in conditions else slod_condition
        soft_baselines = [c for c in conditions if c != soft_primary]

        for k in top_k_values:
            k_str = str(k)
            primary_scores = per_question_soft_f1.get(soft_primary, {}).get(k_str)
            if primary_scores is None or len(primary_scores) == 0:
                continue

            k_tests = {}
            for baseline in soft_baselines:
                baseline_scores = per_question_soft_f1.get(baseline, {}).get(k_str)
                if baseline_scores is None or len(baseline_scores) == 0:
                    continue
                if len(baseline_scores) != len(primary_scores):
                    logging.warning(
                        f"Soft score length mismatch for k={k}: {soft_primary}={len(primary_scores)}, "
                        f"{baseline}={len(baseline_scores)}"
                    )
                    continue

                test_result = paired_bootstrap_test(
                    primary_scores,
                    baseline_scores,
                    n_bootstrap=bootstrap_cfg["n_resamples"],
                    confidence_level=bootstrap_cfg["confidence_level"],
                    random_seed=bootstrap_cfg["random_seed"],
                )
                k_tests[baseline] = test_result

                logging.info(
                    f"  [soft] k={k} {soft_primary} vs {baseline}: "
                    f"diff={test_result['observed_diff']:.4f}, "
                    f"p={test_result['p_value']:.4f}, "
                    f"sig={test_result['significant']}"
                )

            soft_bootstrap_results[k_str] = k_tests

        bootstrap_results["soft_f1"] = soft_bootstrap_results
        bootstrap_results["soft_primary_condition"] = soft_primary

    save_json(bootstrap_results, bootstrap_path)
    logging.info(f"Bootstrap tests saved to {bootstrap_path}")
