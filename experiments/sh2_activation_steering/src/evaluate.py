"""Evaluation functions for SH2: SLoD shift, surface metrics, factuality.

This module measures:
1. SLoD shift — embed outputs with SciBERT, project onto SLoD axis, compare steered vs baseline
2. Surface metrics — entity density, citation density, numeric density, sentence length
3. Factuality — token-F1 vs gold answers
"""
import re
import numpy as np
from scipy import stats


def compute_slod_scores(texts: list, embed_fn, slod_axis: np.ndarray) -> np.ndarray:
    """Embed texts with SciBERT and project onto SLoD axis.

    Args:
        texts: list of answer strings
        embed_fn: function(texts) -> (N, 768) embeddings
        slod_axis: (768,) unit vector

    Returns:
        (N,) array of scalar SLoD scores (higher = more micro)
    """
    embeddings = embed_fn(texts)  # (N, 768)
    scores = embeddings @ slod_axis  # (N,)
    return scores


def compute_slod_shift(baseline_scores: np.ndarray, steered_scores: np.ndarray) -> dict:
    """Compute SLoD shift statistics (paired comparison).

    Args:
        baseline_scores: (N,) SLoD scores for baseline answers
        steered_scores: (N,) SLoD scores for steered answers

    Returns:
        dict with mean_delta, std_delta, p_value (paired t-test), cohens_d
    """
    deltas = steered_scores - baseline_scores
    t_stat, p_value = stats.ttest_rel(steered_scores, baseline_scores)
    cohens_d = float(deltas.mean() / deltas.std()) if deltas.std() > 0 else 0.0
    return {
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
    }


def compute_surface_metrics(text: str) -> dict:
    """Compute surface-level metrics for a single text.

    Args:
        text: answer string

    Returns:
        dict with entity_density, citation_density, numeric_density, mean_sentence_length
    """
    words = text.split()
    total_words = max(len(words), 1)

    # entity_density: capitalized multi-word spans (sequences of Title Case words)
    entities = re.findall(r'(?:[A-Z][a-z]+\s+){1,}[A-Z][a-z]+', text)
    entity_density = len(entities) / total_words

    # citation_density: [N] or (Author, Year) patterns
    citations = re.findall(r'\[\d+\]|\([^)]*\d{4}[^)]*\)', text)
    citation_density = len(citations) / total_words

    # numeric_density
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    numeric_density = len(numbers) / total_words

    # mean_sentence_length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        mean_sentence_length = float(np.mean([len(s.split()) for s in sentences]))
    else:
        mean_sentence_length = float(total_words)

    return {
        "entity_density": entity_density,
        "citation_density": citation_density,
        "numeric_density": numeric_density,
        "mean_sentence_length": mean_sentence_length,
    }


def compare_surface_metrics(baseline_metrics: list, steered_metrics: list) -> dict:
    """Compare surface metrics between baseline and steered answers (paired).

    Args:
        baseline_metrics: list of dicts from compute_surface_metrics
        steered_metrics: list of dicts from compute_surface_metrics

    Returns:
        dict with per-metric comparison (mean, p-value) and n_significant count
    """
    metric_keys = ["entity_density", "citation_density", "numeric_density", "mean_sentence_length"]
    comparison = {}
    n_significant = 0

    for key in metric_keys:
        baseline_vals = np.array([m[key] for m in baseline_metrics])
        steered_vals = np.array([m[key] for m in steered_metrics])

        t_stat, p_value = stats.ttest_rel(steered_vals, baseline_vals)
        mean_baseline = float(baseline_vals.mean())
        mean_steered = float(steered_vals.mean())
        mean_delta = mean_steered - mean_baseline

        significant = bool(p_value < 0.05)
        if significant:
            n_significant += 1

        comparison[key] = {
            "mean_baseline": mean_baseline,
            "mean_steered": mean_steered,
            "mean_delta": float(mean_delta),
            "p_value": float(p_value),
            "significant": significant,
        }

    comparison["n_significant"] = n_significant
    return comparison


def evaluate_factuality(answers: list, gold_answers: list, token_f1_fn) -> dict:
    """Evaluate answer factuality vs gold references.

    Args:
        answers: list of answer strings
        gold_answers: list of gold answer strings
        token_f1_fn: function(pred, ref) -> float

    Returns:
        dict with mean_f1, per-question f1 scores
    """
    scores = [token_f1_fn(a, g) for a, g in zip(answers, gold_answers)]
    return {"mean_f1": float(np.mean(scores)), "scores": scores}
