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
    # TODO: Engineer implements this
    # embeddings = embed_fn(texts)
    # scores = embeddings @ slod_axis
    # return scores
    raise NotImplementedError("Engineer agent should implement this")


def compute_slod_shift(baseline_scores: np.ndarray, steered_scores: np.ndarray) -> dict:
    """Compute SLoD shift statistics (paired comparison).

    Args:
        baseline_scores: (N,) SLoD scores for baseline answers
        steered_scores: (N,) SLoD scores for steered answers

    Returns:
        dict with mean_delta, std_delta, p_value (paired t-test), cohens_d
    """
    # TODO: Engineer implements this
    # deltas = steered_scores - baseline_scores
    # t_stat, p_value = stats.ttest_rel(steered_scores, baseline_scores)
    # cohens_d = deltas.mean() / deltas.std()
    # return {"mean_delta": ..., "std_delta": ..., "p_value": ..., "cohens_d": ...}
    raise NotImplementedError("Engineer agent should implement this")


def compute_surface_metrics(text: str) -> dict:
    """Compute surface-level metrics for a single text.

    Args:
        text: answer string

    Returns:
        dict with entity_density, citation_density, numeric_density, mean_sentence_length
    """
    # TODO: Engineer implements this
    # Key patterns:
    # - entity_density: count capitalized multi-word spans / total_words
    # - citation_density: re.findall(r'\[\d+\]|\([^)]*\d{4}[^)]*\)', text) / total_words
    # - numeric_density: re.findall(r'\b\d+\.?\d*\b', text) / total_words
    # - mean_sentence_length: split by sentence boundaries, mean word count
    raise NotImplementedError("Engineer agent should implement this")


def compare_surface_metrics(baseline_metrics: list, steered_metrics: list) -> dict:
    """Compare surface metrics between baseline and steered answers (paired).

    Args:
        baseline_metrics: list of dicts from compute_surface_metrics
        steered_metrics: list of dicts from compute_surface_metrics

    Returns:
        dict with per-metric comparison (mean, p-value) and n_significant count
    """
    # TODO: Engineer implements this
    # For each metric: paired t-test between baseline and steered values
    raise NotImplementedError("Engineer agent should implement this")


def evaluate_factuality(answers: list, gold_answers: list, token_f1_fn) -> dict:
    """Evaluate answer factuality vs gold references.

    Args:
        answers: list of answer strings
        gold_answers: list of gold answer strings
        token_f1_fn: function(pred, ref) -> float

    Returns:
        dict with mean_f1, per-question f1 scores
    """
    # TODO: Engineer implements this
    # scores = [token_f1_fn(a, g) for a, g in zip(answers, gold_answers)]
    # return {"mean_f1": np.mean(scores), "scores": scores}
    raise NotImplementedError("Engineer agent should implement this")
