"""Cross-encoder re-ranking module for SH3.

Two-stage retrieval: first retrieve a larger candidate set using existing methods,
then re-rank with a cross-encoder and keep top-k.
"""

import logging

import numpy as np
from sentence_transformers import CrossEncoder


def load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load a cross-encoder model (cached after first call per model name).

    Args:
        model_name: HuggingFace model name, e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2'.

    Returns:
        CrossEncoder instance.
    """
    logging.info(f"Loading cross-encoder model: {model_name}")
    model = CrossEncoder(model_name)
    logging.info(f"Cross-encoder loaded: {model_name}")
    return model


def rerank_candidates(
    cross_encoder: CrossEncoder,
    query_text: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """Re-rank candidate documents using cross-encoder scores.

    Args:
        cross_encoder: Loaded CrossEncoder model.
        query_text: Raw query string.
        candidates: List of candidate dicts with at least 'doc_id' and 'text' keys.
        top_k: Number of top results to return after re-ranking.

    Returns:
        List of top-k re-ranked document dicts with updated scores and ranks.
    """
    if not candidates:
        return []

    # Build (query, doc) pairs for cross-encoder
    pairs = [[query_text, c["text"]] for c in candidates]

    # Cross-encoder scores all pairs in one batch
    scores = cross_encoder.predict(pairs)

    # Sort by cross-encoder score descending
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-k with updated scores and ranks
    results = []
    for rank, (score, cand) in enumerate(scored[:top_k]):
        results.append({
            "doc_id": cand["doc_id"],
            "text": cand["text"],
            "score": float(score),
            "rank": rank + 1,
        })

    return results
