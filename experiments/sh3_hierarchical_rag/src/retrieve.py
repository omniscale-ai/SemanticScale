"""Retrieval logic for SH3: 13 conditions (chunks-only, summaries-only, naive-hybrid,
naive-hybrid-parent, SLoD-routed, SLoD-routed-v2, SLoD-weighted, SLoD-weighted-parent,
naive-hybrid-bm25, slod-weighted-bm25, slod-weighted-parent-bm25,
naive-hybrid-rerank, slod-weighted-parent-bm25-rerank)."""

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.utils import load_config, load_json, load_jsonl, save_json


def _tokenize_bm25(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing for BM25."""
    return text.lower().split()


def build_bm25_index(documents: list[str]) -> BM25Okapi:
    """Build a BM25Okapi index over document texts.

    Args:
        documents: List of document texts.

    Returns:
        BM25Okapi index.
    """
    tokenized = [_tokenize_bm25(doc) for doc in documents]
    return BM25Okapi(tokenized)


def bm25_score_query(bm25_index: BM25Okapi, query_text: str) -> np.ndarray:
    """Score a query against a BM25 index, returning min-max normalized scores in [0, 1].

    Args:
        bm25_index: Pre-built BM25Okapi index.
        query_text: Raw query string.

    Returns:
        np.ndarray of normalized scores in [0, 1], same length as index documents.
    """
    tokenized_query = _tokenize_bm25(query_text)
    raw_scores = bm25_index.get_scores(tokenized_query)

    # Min-max normalization to [0, 1]
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    if max_score - min_score > 0:
        normalized = (raw_scores - min_score) / (max_score - min_score)
    else:
        normalized = np.zeros_like(raw_scores)

    return normalized


def _collect_all_level_docs(paper_id: str, paper_lookups: dict) -> tuple:
    """Collect all documents across macro/meso/micro levels for a paper.

    Returns:
        Tuple of (all_embs_list, all_texts, all_doc_ids, all_levels).
        all_embs_list is a list of numpy arrays to be concatenated.
    """
    all_embs = []
    all_texts = []
    all_doc_ids = []
    all_levels = []
    for level in ["macro", "meso", "micro"]:
        paper_data = paper_lookups.get(level, {}).get(paper_id)
        if paper_data:
            n = paper_data["embeddings"].shape[0]
            all_embs.append(paper_data["embeddings"])
            all_texts.extend(paper_data["texts"])
            all_doc_ids.extend(paper_data["doc_ids"])
            all_levels.extend([level] * n)
    return all_embs, all_texts, all_doc_ids, all_levels


def retrieve_for_paper(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    doc_texts: list[str],
    doc_ids: list[str],
    top_k: int,
) -> list[dict]:
    """Cosine similarity retrieval within a single paper's documents.

    Args:
        query_emb: (1, dim) or (dim,) query embedding.
        doc_embs: (N, dim) document embeddings.
        doc_texts: List of document texts.
        doc_ids: List of document IDs.
        top_k: Number of top documents to return.

    Returns:
        List of {doc_id, text, score, rank} dicts, sorted by score descending.
    """
    if doc_embs.shape[0] == 0:
        return []

    query_emb = query_emb.reshape(1, -1)
    scores = cosine_similarity(query_emb, doc_embs)[0]

    # Cap k at available documents
    k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "doc_id": doc_ids[idx],
            "text": doc_texts[idx],
            "score": float(scores[idx]),
            "rank": rank + 1,
        })
    return results


def _build_paper_lookup(level_data: dict, level_name: str) -> dict:
    """Build paper_id -> {embeddings, texts, doc_ids} lookup for a level.

    Args:
        level_data: Loaded .npz with embeddings, paper_ids, doc_ids, texts.
        level_name: Level name for logging.

    Returns:
        Dict: paper_id -> {embeddings: ndarray, texts: list, doc_ids: list}
    """
    embeddings = level_data["embeddings"]
    paper_ids = level_data["paper_ids"]
    doc_ids = level_data["doc_ids"]
    texts = level_data["texts"]

    lookup = defaultdict(lambda: {"emb_indices": []})
    for i, pid in enumerate(paper_ids):
        pid_str = str(pid)
        lookup[pid_str]["emb_indices"].append(i)

    # Convert to arrays
    result = {}
    for pid, info in lookup.items():
        indices = info["emb_indices"]
        result[pid] = {
            "embeddings": embeddings[indices],
            "texts": [str(texts[i]) for i in indices],
            "doc_ids": [str(doc_ids[i]) for i in indices],
        }

    logging.info(f"Built {level_name} lookup: {len(result)} papers, {len(embeddings)} total docs")
    return result


def build_micro_to_meso_parent_map(meso_lookup: dict) -> dict:
    """Build a mapping from micro doc_id to parent meso doc text.

    Micro doc_id format: {paper_id}__micro__s{sec}_p{para}_c{chunk}
    Parent meso doc_id:  {paper_id}__meso__s{sec}_p{para}

    Caption micro docs (section_index=-1) have no meso parent and are skipped.

    Args:
        meso_lookup: Dict paper_id -> {texts: list, doc_ids: list, ...}

    Returns:
        Dict: micro_doc_id_prefix -> {"text": str, "doc_id": str}
        where prefix is "{paper_id}__micro__s{sec}_p{para}" (without _c{chunk}).
    """
    # Build a mapping: meso_doc_id -> text
    meso_text_map = {}
    for paper_id, paper_data in meso_lookup.items():
        for doc_id, text in zip(paper_data["doc_ids"], paper_data["texts"]):
            meso_text_map[doc_id] = text

    return meso_text_map


def _micro_doc_id_to_meso_doc_id(micro_doc_id: str) -> str | None:
    """Convert a micro doc_id to its parent meso doc_id.

    micro: {paper_id}__micro__s{sec}_p{para}_c{chunk}
    meso:  {paper_id}__meso__s{sec}_p{para}

    Returns None for caption micro docs (no parent).
    """
    # Match pattern: {paper_id}__micro__s{sec}_p{para}_c{chunk}
    m = re.match(r'^(.+)__micro__s(\d+)_p(\d+)_c\d+$', micro_doc_id)
    if m:
        paper_id, sec, para = m.group(1), m.group(2), m.group(3)
        return f"{paper_id}__meso__s{sec}_p{para}"
    return None


def expand_with_parents(
    retrieved: list[dict],
    meso_text_map: dict,
) -> list[dict]:
    """Expand retrieved results by adding parent meso paragraphs for micro chunks.

    For each micro chunk in the retrieved set, looks up the parent meso paragraph
    and adds it to the results (deduplicating by doc_id). The parent documents
    are appended after the original results with score=0 and rank adjusted.

    Args:
        retrieved: List of retrieved doc dicts from run_condition.
        meso_text_map: Dict meso_doc_id -> text.

    Returns:
        Expanded list of retrieved doc dicts (may be larger than top_k).
    """
    existing_doc_ids = {r["doc_id"] for r in retrieved}
    parents_to_add = {}

    for r in retrieved:
        doc_id = r["doc_id"]
        if "__micro__" not in doc_id:
            continue

        parent_id = _micro_doc_id_to_meso_doc_id(doc_id)
        if parent_id is None:
            continue
        if parent_id in existing_doc_ids:
            continue
        if parent_id in parents_to_add:
            continue

        parent_text = meso_text_map.get(parent_id)
        if parent_text:
            parents_to_add[parent_id] = parent_text

    if not parents_to_add:
        return retrieved

    # Append parents after original results
    expanded = list(retrieved)
    next_rank = len(expanded) + 1
    for parent_id, parent_text in parents_to_add.items():
        expanded.append({
            "doc_id": parent_id,
            "text": parent_text,
            "score": 0.0,  # Not scored via cosine sim, added as context expansion
            "rank": next_rank,
        })
        next_rank += 1

    return expanded


def run_condition(
    condition_name: str,
    query_emb: np.ndarray,
    paper_id: str,
    paper_lookups: dict,
    top_k: int,
    slod_pred: str | None = None,
    slod_confidence: float | None = None,
    config: dict | None = None,
    query_text: str | None = None,
    **kwargs,
) -> list[dict]:
    """Run retrieval for one query under one condition.

    Args:
        condition_name: One of chunks_only, summaries_only, naive_hybrid, naive_hybrid_parent,
            slod_routed, slod_routed_v2, slod_weighted, slod_weighted_parent,
            naive_hybrid_bm25, slod_weighted_bm25, slod_weighted_parent_bm25,
            naive_hybrid_rerank, slod_weighted_parent_bm25_rerank.
        query_emb: Query embedding vector.
        paper_id: Target paper ID.
        paper_lookups: Dict of level -> paper_id -> {embeddings, texts, doc_ids}.
        top_k: Number of results to return.
        slod_pred: Predicted SLoD level (for slod_routed/slod_routed_v2 conditions).
        slod_confidence: Confidence score for the SLoD prediction (for slod_routed_v2).
        config: Full config dict (needed for slod_routed_v2 threshold).
        query_text: Raw query text (needed for BM25 and rerank conditions).
        **kwargs: Additional keyword arguments (cross_encoder, first_stage_k for rerank conditions).

    Returns:
        List of retrieved document dicts.
    """
    if condition_name == "chunks_only":
        # Meso level only
        paper_data = paper_lookups.get("meso", {}).get(paper_id)
        if not paper_data:
            return []
        return retrieve_for_paper(
            query_emb, paper_data["embeddings"],
            paper_data["texts"], paper_data["doc_ids"], top_k,
        )

    elif condition_name == "summaries_only":
        # Macro level only
        paper_data = paper_lookups.get("macro", {}).get(paper_id)
        if not paper_data:
            return []
        return retrieve_for_paper(
            query_emb, paper_data["embeddings"],
            paper_data["texts"], paper_data["doc_ids"], top_k,
        )

    elif condition_name == "naive_hybrid":
        # All three levels combined
        all_embs = []
        all_texts = []
        all_doc_ids = []
        for level in ["macro", "meso", "micro"]:
            paper_data = paper_lookups.get(level, {}).get(paper_id)
            if paper_data:
                all_embs.append(paper_data["embeddings"])
                all_texts.extend(paper_data["texts"])
                all_doc_ids.extend(paper_data["doc_ids"])

        if not all_embs:
            return []

        combined_embs = np.concatenate(all_embs, axis=0)
        return retrieve_for_paper(
            query_emb, combined_embs, all_texts, all_doc_ids, top_k,
        )

    elif condition_name == "slod_routed":
        # Route to the predicted SLoD level (original v1 logic, kept for comparison)
        if slod_pred is None:
            logging.warning("No SLoD prediction for slod_routed condition, defaulting to meso")
            slod_pred = "meso"

        paper_data = paper_lookups.get(slod_pred, {}).get(paper_id)
        if not paper_data:
            return []
        return retrieve_for_paper(
            query_emb, paper_data["embeddings"],
            paper_data["texts"], paper_data["doc_ids"], top_k,
        )

    elif condition_name == "slod_routed_v2":
        # Improved routing: macro→meso fallback + confidence-based hybrid fallback
        if slod_pred is None:
            logging.warning("No SLoD prediction for slod_routed_v2 condition, defaulting to meso")
            slod_pred = "meso"

        # Get confidence threshold from config
        conf_threshold = 0.6
        if config is not None:
            conf_threshold = config.get("retrieval", {}).get(
                "routing_confidence_threshold", 0.6
            )

        slod_conf = slod_confidence if slod_confidence is not None else 1.0

        # Change 1: Disable macro routing — route macro-classified queries to meso
        if slod_pred == "macro":
            slod_pred = "meso"

        # Change 2: Low confidence → fall back to naive_hybrid
        if slod_conf < conf_threshold:
            return run_condition(
                "naive_hybrid", query_emb, paper_id,
                paper_lookups, top_k,
            )

        # High confidence: route to predicted level (meso or micro)
        paper_data = paper_lookups.get(slod_pred, {}).get(paper_id)
        if not paper_data:
            return []
        return retrieve_for_paper(
            query_emb, paper_data["embeddings"],
            paper_data["texts"], paper_data["doc_ids"], top_k,
        )

    elif condition_name == "slod_weighted":
        # Soft-weighted hybrid: all levels in pool, predicted level boosted
        if slod_pred is None:
            slod_pred = "meso"

        # Disable macro boost (same rationale as v2: macro index is weak)
        effective_pred = "meso" if slod_pred == "macro" else slod_pred

        boost_base = 0.5
        if config is not None:
            boost_base = config.get("retrieval", {}).get(
                "slod_weighted", {}
            ).get("boost_base", 0.5)

        slod_conf = slod_confidence if slod_confidence is not None else 0.5

        # Collect all documents with level tags (like naive_hybrid)
        all_embs = []
        all_texts = []
        all_doc_ids = []
        all_levels = []
        for level in ["macro", "meso", "micro"]:
            paper_data = paper_lookups.get(level, {}).get(paper_id)
            if paper_data:
                n = paper_data["embeddings"].shape[0]
                all_embs.append(paper_data["embeddings"])
                all_texts.extend(paper_data["texts"])
                all_doc_ids.extend(paper_data["doc_ids"])
                all_levels.extend([level] * n)

        if not all_embs:
            return []

        combined_embs = np.concatenate(all_embs, axis=0)
        query_emb_2d = query_emb.reshape(1, -1)
        scores = cosine_similarity(query_emb_2d, combined_embs)[0]

        # Apply confidence-weighted boost to predicted level
        boost = 1.0 + boost_base * slod_conf
        for i, level in enumerate(all_levels):
            if level == effective_pred:
                scores[i] *= boost

        # Rank and return top-k
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "doc_id": all_doc_ids[idx],
                "text": all_texts[idx],
                "score": float(scores[idx]),
                "rank": rank + 1,
            })
        return results

    elif condition_name == "naive_hybrid_parent":
        # Naive hybrid with parent paragraph expansion for micro chunks
        base_results = run_condition(
            "naive_hybrid", query_emb, paper_id,
            paper_lookups, top_k,
        )
        meso_text_map = build_micro_to_meso_parent_map(paper_lookups.get("meso", {}))
        return expand_with_parents(base_results, meso_text_map)

    elif condition_name == "slod_weighted_parent":
        # SLoD-weighted with parent paragraph expansion for micro chunks
        base_results = run_condition(
            "slod_weighted", query_emb, paper_id,
            paper_lookups, top_k,
            slod_pred=slod_pred, slod_confidence=slod_confidence, config=config,
        )
        meso_text_map = build_micro_to_meso_parent_map(paper_lookups.get("meso", {}))
        return expand_with_parents(base_results, meso_text_map)

    elif condition_name == "naive_hybrid_bm25":
        # Naive hybrid with BM25 sparse signal: score = alpha * cosine + (1-alpha) * bm25
        if query_text is None:
            logging.warning("No query_text for naive_hybrid_bm25, falling back to naive_hybrid")
            return run_condition("naive_hybrid", query_emb, paper_id, paper_lookups, top_k)

        alpha = 0.7
        if config is not None:
            alpha = config.get("retrieval", {}).get("bm25", {}).get("alpha", 0.7)

        all_embs, all_texts, all_doc_ids, all_levels = _collect_all_level_docs(paper_id, paper_lookups)
        if not all_embs:
            return []

        combined_embs = np.concatenate(all_embs, axis=0)
        query_emb_2d = query_emb.reshape(1, -1)
        cosine_scores = cosine_similarity(query_emb_2d, combined_embs)[0]

        # Normalize cosine scores to [0, 1] (they can be negative for non-normalized embeddings)
        cos_min, cos_max = cosine_scores.min(), cosine_scores.max()
        if cos_max - cos_min > 0:
            cosine_norm = (cosine_scores - cos_min) / (cos_max - cos_min)
        else:
            cosine_norm = np.zeros_like(cosine_scores)

        # BM25 scoring
        bm25_index = build_bm25_index(all_texts)
        bm25_scores = bm25_score_query(bm25_index, query_text)

        # Hybrid combination
        final_scores = alpha * cosine_norm + (1 - alpha) * bm25_scores

        k = min(top_k, len(final_scores))
        top_indices = np.argsort(final_scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "doc_id": all_doc_ids[idx],
                "text": all_texts[idx],
                "score": float(final_scores[idx]),
                "rank": rank + 1,
            })
        return results

    elif condition_name == "slod_weighted_bm25":
        # SLoD-weighted hybrid with BM25: hybrid score + SLoD boost
        if query_text is None:
            logging.warning("No query_text for slod_weighted_bm25, falling back to slod_weighted")
            return run_condition(
                "slod_weighted", query_emb, paper_id, paper_lookups, top_k,
                slod_pred=slod_pred, slod_confidence=slod_confidence, config=config,
            )

        if slod_pred is None:
            slod_pred = "meso"

        # Disable macro boost (same rationale as slod_weighted)
        effective_pred = "meso" if slod_pred == "macro" else slod_pred

        alpha = 0.7
        if config is not None:
            alpha = config.get("retrieval", {}).get("bm25", {}).get("alpha", 0.7)

        boost_base = 0.5
        if config is not None:
            boost_base = config.get("retrieval", {}).get(
                "slod_weighted", {}
            ).get("boost_base", 0.5)

        slod_conf = slod_confidence if slod_confidence is not None else 0.5

        all_embs, all_texts, all_doc_ids, all_levels = _collect_all_level_docs(paper_id, paper_lookups)
        if not all_embs:
            return []

        combined_embs = np.concatenate(all_embs, axis=0)
        query_emb_2d = query_emb.reshape(1, -1)
        cosine_scores = cosine_similarity(query_emb_2d, combined_embs)[0]

        # Normalize cosine scores to [0, 1]
        cos_min, cos_max = cosine_scores.min(), cosine_scores.max()
        if cos_max - cos_min > 0:
            cosine_norm = (cosine_scores - cos_min) / (cos_max - cos_min)
        else:
            cosine_norm = np.zeros_like(cosine_scores)

        # BM25 scoring
        bm25_index = build_bm25_index(all_texts)
        bm25_scores = bm25_score_query(bm25_index, query_text)

        # Hybrid combination
        scores = alpha * cosine_norm + (1 - alpha) * bm25_scores

        # Apply SLoD boost AFTER hybrid combination
        boost = 1.0 + boost_base * slod_conf
        for i, level in enumerate(all_levels):
            if level == effective_pred:
                scores[i] *= boost

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "doc_id": all_doc_ids[idx],
                "text": all_texts[idx],
                "score": float(scores[idx]),
                "rank": rank + 1,
            })
        return results

    elif condition_name == "slod_weighted_parent_bm25":
        # SLoD-weighted BM25 hybrid with parent paragraph expansion
        base_results = run_condition(
            "slod_weighted_bm25", query_emb, paper_id,
            paper_lookups, top_k,
            slod_pred=slod_pred, slod_confidence=slod_confidence,
            config=config, query_text=query_text,
        )
        meso_text_map = build_micro_to_meso_parent_map(paper_lookups.get("meso", {}))
        return expand_with_parents(base_results, meso_text_map)

    elif condition_name == "naive_hybrid_rerank":
        # Two-stage: naive_hybrid (first_stage_k) → cross-encoder re-rank → top_k
        # cross_encoder and first_stage_k are passed via kwargs
        cross_encoder = kwargs.get("cross_encoder")
        first_stage_k = kwargs.get("first_stage_k", 50)
        if cross_encoder is None or query_text is None:
            logging.warning("naive_hybrid_rerank requires cross_encoder and query_text, "
                            "falling back to naive_hybrid")
            return run_condition("naive_hybrid", query_emb, paper_id, paper_lookups, top_k)

        from src.rerank import rerank_candidates

        # First stage: retrieve larger candidate set
        candidates = run_condition(
            "naive_hybrid", query_emb, paper_id,
            paper_lookups, first_stage_k,
        )
        # Second stage: re-rank with cross-encoder
        return rerank_candidates(cross_encoder, query_text, candidates, top_k)

    elif condition_name == "slod_weighted_parent_bm25_rerank":
        # Two-stage: slod_weighted_parent_bm25 (first_stage_k) → cross-encoder re-rank → top_k
        # Parent expansion happens in the first stage (before re-ranking)
        cross_encoder = kwargs.get("cross_encoder")
        first_stage_k = kwargs.get("first_stage_k", 50)
        if cross_encoder is None or query_text is None:
            logging.warning("slod_weighted_parent_bm25_rerank requires cross_encoder and query_text, "
                            "falling back to slod_weighted_parent_bm25")
            return run_condition(
                "slod_weighted_parent_bm25", query_emb, paper_id, paper_lookups, top_k,
                slod_pred=slod_pred, slod_confidence=slod_confidence,
                config=config, query_text=query_text,
            )

        from src.rerank import rerank_candidates

        # First stage: retrieve larger candidate set (includes parent expansion)
        candidates = run_condition(
            "slod_weighted_parent_bm25", query_emb, paper_id,
            paper_lookups, first_stage_k,
            slod_pred=slod_pred, slod_confidence=slod_confidence,
            config=config, query_text=query_text,
        )
        # Second stage: re-rank with cross-encoder
        return rerank_candidates(cross_encoder, query_text, candidates, top_k)

    else:
        raise ValueError(f"Unknown condition: {condition_name}")


def run_all_retrieval(config: dict, project_root: Path | None = None) -> None:
    """Orchestrate retrieval: for each question x condition x k, retrieve and save results."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "retrieval_results.json"
    if out_path.exists():
        logging.info("Retrieval results already exist, skipping.")
        return

    # Load questions
    questions = load_jsonl(data_dir / "questions.jsonl")
    logging.info(f"Loaded {len(questions)} questions")

    # Determine which embedding model to use for retrieval
    retrieval_model = config.get("embedding", {}).get("retrieval_model", "minilm")
    logging.info(f"Using retrieval embedding model: {retrieval_model}")

    # Select embedding file names based on retrieval model
    if retrieval_model == "specter2":
        query_emb_file = "specter2_queries.npz"
        level_emb_pattern = "specter2_{level}.npz"
    else:
        query_emb_file = "query_embeddings.npz"
        level_emb_pattern = "{level}_embeddings.npz"

    # Load query embeddings
    query_data = np.load(data_dir / "embeddings" / query_emb_file, allow_pickle=True)
    query_embeddings = query_data["embeddings"]
    query_ids = [str(qid) for qid in query_data["question_ids"]]

    # Build question_id -> index mapping
    qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}

    # Load SLoD predictions (configurable via retrieval.active_predictions_file)
    predictions_file = config.get("retrieval", {}).get(
        "active_predictions_file", None
    )
    if predictions_file is None:
        # Legacy fallback: check query_classifier_v2.predictions_file, then v1
        predictions_file = config.get("query_classifier_v2", {}).get(
            "predictions_file", "query_slod_predictions.json"
        )
    slod_preds_path = data_dir / predictions_file
    if not slod_preds_path.exists():
        # Fall back to v1 predictions
        logging.warning(f"Predictions file {slod_preds_path} not found, falling back to v1")
        slod_preds_path = data_dir / "query_slod_predictions.json"
    slod_preds = load_json(slod_preds_path)
    logging.info(f"Loaded SLoD predictions from {slod_preds_path.name} ({len(slod_preds)} entries)")

    # Load document embeddings and build per-paper lookups
    logging.info("Building per-paper document lookups...")
    paper_lookups = {}
    for level in ["macro", "meso", "micro"]:
        npz_path = data_dir / "embeddings" / level_emb_pattern.format(level=level)
        level_data = np.load(npz_path, allow_pickle=True)
        paper_lookups[level] = _build_paper_lookup(level_data, level)

    # Retrieval conditions and k values
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]

    # Load cross-encoder model if any rerank conditions are present
    rerank_conditions = {"naive_hybrid_rerank", "slod_weighted_parent_bm25_rerank"}
    needs_rerank = bool(rerank_conditions & set(conditions))
    cross_encoder = None
    first_stage_k = 50
    if needs_rerank:
        rerank_cfg = config.get("retrieval", {}).get("rerank", {})
        rerank_model = rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        first_stage_k = rerank_cfg.get("first_stage_k", 50)
        from src.rerank import load_cross_encoder
        cross_encoder = load_cross_encoder(rerank_model)

    # Run retrieval
    all_results = {}

    # Mapping from rerank condition to its base condition
    rerank_base_map = {
        "naive_hybrid_rerank": "naive_hybrid",
        "slod_weighted_parent_bm25_rerank": "slod_weighted_parent_bm25",
    }

    for condition in conditions:
        logging.info(f"Running condition: {condition}")
        is_rerank = condition in rerank_conditions

        if is_rerank:
            # OPTIMIZED: For rerank conditions, retrieve+rerank ONCE per query
            # at first_stage_k, then slice for each k value.
            from src.rerank import rerank_candidates

            base_condition = rerank_base_map[condition]
            max_k = max(top_k_values)

            # Retrieve + rerank all queries once at max(first_stage_k, max_k)
            reranked_per_query = {}  # qid -> list of reranked docs (up to first_stage_k)

            for q in tqdm(questions, desc=f"{condition} rerank"):
                qid = q["question_id"]
                paper_id = q["paper_id"]

                q_idx = qid_to_idx.get(qid)
                if q_idx is None:
                    logging.warning(f"No embedding found for question {qid}")
                    reranked_per_query[qid] = {"paper_id": paper_id, "split": q.get("split", ""), "retrieved": []}
                    continue

                query_emb = query_embeddings[q_idx]
                query_text = q.get("question_text")

                # Get SLoD prediction for base conditions that need it
                slod_pred = None
                slod_confidence = None
                slod_conditions = (
                    "slod_routed", "slod_routed_v2", "slod_weighted",
                    "slod_weighted_parent", "slod_weighted_bm25",
                    "slod_weighted_parent_bm25",
                    "slod_weighted_parent_bm25_rerank",
                )
                if condition in slod_conditions and qid in slod_preds:
                    slod_pred = slod_preds[qid]["predicted_slod"]
                    slod_confidence = slod_preds[qid]["probabilities"].get(slod_pred)

                try:
                    # First stage: retrieve candidates using base condition
                    candidates = run_condition(
                        base_condition, query_emb, paper_id,
                        paper_lookups, first_stage_k,
                        slod_pred=slod_pred, slod_confidence=slod_confidence,
                        config=config, query_text=query_text,
                    )
                    # Second stage: re-rank with cross-encoder (keep all for slicing)
                    reranked = rerank_candidates(
                        cross_encoder, query_text, candidates, first_stage_k,
                    )
                except Exception as e:
                    logging.warning(f"Retrieval error for {qid} ({condition}): {e}")
                    reranked = []

                reranked_per_query[qid] = {
                    "paper_id": paper_id,
                    "split": q.get("split", ""),
                    "retrieved": reranked,
                }

            # Now slice for each k value
            condition_results = {}
            for k in top_k_values:
                k_results = []
                for q in questions:
                    qid = q["question_id"]
                    entry = reranked_per_query[qid]
                    # Slice top-k and re-number ranks
                    sliced = entry["retrieved"][:k]
                    for rank, doc in enumerate(sliced):
                        doc["rank"] = rank + 1
                    k_results.append({
                        "question_id": qid,
                        "paper_id": entry["paper_id"],
                        "split": entry["split"],
                        "retrieved": sliced,
                    })
                condition_results[str(k)] = k_results

        else:
            # Standard retrieval: per k, per query
            condition_results = {}
            for k in top_k_values:
                k_results = []

                for q in tqdm(questions, desc=f"{condition} k={k}"):
                    qid = q["question_id"]
                    paper_id = q["paper_id"]

                    # Get query embedding
                    q_idx = qid_to_idx.get(qid)
                    if q_idx is None:
                        logging.warning(f"No embedding found for question {qid}")
                        k_results.append({
                            "question_id": qid,
                            "paper_id": paper_id,
                            "retrieved": [],
                        })
                        continue

                    query_emb = query_embeddings[q_idx]

                    # Get SLoD prediction and confidence for routed conditions
                    slod_pred = None
                    slod_confidence = None
                    slod_conditions = (
                        "slod_routed", "slod_routed_v2", "slod_weighted",
                        "slod_weighted_parent", "slod_weighted_bm25",
                        "slod_weighted_parent_bm25",
                    )
                    if condition in slod_conditions and qid in slod_preds:
                        slod_pred = slod_preds[qid]["predicted_slod"]
                        slod_confidence = slod_preds[qid]["probabilities"].get(slod_pred)

                    # Get query text for BM25 conditions
                    query_text = q.get("question_text") if condition.endswith("_bm25") else None

                    # Retrieve
                    try:
                        retrieved = run_condition(
                            condition, query_emb, paper_id,
                            paper_lookups, k, slod_pred=slod_pred,
                            slod_confidence=slod_confidence, config=config,
                            query_text=query_text,
                        )
                    except Exception as e:
                        logging.warning(f"Retrieval error for {qid} ({condition}, k={k}): {e}")
                        retrieved = []

                    k_results.append({
                        "question_id": qid,
                        "paper_id": paper_id,
                        "split": q.get("split", ""),
                        "retrieved": retrieved,
                    })

                condition_results[str(k)] = k_results

        all_results[condition] = condition_results

    save_json(all_results, out_path)
    logging.info(f"Retrieval results saved to {out_path}")
