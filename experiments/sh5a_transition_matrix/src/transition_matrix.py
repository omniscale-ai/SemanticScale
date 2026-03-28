"""Stage B: Build per-trace 3x3 transition matrices (hard and soft)."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


N_LEVELS = 3  # macro=0, meso=1, micro=2


def build_hard_matrix(hard_labels: list[int]) -> np.ndarray:
    """Build a 3x3 hard transition matrix from a sequence of integer labels."""
    mat = np.zeros((N_LEVELS, N_LEVELS), dtype=np.float64)
    for i in range(len(hard_labels) - 1):
        mat[hard_labels[i], hard_labels[i + 1]] += 1.0
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def build_soft_matrix(prob_vectors: list[list[float]]) -> np.ndarray:
    """Build a 3x3 soft transition matrix from consecutive probability vectors.

    For each consecutive pair (p_i, p_{i+1}), compute the outer product
    and average across all transitions.
    """
    mat = np.zeros((N_LEVELS, N_LEVELS), dtype=np.float64)
    n_transitions = len(prob_vectors) - 1
    if n_transitions == 0:
        return mat

    for i in range(n_transitions):
        p_from = np.array(prob_vectors[i])
        p_to = np.array(prob_vectors[i + 1])
        mat += np.outer(p_from, p_to)

    mat /= n_transitions
    return mat


def build_all_matrices(traces: list[dict]) -> dict:
    """Build hard and soft transition matrices for all traces.

    Returns dict with:
      hard_matrices: (N, 3, 3) array
      soft_matrices: (N, 3, 3) array
      trace_ids: list of (question_id, condition) tuples
      metadata: list of dicts with per-trace info
    """
    n = len(traces)
    hard_matrices = np.zeros((n, N_LEVELS, N_LEVELS), dtype=np.float64)
    soft_matrices = np.zeros((n, N_LEVELS, N_LEVELS), dtype=np.float64)
    trace_ids = []
    metadata = []

    for idx, t in enumerate(traces):
        hard_matrices[idx] = build_hard_matrix(t["hard_labels"])
        soft_matrices[idx] = build_soft_matrix(t["prob_vectors"])
        trace_ids.append((t["question_id"], t["condition"]))
        metadata.append({
            "question_id": t["question_id"],
            "condition": t["condition"],
            "n_steps": t["n_steps"],
            "n_transitions": t["n_steps"] - 1,
            "answer_token_f1": t["answer_token_f1"],
            "attribution_f1": t["attribution_f1"],
            "jump_count": t["jump_count"],
            "normalized_jump_rate": t["normalized_jump_rate"],
        })

    return {
        "hard_matrices": hard_matrices,
        "soft_matrices": soft_matrices,
        "trace_ids": trace_ids,
        "metadata": metadata,
    }


def aggregate_per_question(
    matrices: np.ndarray,
    metadata: list[dict],
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Aggregate matrices per question (average across 4 conditions).

    Returns:
      agg_matrices: (Q, 3, 3) array
      question_ids: list of question_id strings
      agg_metadata: list of dicts with averaged quality metrics
    """
    # Group by question_id
    groups = defaultdict(list)
    meta_groups = defaultdict(list)
    for idx, m in enumerate(metadata):
        qid = m["question_id"]
        groups[qid].append(idx)
        meta_groups[qid].append(m)

    question_ids = sorted(groups.keys())
    n_q = len(question_ids)
    agg_matrices = np.zeros((n_q, N_LEVELS, N_LEVELS), dtype=np.float64)
    agg_metadata = []

    for qi, qid in enumerate(question_ids):
        indices = groups[qid]
        agg_matrices[qi] = matrices[indices].mean(axis=0)
        metas = meta_groups[qid]
        agg_metadata.append({
            "question_id": qid,
            "n_conditions": len(indices),
            "n_transitions_total": sum(m["n_transitions"] for m in metas),
            "mean_token_f1": np.mean([m["answer_token_f1"] for m in metas]),
            "mean_attribution_f1": np.mean([m["attribution_f1"] for m in metas]),
        })

    return agg_matrices, question_ids, agg_metadata


def save_matrices(result: dict, agg_hard: np.ndarray, agg_soft: np.ndarray,
                  question_ids: list[str], output_dir: str = "data") -> None:
    """Save all matrices and metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.savez_compressed(
        out / "transition_matrices.npz",
        hard_matrices=result["hard_matrices"],
        soft_matrices=result["soft_matrices"],
        agg_hard_matrices=agg_hard,
        agg_soft_matrices=agg_soft,
    )
    print(f"Saved matrices to {out / 'transition_matrices.npz'}")

    # Save metadata
    meta_path = out / "transition_matrices_meta.jsonl"
    with open(meta_path, "w") as f:
        for m in result["metadata"]:
            f.write(json.dumps(m) + "\n")
    print(f"Saved metadata to {meta_path}")

    # Save trace_ids and question_ids as JSON for reference
    ids_path = out / "matrix_ids.json"
    with open(ids_path, "w") as f:
        json.dump({
            "trace_ids": result["trace_ids"],
            "question_ids": question_ids,
        }, f)
    print(f"Saved IDs to {ids_path}")
