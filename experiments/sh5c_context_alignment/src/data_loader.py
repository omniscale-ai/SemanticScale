"""Stage A: Data Loading & Context SLoD Extraction.

Loads SH5 per-step tags, answer scores, jump metrics, selected questions,
and SH3 retrieval results. Computes context SLoD profiles from retrieved
chunk doc_ids and merges everything into a single per-trace dataset.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import yaml


LEVEL_MAP = {"macro": 0, "meso": 1, "micro": 2}
LEVEL_NAMES = ["macro", "meso", "micro"]


def load_config(working_dir: Path) -> dict:
    """Load config.yaml."""
    with open(working_dir / "config.yaml") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sh5_data(working_dir: Path) -> dict:
    """Load all SH5 JSONL files into DataFrames."""
    tags = pd.DataFrame(load_jsonl(working_dir / "data_sh5" / "cot_slod_tags.jsonl"))
    scores = pd.DataFrame(load_jsonl(working_dir / "data_sh5" / "answer_scores.jsonl"))
    metrics = pd.DataFrame(load_jsonl(working_dir / "data_sh5" / "jump_metrics.jsonl"))
    questions = pd.DataFrame(load_jsonl(working_dir / "data_sh5" / "selected_questions.jsonl"))
    return {
        "tags": tags,
        "scores": scores,
        "metrics": metrics,
        "questions": questions,
    }


def load_sh3_retrieval(working_dir: Path, conditions: list[str], k: str) -> dict:
    """Load SH3 retrieval results for specified conditions and k.

    Returns dict: {(question_id, condition): [list of retrieved doc dicts]}
    """
    path = working_dir / "data_sh3" / "results" / "retrieval_results.json"
    with open(path) as f:
        data = json.load(f)

    retrieval = {}
    for cond in conditions:
        if cond not in data:
            print(f"WARNING: condition '{cond}' not found in retrieval results")
            continue
        if k not in data[cond]:
            print(f"WARNING: k={k} not found for condition '{cond}'")
            continue
        for entry in data[cond][k]:
            qid = entry["question_id"]
            retrieval[(qid, cond)] = entry.get("retrieved", [])
    return retrieval


def parse_slod_from_doc_id(doc_id: str) -> str:
    """Extract SLoD level from doc_id.

    Format: '{paper_id}__{level}__{rest}'
    e.g., '1912.01214__meso__s7_p1' -> 'meso'
    """
    parts = doc_id.split("__")
    if len(parts) >= 2 and parts[1] in LEVEL_MAP:
        return parts[1]
    return "unknown"


def compute_context_profile(retrieved_docs: list[dict]) -> dict:
    """Compute context SLoD profile from retrieved chunks.

    Returns dict with:
        context_slod_counts: {macro: n, meso: n, micro: n}
        context_slod_mean: float
        context_slod_distribution: [p_macro, p_meso, p_micro]
        context_n_docs: int
    """
    if not retrieved_docs:
        return {
            "context_slod_counts": {"macro": 0, "meso": 0, "micro": 0},
            "context_slod_mean": np.nan,
            "context_slod_distribution": [0.0, 0.0, 0.0],
            "context_n_docs": 0,
        }

    levels = []
    for doc in retrieved_docs:
        level_str = parse_slod_from_doc_id(doc["doc_id"])
        if level_str in LEVEL_MAP:
            levels.append(LEVEL_MAP[level_str])

    if not levels:
        return {
            "context_slod_counts": {"macro": 0, "meso": 0, "micro": 0},
            "context_slod_mean": np.nan,
            "context_slod_distribution": [0.0, 0.0, 0.0],
            "context_n_docs": 0,
        }

    counts = Counter(levels)
    n = len(levels)
    distribution = [counts.get(i, 0) / n for i in range(3)]

    return {
        "context_slod_counts": {
            "macro": counts.get(0, 0),
            "meso": counts.get(1, 0),
            "micro": counts.get(2, 0),
        },
        "context_slod_mean": float(np.mean(levels)),
        "context_slod_distribution": distribution,
        "context_n_docs": n,
    }


def compute_reasoning_profile(trace_rows: pd.DataFrame) -> dict:
    """Compute reasoning SLoD profile from per-step tags.

    Returns dict with:
        reasoning_slod_distribution: [p_macro, p_meso, p_micro] from hard labels
        reasoning_slod_mean: float
        reasoning_slod_soft_distribution: [p_macro, p_meso, p_micro] averaged probabilities
        reasoning_n_steps: int
        reasoning_step_slods: list of int (predicted_slod per step)
        reasoning_step_confidences: list of float (max probability per step)
        reasoning_step_probabilities: list of [p_macro, p_meso, p_micro] per step
    """
    n = len(trace_rows)
    if n == 0:
        return {
            "reasoning_slod_distribution": [0.0, 0.0, 0.0],
            "reasoning_slod_mean": np.nan,
            "reasoning_slod_soft_distribution": [0.0, 0.0, 0.0],
            "reasoning_n_steps": 0,
            "reasoning_step_slods": [],
            "reasoning_step_confidences": [],
            "reasoning_step_probabilities": [],
        }

    step_slods = trace_rows["predicted_slod"].tolist()
    counts = Counter(step_slods)
    hard_dist = [counts.get(i, 0) / n for i in range(3)]

    # Soft distribution: average of probability vectors
    probs = trace_rows["probabilities"].tolist()
    soft_dist = [0.0, 0.0, 0.0]
    for p in probs:
        soft_dist[0] += p["macro"]
        soft_dist[1] += p["meso"]
        soft_dist[2] += p["micro"]
    soft_dist = [x / n for x in soft_dist]

    # Step confidences (max probability per step)
    confidences = [max(p["macro"], p["meso"], p["micro"]) for p in probs]

    # Step probability vectors
    step_probs = [[p["macro"], p["meso"], p["micro"]] for p in probs]

    return {
        "reasoning_slod_distribution": hard_dist,
        "reasoning_slod_mean": float(np.mean(step_slods)),
        "reasoning_slod_soft_distribution": soft_dist,
        "reasoning_n_steps": n,
        "reasoning_step_slods": step_slods,
        "reasoning_step_confidences": confidences,
        "reasoning_step_probabilities": step_probs,
    }


def build_merged_traces(working_dir: Path) -> pd.DataFrame:
    """Build merged dataset with context and reasoning SLoD profiles.

    Returns DataFrame with one row per (question_id, condition).
    """
    cfg = load_config(working_dir)
    conditions = cfg["data"]["conditions"]
    k = cfg["data"]["retrieval_k"]

    # Load data
    sh5 = load_sh5_data(working_dir)
    retrieval = load_sh3_retrieval(working_dir, conditions, k)

    # Get the set of SH5 question_ids
    sh5_qids = set(sh5["questions"]["question_id"].tolist())

    # Group tags by (question_id, condition)
    tags = sh5["tags"]
    grouped_tags = tags.groupby(["question_id", "condition"])

    rows = []
    for _, score_row in sh5["scores"].iterrows():
        qid = score_row["question_id"]
        cond = score_row["condition"]

        # Get context profile from SH3
        retrieved = retrieval.get((qid, cond), [])
        context = compute_context_profile(retrieved)

        # Get reasoning profile from SH5 tags
        try:
            trace = grouped_tags.get_group((qid, cond))
        except KeyError:
            continue
        reasoning = compute_reasoning_profile(trace.sort_values("step_index"))

        # Get jump metrics
        metrics_row = sh5["metrics"][
            (sh5["metrics"]["question_id"] == qid) & (sh5["metrics"]["condition"] == cond)
        ]

        # Get question metadata
        q_row = sh5["questions"][sh5["questions"]["question_id"] == qid]
        answer_type = score_row.get("answer_type", "unknown")

        row = {
            "question_id": qid,
            "condition": cond,
            "answer_token_f1": score_row["answer_token_f1"],
            "attribution_f1": score_row["attribution_f1"],
            "answer_type": answer_type,
            # Context profile
            **context,
            # Reasoning profile
            **reasoning,
        }

        # Add jump metrics if available
        if len(metrics_row) > 0:
            m = metrics_row.iloc[0]
            for col in ["n_steps", "jump_count", "mean_abs_delta", "max_jump",
                        "slod_variance", "direction_changes", "normalized_jump_rate"]:
                row[col] = m[col]

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built merged traces: {len(df)} rows, "
          f"{df['question_id'].nunique()} unique questions, "
          f"{df['condition'].nunique()} conditions")
    print(f"Context coverage: {(df['context_n_docs'] > 0).sum()}/{len(df)} traces have retrieval data")

    return df


def save_merged_traces(df: pd.DataFrame, output_path: Path):
    """Save merged traces to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), default=_json_serialize) + "\n")
    print(f"Saved {len(df)} traces to {output_path}")


def _json_serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Type {type(obj)} not serializable")
