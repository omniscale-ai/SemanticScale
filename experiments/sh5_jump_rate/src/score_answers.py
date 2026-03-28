"""Stage E: Score CoT final answers against gold answers (token-F1 + attribution F1)."""

import json
import logging
from pathlib import Path

from src.utils import (
    load_jsonl,
    normalize_text,
    resolve_path,
    save_jsonl,
    token_f1,
    tokenize_for_f1,
)

logger = logging.getLogger(__name__)


def score_answer_token_f1(
    predicted: str,
    gold: str,
    answer_type: str,
    gold_yes_no: bool | None = None,
) -> float:
    """Compute token-F1 for one answer.

    For yes_no questions: exact match (1.0 or 0.0).
    For extractive/abstractive: token-level F1.
    """
    if answer_type == "yes_no":
        # Parse predicted as yes/no
        pred_lower = predicted.lower().strip()
        pred_yn = None
        if "yes" in pred_lower and "no" not in pred_lower:
            pred_yn = True
        elif "no" in pred_lower and "yes" not in pred_lower:
            pred_yn = False
        elif pred_lower.startswith("yes"):
            pred_yn = True
        elif pred_lower.startswith("no"):
            pred_yn = False

        if gold_yes_no is not None and pred_yn is not None:
            return 1.0 if pred_yn == gold_yes_no else 0.0
        # Fallback: treat as text comparison
        if not gold:
            return 0.0

    if not gold or not predicted:
        return 0.0

    pred_tokens = tokenize_for_f1(predicted, remove_stopwords=True)
    gold_tokens = tokenize_for_f1(gold, remove_stopwords=True)

    result = token_f1(pred_tokens, gold_tokens)
    return result["f1"]


def compute_attribution_f1(
    retrieved_texts: list[str],
    gold_evidence: list[str],
    threshold: float = 0.5,
) -> float:
    """Compute per-question attribution F1 between retrieved and gold evidence.

    Uses token-level overlap: a retrieved passage is "matched" if its token-F1
    with any gold evidence passage exceeds the threshold.
    """
    if not gold_evidence or not retrieved_texts:
        return 0.0

    # Tokenize gold evidence passages
    gold_token_sets = [
        set(tokenize_for_f1(g, remove_stopwords=True)) for g in gold_evidence
    ]
    # Remove empty sets
    gold_token_sets = [g for g in gold_token_sets if g]
    if not gold_token_sets:
        return 0.0

    # For each retrieved passage, check if it matches any gold passage
    matched_retrieved = 0
    matched_gold = set()

    for ret_text in retrieved_texts:
        ret_tokens = set(tokenize_for_f1(ret_text, remove_stopwords=True))
        if not ret_tokens:
            continue

        for g_idx, gold_tokens in enumerate(gold_token_sets):
            overlap = ret_tokens & gold_tokens
            if not overlap:
                continue
            # Token-F1 between this retrieved passage and this gold passage
            p = len(overlap) / len(ret_tokens)
            r = len(overlap) / len(gold_tokens)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f >= threshold:
                matched_retrieved += 1
                matched_gold.add(g_idx)
                break  # Count each retrieved passage only once

    # Attribution precision: fraction of retrieved that match gold
    precision = matched_retrieved / len(retrieved_texts) if retrieved_texts else 0
    # Attribution recall: fraction of gold that are matched
    recall = len(matched_gold) / len(gold_token_sets) if gold_token_sets else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_all(config: dict) -> Path:
    """Orchestrate Stage E: load answers + gold, compute both scores, save.

    Returns path to saved output.
    """
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    output_path = data_dir / "answer_scores.jsonl"

    if output_path.exists():
        existing = load_jsonl(output_path)
        logger.info(
            f"Answer scores already exist at {output_path} ({len(existing)} records), skipping."
        )
        return output_path

    # Load selected questions (for gold answers)
    questions_path = data_dir / "selected_questions.jsonl"
    questions = load_jsonl(questions_path)
    gold_map = {
        q["question_id"]: q for q in questions
    }
    logger.info(f"Loaded {len(questions)} questions with gold answers")

    # Load CoT traces for final answers
    conditions = config["retrieval"]["conditions"]
    threshold = config["scoring"]["matching_threshold"]

    results = []
    for condition in conditions:
        traces_path = data_dir / "cot_traces" / f"{condition}.jsonl"
        if not traces_path.exists():
            logger.warning(f"Traces not found: {traces_path}")
            continue

        traces = load_jsonl(traces_path)
        logger.info(f"[{condition}] Scoring {len(traces)} traces")

        for trace in traces:
            qid = trace["question_id"]
            gold = gold_map.get(qid)
            if not gold:
                logger.warning(f"No gold answer for {qid}")
                continue

            # Token-F1 for the final answer
            answer_f1 = score_answer_token_f1(
                predicted=trace.get("final_answer", ""),
                gold=gold["gold_answer_text"],
                answer_type=gold["answer_type"],
                gold_yes_no=gold.get("gold_yes_no"),
            )

            # Attribution F1 for retrieved evidence
            attr_f1 = compute_attribution_f1(
                retrieved_texts=trace.get("retrieved_texts", []),
                gold_evidence=gold.get("gold_evidence", []),
                threshold=threshold,
            )

            results.append({
                "question_id": qid,
                "condition": condition,
                "answer_token_f1": round(answer_f1, 6),
                "attribution_f1": round(attr_f1, 6),
                "answer_type": gold["answer_type"],
            })

    save_jsonl(results, output_path)
    logger.info(f"Saved {len(results)} answer scores to {output_path}")

    # Summary statistics
    if results:
        import pandas as pd

        df = pd.DataFrame(results)
        logger.info(f"\nAnswer scores summary:\n{df.describe().to_string()}")

        for cond in conditions:
            cond_df = df[df["condition"] == cond]
            if len(cond_df) > 0:
                logger.info(
                    f"[{cond}] mean_token_f1={cond_df['answer_token_f1'].mean():.4f}, "
                    f"mean_attr_f1={cond_df['attribution_f1'].mean():.4f}"
                )

    return output_path
