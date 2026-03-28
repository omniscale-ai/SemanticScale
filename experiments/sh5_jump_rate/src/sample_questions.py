"""Stage A: Sample 500 test questions and extract gold answers from QASPER."""

import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.utils import load_jsonl, resolve_path, save_jsonl

logger = logging.getLogger(__name__)


def load_sh3_questions(config: dict) -> list[dict]:
    """Load questions.jsonl from SH3 and filter to test split."""
    path = resolve_path(
        config, config["paths"]["sh3_data_dir"], config["paths"]["sh3_questions"]
    )
    logger.info(f"Loading SH3 questions from {path}")
    questions = load_jsonl(path)
    test_qs = [q for q in questions if q["split"] == config["sampling"]["split"]]
    logger.info(f"Loaded {len(test_qs)} test questions (from {len(questions)} total)")
    return test_qs


def stratified_sample(
    questions: list[dict], n: int, seed: int
) -> list[dict]:
    """Stratified sample by answer_type, preserving proportions."""
    rng = np.random.RandomState(seed)

    # Group by answer_type
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        by_type.setdefault(q["answer_type"], []).append(q)

    total = len(questions)
    selected = []

    # Allocate proportionally, handling rounding
    type_counts = {}
    remaining = n
    types_sorted = sorted(by_type.keys())
    for i, atype in enumerate(types_sorted):
        if i == len(types_sorted) - 1:
            # Last type gets whatever is left
            type_counts[atype] = remaining
        else:
            count = round(n * len(by_type[atype]) / total)
            type_counts[atype] = min(count, len(by_type[atype]))
            remaining -= type_counts[atype]

    logger.info(f"Stratified sampling: {type_counts}")

    for atype in types_sorted:
        pool = by_type[atype]
        k = type_counts[atype]
        indices = rng.choice(len(pool), size=k, replace=False)
        selected.extend(pool[i] for i in sorted(indices))

    rng.shuffle(selected)
    return selected


def extract_gold_answers(
    question_ids: list[str], config: dict
) -> dict[str, dict]:
    """Fetch gold answers from the original QASPER HF dataset.

    Maps SH3 question_id (paper_id__qN) to QASPER answers.

    Returns dict mapping question_id -> {gold_answer_text, gold_yes_no}.
    """
    from datasets import load_dataset

    logger.info("Loading QASPER dataset for gold answer extraction")
    ds = load_dataset("allenai/qasper", split="test")

    # Build lookup: paper_id -> {q_index -> answers_list}
    paper_answers = {}
    for item in ds:
        pid = item["id"]
        paper_answers[pid] = item["qas"]["answers"]

    # Parse question_ids and extract answers
    gold = {}
    missing = 0

    for qid in question_ids:
        # Parse "paper_id__qN"
        parts = qid.rsplit("__q", 1)
        if len(parts) != 2:
            logger.warning(f"Cannot parse question_id: {qid}")
            missing += 1
            continue

        paper_id = parts[0]
        q_index = int(parts[1])

        if paper_id not in paper_answers:
            logger.warning(f"Paper {paper_id} not found in QASPER")
            missing += 1
            continue

        answers_for_paper = paper_answers[paper_id]
        if q_index >= len(answers_for_paper):
            logger.warning(f"Question index {q_index} out of range for paper {paper_id}")
            missing += 1
            continue

        # answers_for_paper[q_index] is a dict with key 'answer' containing list of annotator answers
        annotator_answers = answers_for_paper[q_index]["answer"]

        # Extract gold answer text
        gold_answer_text = _extract_answer_text(annotator_answers)
        gold_yes_no = _extract_yes_no(annotator_answers)

        gold[qid] = {
            "gold_answer_text": gold_answer_text,
            "gold_yes_no": gold_yes_no,
        }

    logger.info(
        f"Extracted gold answers for {len(gold)}/{len(question_ids)} questions "
        f"({missing} missing)"
    )
    return gold


def _extract_answer_text(annotator_answers: list[dict]) -> str:
    """Extract best gold answer text from annotator answers.

    For extractive: union of extractive_spans across annotators.
    For abstractive: first non-empty free_form_answer.
    Falls back through both strategies.
    """
    # Collect all extractive spans
    all_spans = []
    for ans in annotator_answers:
        if ans.get("unanswerable"):
            continue
        spans = ans.get("extractive_spans") or []
        all_spans.extend(s.strip() for s in spans if s and s.strip())

    # Collect free-form answers
    free_forms = []
    for ans in annotator_answers:
        if ans.get("unanswerable"):
            continue
        ff = ans.get("free_form_answer", "")
        if ff and ff.strip():
            free_forms.append(ff.strip())

    # Prefer extractive spans if available, otherwise free-form
    if all_spans:
        # Deduplicate while preserving order
        seen = set()
        unique_spans = []
        for s in all_spans:
            if s not in seen:
                seen.add(s)
                unique_spans.append(s)
        return " ".join(unique_spans)
    elif free_forms:
        return free_forms[0]
    else:
        return ""


def _extract_yes_no(annotator_answers: list[dict]) -> bool | None:
    """Extract yes/no answer via majority vote.

    Returns True/False/None.
    """
    votes = []
    for ans in annotator_answers:
        if ans.get("unanswerable"):
            continue
        yn = ans.get("yes_no")
        if yn is not None:
            votes.append(yn)

    if not votes:
        return None

    # Majority vote
    yes_count = sum(1 for v in votes if v is True)
    no_count = sum(1 for v in votes if v is False)

    if yes_count > no_count:
        return True
    elif no_count > yes_count:
        return False
    else:
        # Tie: return first vote
        return votes[0]


def prepare_selected_questions(config: dict) -> Path:
    """Orchestrate Stage A: sample questions and extract gold answers.

    Returns path to saved output.
    """
    output_path = resolve_path(config, config["paths"]["data_dir"], "selected_questions.jsonl")

    if output_path.exists():
        existing = load_jsonl(output_path)
        logger.info(f"Selected questions already exist at {output_path} ({len(existing)} records), skipping.")
        return output_path

    # Load and sample
    test_qs = load_sh3_questions(config)
    n = config["sampling"]["n_questions"]
    seed = config["sampling"]["random_seed"]
    selected = stratified_sample(test_qs, n, seed)

    logger.info(f"Selected {len(selected)} questions")
    type_dist = Counter(q["answer_type"] for q in selected)
    logger.info(f"Answer type distribution: {dict(type_dist)}")

    # Extract gold answers from QASPER
    question_ids = [q["question_id"] for q in selected]
    gold_answers = extract_gold_answers(question_ids, config)

    # Merge
    output = []
    for q in selected:
        qid = q["question_id"]
        gold = gold_answers.get(qid, {"gold_answer_text": "", "gold_yes_no": None})
        output.append({
            "question_id": qid,
            "question_text": q["question_text"],
            "paper_id": q["paper_id"],
            "answer_type": q["answer_type"],
            "gold_evidence": q["gold_evidence"],
            "gold_answer_text": gold["gold_answer_text"],
            "gold_yes_no": gold["gold_yes_no"],
        })

    save_jsonl(output, output_path)
    logger.info(f"Saved {len(output)} selected questions to {output_path}")

    # Report stats
    with_gold = sum(1 for r in output if r["gold_answer_text"])
    logger.info(f"Questions with gold answer text: {with_gold}/{len(output)}")

    return output_path
