"""HyDE-based query SLoD classifier.

For each question, generate a hypothetical answer using Claude Haiku,
then classify that answer with the SH1 SciBERT+LogReg probe.
This bridges the domain mismatch: the SH1 probe was trained on document spans,
and hypothetical answers resemble document spans better than raw questions do.
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.query_classifier import embed_queries_scibert, train_slod_probe
from src.utils import LABEL_NAMES, load_config, load_json, load_jsonl, save_json


def generate_hypothetical_answers(
    questions: list[dict],
    config: dict,
    project_root: Path,
) -> dict:
    """Generate hypothetical answers for questions using Claude Haiku.

    Uses checkpoint/resume: if data/hyde_answers.json already has some answers,
    skips those and continues from where we left off.

    Args:
        questions: List of question dicts with question_id, question_text, paper_id.
        config: Full config dict (reads hyde section).
        project_root: Project root directory.

    Returns:
        Dict mapping question_id -> hypothetical answer text.
    """
    import anthropic

    data_dir = project_root / config["paths"]["data_dir"]
    checkpoint_path = data_dir / "hyde_answers.json"

    hyde_cfg = config.get("hyde", {})
    model = hyde_cfg.get("model", "claude-haiku-4-5-20251001")
    max_tokens = hyde_cfg.get("max_tokens", 150)
    temperature = hyde_cfg.get("temperature", 0.3)
    prompt_template = hyde_cfg.get(
        "prompt_template",
        "You are reading a scientific paper. Answer this question in 1-3 sentences "
        "as if you were writing the relevant part of the paper: {question}",
    )
    batch_size = hyde_cfg.get("batch_size", 10)

    # Load existing checkpoint if available
    answers = {}
    if checkpoint_path.exists():
        answers = load_json(checkpoint_path)
        logging.info(f"Loaded {len(answers)} existing hypothetical answers from checkpoint")

    # Filter to questions that still need answers
    remaining = [q for q in questions if q["question_id"] not in answers]
    if not remaining:
        logging.info("All hypothetical answers already generated, nothing to do")
        return answers

    logging.info(f"Generating hypothetical answers for {len(remaining)} questions "
                 f"({len(answers)} already done, {len(questions)} total)")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    errors = 0
    max_retries = 3

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question_text = q["question_text"]
        prompt = prompt_template.format(question=question_text)

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer_text = response.content[0].text
                answers[qid] = answer_text
                break
            except Exception as e:
                wait_time = 2 ** attempt
                logging.warning(
                    f"API error for {qid} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        else:
            # All retries failed
            logging.error(f"Failed to generate answer for {qid} after {max_retries} retries, skipping")
            errors += 1

        # Rate limiting
        time.sleep(0.1)

        # Checkpoint every batch_size questions
        if (i + 1) % batch_size == 0:
            save_json(answers, checkpoint_path)
            logging.info(f"Checkpoint: {len(answers)}/{len(questions)} answers saved "
                         f"({i + 1}/{len(remaining)} this session)")

    # Final save
    save_json(answers, checkpoint_path)
    logging.info(f"Hypothetical answer generation complete: {len(answers)} answers, {errors} errors")

    return answers


def classify_hyde_answers(
    config: dict,
    project_root: Path,
) -> dict:
    """Classify hypothetical answers with the SH1 SciBERT+LogReg probe.

    Steps:
    1. Load hypothetical answers from data/hyde_answers.json
    2. Embed them with SciBERT ([CLS] token)
    3. Classify with the SH1 probe (retrained from SH1 embeddings/splits)
    4. Save predictions to data/query_slod_predictions_hyde.json

    Args:
        config: Full config dict.
        project_root: Project root directory.

    Returns:
        Dict of predictions: question_id -> {predicted_slod, probabilities}.
    """
    data_dir = project_root / config["paths"]["data_dir"]

    hyde_cfg = config.get("hyde", {})
    predictions_file = hyde_cfg.get("predictions_file", "query_slod_predictions_hyde.json")
    out_path = data_dir / predictions_file

    if out_path.exists():
        logging.info(f"HyDE predictions already exist at {out_path}, skipping.")
        return load_json(out_path)

    # Load hypothetical answers
    answers_path = data_dir / "hyde_answers.json"
    if not answers_path.exists():
        raise FileNotFoundError(f"Hypothetical answers not found at {answers_path}. Run generation first.")

    answers = load_json(answers_path)
    logging.info(f"Loaded {len(answers)} hypothetical answers")

    # Build pseudo-question dicts for embedding (reuse embed_queries_scibert which expects question_text)
    answer_items = sorted(answers.items())  # deterministic order
    pseudo_questions = [{"question_text": text} for _, text in answer_items]
    question_ids = [qid for qid, _ in answer_items]

    # Train SH1 probe
    logging.info("Training SH1 probe from SH1 embeddings...")
    scaler, clf = train_slod_probe(config, project_root=project_root)

    # Embed hypothetical answers with SciBERT
    logging.info(f"Embedding {len(pseudo_questions)} hypothetical answers with SciBERT...")
    embeddings = embed_queries_scibert(pseudo_questions, config)
    logging.info(f"HyDE embeddings shape: {embeddings.shape}")

    # Classify
    X_scaled = scaler.transform(embeddings)
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)

    # Build results dict
    results = {}
    class_counts = Counter()

    for qid, pred, probs in zip(question_ids, predictions, probabilities):
        pred_label = LABEL_NAMES[int(pred)]
        class_counts[pred_label] += 1
        results[qid] = {
            "predicted_slod": pred_label,
            "probabilities": {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
        }

    # Log distribution
    total = len(results)
    logging.info("HyDE SLoD prediction distribution:")
    for name in LABEL_NAMES:
        count = class_counts.get(name, 0)
        pct = count / total * 100 if total > 0 else 0
        logging.info(f"  {name}: {count} ({pct:.1f}%)")

    # Confidence stats
    all_confs = [max(r["probabilities"].values()) for r in results.values()]
    mean_conf = np.mean(all_confs)
    low_conf = sum(1 for c in all_confs if c < 0.6)
    logging.info(f"Mean confidence: {mean_conf:.3f}")
    logging.info(f"Low-confidence (<0.6): {low_conf}/{total} ({low_conf / total * 100:.1f}%)")

    save_json(results, out_path)
    logging.info(f"HyDE predictions saved to {out_path}")

    return results


def compare_predictions(
    hyde_preds: dict,
    config: dict,
    project_root: Path,
) -> None:
    """Compare HyDE predictions vs v1 and v2 predictions."""
    data_dir = project_root / config["paths"]["data_dir"]

    comparisons = [
        ("v1", data_dir / "query_slod_predictions.json"),
        ("v2", data_dir / "query_slod_predictions_v2.json"),
    ]

    for name, path in comparisons:
        if not path.exists():
            logging.info(f"No {name} predictions found at {path}, skipping comparison")
            continue

        other_preds = load_json(path)

        # Find common question IDs
        common_ids = set(hyde_preds.keys()) & set(other_preds.keys())
        if not common_ids:
            logging.info(f"No overlapping questions between HyDE and {name}")
            continue

        agree = 0
        disagree = 0
        changes = Counter()
        hyde_confs = []
        other_confs = []

        for qid in common_ids:
            hyde_label = hyde_preds[qid]["predicted_slod"]
            other_label = other_preds[qid]["predicted_slod"]
            hyde_conf = max(hyde_preds[qid]["probabilities"].values())
            other_conf = max(other_preds[qid]["probabilities"].values())

            hyde_confs.append(hyde_conf)
            other_confs.append(other_conf)

            if hyde_label == other_label:
                agree += 1
            else:
                disagree += 1
                changes[f"{other_label}->{ hyde_label}"] += 1

        total = agree + disagree
        logging.info(f"=== HyDE vs {name} comparison (n={total}) ===")
        logging.info(f"Agreement: {agree}/{total} ({agree / total * 100:.1f}%)")
        logging.info(f"Disagreement: {disagree}/{total} ({disagree / total * 100:.1f}%)")
        logging.info(f"Change breakdown: {dict(changes.most_common())}")
        logging.info(f"Mean confidence — HyDE: {np.mean(hyde_confs):.3f}, {name}: {np.mean(other_confs):.3f}")

        # Distribution comparison
        hyde_dist = Counter(hyde_preds[qid]["predicted_slod"] for qid in common_ids)
        other_dist = Counter(other_preds[qid]["predicted_slod"] for qid in common_ids)
        logging.info(f"Distribution — HyDE: {dict(hyde_dist)}, {name}: {dict(other_dist)}")


def main(config: dict, project_root: Path | None = None) -> None:
    """Run HyDE classification pipeline: generate answers, embed, classify, compare."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]

    # Load questions
    questions = load_jsonl(data_dir / "questions.jsonl")
    logging.info(f"Loaded {len(questions)} questions")

    # Step 1: Generate hypothetical answers
    logging.info("=" * 70)
    logging.info("STEP 1: Generate hypothetical answers with Claude Haiku")
    logging.info("=" * 70)
    answers = generate_hypothetical_answers(questions, config, project_root)

    # Check coverage
    missing = [q["question_id"] for q in questions if q["question_id"] not in answers]
    if missing:
        logging.warning(f"{len(missing)} questions have no hypothetical answer")

    # Step 2: Classify hypothetical answers with SH1 probe
    logging.info("=" * 70)
    logging.info("STEP 2: Classify hypothetical answers with SH1 SciBERT+LogReg probe")
    logging.info("=" * 70)
    hyde_preds = classify_hyde_answers(config, project_root)

    # Step 3: Compare with previous predictions
    logging.info("=" * 70)
    logging.info("STEP 3: Compare HyDE predictions with v1 and v2")
    logging.info("=" * 70)
    compare_predictions(hyde_preds, config, project_root)
