"""Question-specific SLoD probe for SH3 iteration 4.

Train a LogReg probe on QASPER validation questions with SLoD labels derived
from gold evidence matched to SH0-labeled spans. This replaces the domain-
mismatched SH1 document-span probe applied to questions.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import LABEL_MAP, LABEL_NAMES, load_config, load_json, load_jsonl, normalize_text, save_json


def _compute_token_f1(text_a: str, text_b: str) -> float:
    """Token-level F1 between two texts (multiset overlap)."""
    tokens_a = normalize_text(text_a).split()
    tokens_b = normalize_text(text_b).split()
    if not tokens_a or not tokens_b:
        return 0.0
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    overlap = sum(min(counts_a[t], counts_b[t]) for t in counts_a if t in counts_b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def _load_sh0_spans_by_paper(sh0_path: Path) -> dict:
    """Load SH0 spans indexed by paper_id.

    Returns:
        Dict: paper_id -> list of span dicts.
    """
    spans_by_paper = {}
    with open(sh0_path) as f:
        for line in f:
            span = json.loads(line)
            pid = span["paper_id"]
            if pid not in spans_by_paper:
                spans_by_paper[pid] = []
            spans_by_paper[pid].append(span)
    logging.info(f"Loaded SH0 spans: {sum(len(v) for v in spans_by_paper.values())} spans across {len(spans_by_paper)} papers")
    return spans_by_paper


def derive_question_labels(config: dict, project_root: Path) -> list[dict]:
    """Derive SLoD labels for QASPER validation questions from gold evidence.

    For each validation question:
    1. Get gold evidence paragraphs
    2. Match each to SH0 spans (same paper) using token-F1 > 0.5
    3. Take majority SLoD label across matched spans
    4. Skip if no spans match

    Returns:
        List of dicts with keys: question_id, question_text, paper_id, slod_label, label_index,
        n_evidence, n_matched, matched_labels.
    """
    data_dir = project_root / config["paths"]["data_dir"]
    sh0_dir = project_root / config["paths"]["sh0_data_dir"]
    sh0_spans_path = sh0_dir / "qasper_slod_spans.jsonl"

    # Load questions (validation only)
    questions = load_jsonl(data_dir / "questions.jsonl")
    val_questions = [q for q in questions if q["split"] == "validation"]
    logging.info(f"Validation questions: {len(val_questions)}")

    # Load SH0 spans
    spans_by_paper = _load_sh0_spans_by_paper(sh0_spans_path)

    # Match gold evidence to SH0 spans
    labeled_questions = []
    match_stats = {"total": 0, "matched": 0, "no_match": 0, "no_spans_for_paper": 0}

    for q in val_questions:
        match_stats["total"] += 1
        paper_id = q["paper_id"]
        gold_evidence = q.get("gold_evidence", [])

        if not gold_evidence:
            match_stats["no_match"] += 1
            continue

        paper_spans = spans_by_paper.get(paper_id, [])
        if not paper_spans:
            match_stats["no_spans_for_paper"] += 1
            continue

        # For each gold evidence paragraph, find best matching SH0 span
        matched_labels = []
        for evidence_text in gold_evidence:
            best_f1 = 0.0
            best_label = None
            for span in paper_spans:
                f1 = _compute_token_f1(evidence_text, span["text"])
                if f1 > best_f1:
                    best_f1 = f1
                    best_label = span["label"]
            if best_f1 > 0.5 and best_label is not None:
                matched_labels.append(best_label)

        if not matched_labels:
            match_stats["no_match"] += 1
            continue

        match_stats["matched"] += 1

        # Majority label
        label_counts = Counter(matched_labels)
        majority_label = label_counts.most_common(1)[0][0]

        labeled_questions.append({
            "question_id": q["question_id"],
            "question_text": q["question_text"],
            "paper_id": paper_id,
            "slod_label": majority_label,
            "label_index": LABEL_MAP[majority_label],
            "n_evidence": len(gold_evidence),
            "n_matched": len(matched_labels),
            "matched_labels": dict(label_counts),
        })

    logging.info(f"Gold evidence matching: {match_stats}")
    logging.info(f"Labeled questions: {len(labeled_questions)}")

    # Log label distribution
    label_dist = Counter(q["slod_label"] for q in labeled_questions)
    for label in LABEL_NAMES:
        count = label_dist.get(label, 0)
        pct = count / len(labeled_questions) * 100 if labeled_questions else 0
        logging.info(f"  {label}: {count} ({pct:.1f}%)")

    return labeled_questions


def _embed_questions_scibert(texts: list[str], config: dict) -> np.ndarray:
    """Embed question texts with SciBERT [CLS] pooling. Reuses logic from query_classifier.py."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    model_name = config["query_classifier"]["encoder_model"]
    batch_size = config["query_classifier"]["encoder_batch_size"]
    max_length = config["query_classifier"]["encoder_max_length"]

    logging.info(f"Loading SciBERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Sort by length to minimize padding
    sort_idx = np.argsort([len(t.split()) for t in texts])
    sorted_texts = [texts[i] for i in sort_idx]

    all_embeddings = []
    for i in tqdm(range(0, len(sorted_texts), batch_size), desc=f"SciBERT embedding ({len(texts)} questions)"):
        batch_texts = sorted_texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    sorted_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    unsort_idx = np.argsort(sort_idx)
    return sorted_embeddings[unsort_idx]


def train_question_probe(config: dict, project_root: Path) -> dict:
    """Train and validate a question-specific SLoD probe on QASPER validation questions.

    Steps:
    1. Derive labels from gold evidence
    2. Split 80/20 (probe-train / probe-val)
    3. Embed with SciBERT
    4. Train LogReg with StandardScaler
    5. Evaluate on probe-val

    Returns:
        Dict with metrics and the labeled questions.
    """
    v2_cfg = config.get("query_classifier_v2", {})
    train_ratio = v2_cfg.get("probe_train_ratio", 0.8)
    seed = v2_cfg.get("random_seed", 42)

    # Step 1: Derive labels
    labeled_questions = derive_question_labels(config, project_root)
    if len(labeled_questions) < 10:
        raise RuntimeError(f"Too few labeled questions ({len(labeled_questions)}), cannot train probe")

    # Step 2: Split
    texts = [q["question_text"] for q in labeled_questions]
    labels = np.array([q["label_index"] for q in labeled_questions])

    train_idx, val_idx = train_test_split(
        np.arange(len(labeled_questions)),
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    logging.info(f"Probe split: {len(train_idx)} train, {len(val_idx)} val")

    # Step 3: Embed all questions with SciBERT
    embeddings = _embed_questions_scibert(texts, config)
    logging.info(f"SciBERT question embeddings shape: {embeddings.shape}")

    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_val = embeddings[val_idx]
    y_val = labels[val_idx]

    # Step 4: Scale and train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    probe_cfg = config["query_classifier"]["probe"]
    clf = LogisticRegression(
        C=probe_cfg["C"],
        max_iter=probe_cfg["max_iter"],
        solver=probe_cfg["solver"],
        random_state=probe_cfg["random_state"],
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)

    # Step 5: Evaluate on probe-val
    y_val_pred = clf.predict(X_val_s)
    y_val_proba = clf.predict_proba(X_val_s)

    macro_f1 = f1_score(y_val, y_val_pred, average="macro")
    report = classification_report(y_val, y_val_pred, target_names=LABEL_NAMES, output_dict=True)
    logging.info(f"Question probe validation: macro-F1={macro_f1:.4f}")
    logging.info(f"Per-class F1: " + ", ".join(f"{name}={report[name]['f1-score']:.3f}" for name in LABEL_NAMES))

    # Confidence distribution on val set
    val_confidences = np.max(y_val_proba, axis=1)
    conf_bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    conf_hist, _ = np.histogram(val_confidences, bins=conf_bins)
    logging.info("Confidence distribution (probe-val):")
    for i in range(len(conf_bins) - 1):
        logging.info(f"  [{conf_bins[i]:.1f}, {conf_bins[i+1]:.1f}): {conf_hist[i]} ({conf_hist[i]/len(val_confidences)*100:.1f}%)")

    # Compare against old SH1 probe on the same val questions
    _compare_with_sh1_probe(config, project_root, labeled_questions, val_idx, embeddings)

    return {
        "labeled_questions": labeled_questions,
        "embeddings": embeddings,
        "scaler": scaler,
        "clf": clf,
        "macro_f1": macro_f1,
        "report": report,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def _compare_with_sh1_probe(
    config: dict,
    project_root: Path,
    labeled_questions: list[dict],
    val_idx: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    """Compare question probe vs old SH1 probe on the same probe-val questions."""
    from src.query_classifier import train_slod_probe

    try:
        sh1_scaler, sh1_clf = train_slod_probe(config, project_root=project_root)
    except Exception as e:
        logging.warning(f"Could not load SH1 probe for comparison: {e}")
        return

    X_val = embeddings[val_idx]
    y_val = np.array([labeled_questions[i]["label_index"] for i in val_idx])

    X_val_sh1 = sh1_scaler.transform(X_val)
    y_pred_sh1 = sh1_clf.predict(X_val_sh1)
    sh1_proba = sh1_clf.predict_proba(X_val_sh1)

    sh1_f1 = f1_score(y_val, y_pred_sh1, average="macro")
    sh1_confidences = np.max(sh1_proba, axis=1)
    low_conf_sh1 = (sh1_confidences < 0.6).sum()

    logging.info(f"=== SH1 probe on same probe-val questions ===")
    logging.info(f"SH1 probe macro-F1: {sh1_f1:.4f}")
    logging.info(f"SH1 probe low-confidence (<0.6): {low_conf_sh1}/{len(val_idx)} ({low_conf_sh1/len(val_idx)*100:.1f}%)")

    sh1_report = classification_report(y_val, y_pred_sh1, target_names=LABEL_NAMES, output_dict=True)
    logging.info(f"SH1 per-class F1: " + ", ".join(f"{name}={sh1_report[name]['f1-score']:.3f}" for name in LABEL_NAMES))


def classify_test_queries(config: dict, project_root: Path) -> None:
    """Retrain probe on ALL validation questions and classify test questions.

    Saves predictions to data/query_slod_predictions_v2.json.
    """
    v2_cfg = config.get("query_classifier_v2", {})
    predictions_file = v2_cfg.get("predictions_file", "query_slod_predictions_v2.json")

    data_dir = project_root / config["paths"]["data_dir"]
    out_path = data_dir / predictions_file

    if out_path.exists():
        logging.info(f"v2 predictions already exist at {out_path}, skipping.")
        return

    # Derive labels for all validation questions
    labeled_questions = derive_question_labels(config, project_root)

    # Embed all labeled (validation) questions
    val_texts = [q["question_text"] for q in labeled_questions]
    val_labels = np.array([q["label_index"] for q in labeled_questions])
    val_embeddings = _embed_questions_scibert(val_texts, config)

    # Train on full validation set
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(val_embeddings)

    probe_cfg = config["query_classifier"]["probe"]
    clf = LogisticRegression(
        C=probe_cfg["C"],
        max_iter=probe_cfg["max_iter"],
        solver=probe_cfg["solver"],
        random_state=probe_cfg["random_state"],
        class_weight="balanced",
    )
    clf.fit(X_train_s, val_labels)

    label_dist = Counter(val_labels.tolist())
    logging.info(f"Retrained probe on {len(val_labels)} validation questions")
    for idx, name in enumerate(LABEL_NAMES):
        logging.info(f"  {name}: {label_dist.get(idx, 0)}")

    # Load test questions
    questions = load_jsonl(data_dir / "questions.jsonl")
    test_questions = [q for q in questions if q["split"] == "test"]
    logging.info(f"Test questions to classify: {len(test_questions)}")

    # Embed test questions
    test_texts = [q["question_text"] for q in test_questions]
    test_embeddings = _embed_questions_scibert(test_texts, config)

    # Classify
    X_test_s = scaler.transform(test_embeddings)
    predictions = clf.predict(X_test_s)
    probabilities = clf.predict_proba(X_test_s)

    # Build predictions dict
    results = {}
    class_counts = Counter()

    for q, pred, probs in zip(test_questions, predictions, probabilities):
        pred_label = LABEL_NAMES[int(pred)]
        class_counts[pred_label] += 1
        results[q["question_id"]] = {
            "predicted_slod": pred_label,
            "probabilities": {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
        }

    # Log distribution and confidence stats
    logging.info("=== v2 test predictions distribution ===")
    for name in LABEL_NAMES:
        count = class_counts.get(name, 0)
        pct = count / len(test_questions) * 100
        logging.info(f"  {name}: {count} ({pct:.1f}%)")

    # Confidence stats
    all_confs = [max(r["probabilities"].values()) for r in results.values()]
    low_conf = sum(1 for c in all_confs if c < 0.6)
    logging.info(f"Low-confidence predictions (<0.6): {low_conf}/{len(all_confs)} ({low_conf/len(all_confs)*100:.1f}%)")

    # Compare with v1 predictions
    v1_path = data_dir / "query_slod_predictions.json"
    if v1_path.exists():
        v1_preds = load_json(v1_path)
        _compare_v1_v2(v1_preds, results, test_questions)

    save_json(results, out_path)
    logging.info(f"v2 predictions saved to {out_path}")


def _compare_v1_v2(v1_preds: dict, v2_preds: dict, test_questions: list[dict]) -> None:
    """Log comparison between v1 (SH1 probe) and v2 (question probe) predictions."""
    agree = 0
    disagree = 0
    v1_confs = []
    v2_confs = []
    changes = Counter()

    for q in test_questions:
        qid = q["question_id"]
        if qid not in v1_preds or qid not in v2_preds:
            continue
        v1_label = v1_preds[qid]["predicted_slod"]
        v2_label = v2_preds[qid]["predicted_slod"]
        v1_conf = max(v1_preds[qid]["probabilities"].values())
        v2_conf = max(v2_preds[qid]["probabilities"].values())
        v1_confs.append(v1_conf)
        v2_confs.append(v2_conf)

        if v1_label == v2_label:
            agree += 1
        else:
            disagree += 1
            changes[f"{v1_label}->{v2_label}"] += 1

    total = agree + disagree
    logging.info(f"=== v1 vs v2 comparison (n={total}) ===")
    logging.info(f"Agreement: {agree}/{total} ({agree/total*100:.1f}%)")
    logging.info(f"Disagreement: {disagree}/{total} ({disagree/total*100:.1f}%)")
    logging.info(f"Change breakdown: {dict(changes.most_common())}")
    logging.info(f"Mean confidence — v1: {np.mean(v1_confs):.3f}, v2: {np.mean(v2_confs):.3f}")
    logging.info(f"Low-conf (<0.6) — v1: {sum(1 for c in v1_confs if c < 0.6)}, v2: {sum(1 for c in v2_confs if c < 0.6)}")
