"""Query SLoD classifier for SH3: retrain SciBERT+LogReg from SH1 data, classify QASPER queries."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils import LABEL_MAP, LABEL_NAMES, load_config, load_json, load_jsonl, save_json


def train_slod_probe(config: dict, project_root: Path | None = None):
    """Train LogReg probe from SH1 embeddings + splits.

    Returns:
        Tuple of (scaler, classifier) fitted on SH1 training data.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    sh1_dir = project_root / config["paths"]["sh1_data_dir"]
    emb_path = sh1_dir / config["paths"]["sh1_embeddings"]
    splits_path = sh1_dir / config["paths"]["sh1_splits"]

    logging.info(f"Loading SH1 embeddings from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    logging.info(f"Loading SH1 splits from {splits_path}")
    splits = load_json(splits_path)

    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])

    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_val = embeddings[val_idx]
    y_val = labels[val_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Train LogReg with config hyperparameters
    probe_cfg = config["query_classifier"]["probe"]
    clf = LogisticRegression(
        C=probe_cfg["C"],
        max_iter=probe_cfg["max_iter"],
        solver=probe_cfg["solver"],
        random_state=probe_cfg["random_state"],
    )
    clf.fit(X_train_s, y_train)

    # Report validation performance
    from sklearn.metrics import f1_score, accuracy_score

    y_val_pred = clf.predict(X_val_s)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    val_acc = accuracy_score(y_val, y_val_pred)
    logging.info(f"SLoD probe retrained: val macro-F1={val_f1:.4f}, val acc={val_acc:.4f}")

    return scaler, clf


def embed_queries_scibert(
    questions: list[dict],
    config: dict,
) -> np.ndarray:
    """Embed QASPER questions with SciBERT using [CLS] pooling.

    Reuses the pattern from SH1 embed_with_transformers().

    Returns:
        np.ndarray of shape (N_questions, 768).
    """
    from transformers import AutoModel, AutoTokenizer

    model_name = config["query_classifier"]["encoder_model"]
    batch_size = config["query_classifier"]["encoder_batch_size"]
    max_length = config["query_classifier"]["encoder_max_length"]

    logging.info(f"Loading SciBERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    texts = [q["question_text"] for q in questions]

    # Sort by length to minimize padding
    sort_idx = np.argsort([len(t.split()) for t in texts])
    sorted_texts = [texts[i] for i in sort_idx]

    all_embeddings = []
    for i in tqdm(
        range(0, len(sorted_texts), batch_size),
        desc=f"SciBERT embedding ({len(texts)} queries)",
    ):
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
        # [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    sorted_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Restore original order
    unsort_idx = np.argsort(sort_idx)
    return sorted_embeddings[unsort_idx]


def classify_queries(config: dict, project_root: Path | None = None) -> None:
    """Orchestrate: embed QASPER queries with SciBERT, classify with probe, save predictions."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    out_path = data_dir / "query_slod_predictions.json"

    if out_path.exists():
        logging.info("Query SLoD predictions already exist, skipping.")
        return

    # Load questions
    questions = load_jsonl(data_dir / "questions.jsonl")
    logging.info(f"Loaded {len(questions)} questions for SLoD classification")

    # Train probe from SH1 data
    scaler, clf = train_slod_probe(config, project_root=project_root)

    # Embed questions with SciBERT
    logging.info("Embedding queries with SciBERT...")
    query_embeddings = embed_queries_scibert(questions, config)
    logging.info(f"SciBERT query embeddings shape: {query_embeddings.shape}")

    # Classify
    X_scaled = scaler.transform(query_embeddings)
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)

    # Build predictions dict
    results = {}
    class_counts = {name: 0 for name in LABEL_NAMES}

    for q, pred, probs in zip(questions, predictions, probabilities):
        pred_label = LABEL_NAMES[int(pred)]
        class_counts[pred_label] += 1
        results[q["question_id"]] = {
            "predicted_slod": pred_label,
            "probabilities": {
                name: float(p) for name, p in zip(LABEL_NAMES, probs)
            },
        }

    # Log distribution
    logging.info("Predicted SLoD distribution:")
    for name, count in class_counts.items():
        pct = count / len(questions) * 100
        logging.info(f"  {name}: {count} ({pct:.1f}%)")

    save_json(results, out_path)
    logging.info(f"Query SLoD predictions saved to {out_path}")
