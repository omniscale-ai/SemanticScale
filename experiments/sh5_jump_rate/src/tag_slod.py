"""Stage C: Retrain SH1 probe, embed CoT steps with SciBERT, classify SLoD levels."""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils import SLOD_MAP, load_jsonl, resolve_path, save_jsonl

logger = logging.getLogger(__name__)


def train_probe(config: dict) -> tuple[StandardScaler, LogisticRegression, dict]:
    """Retrain SciBERT+LogReg probe from SH1 saved embeddings.

    Returns (scaler, classifier, metrics_dict).
    """
    import json

    # Load embeddings
    emb_path = resolve_path(
        config,
        config["paths"]["sh1_data_dir"],
        config["paths"]["sh1_embeddings"],
    )
    logger.info(f"Loading SH1 embeddings from {emb_path}")
    data = np.load(emb_path)
    embeddings = data["embeddings"]
    labels = data["labels"]
    logger.info(f"Embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")

    # Load splits
    splits_path = resolve_path(
        config, config["paths"]["sh1_data_dir"], config["paths"]["sh1_splits"]
    )
    with open(splits_path) as f:
        splits = json.load(f)

    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])
    test_idx = np.array(splits["test"])

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_val, y_val = embeddings[val_idx], labels[val_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    logger.info(
        f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train LogReg
    probe_cfg = config["probe"]
    clf = LogisticRegression(
        C=probe_cfg["logreg_C"],
        max_iter=probe_cfg["logreg_max_iter"],
        solver=probe_cfg["logreg_solver"],
    )
    logger.info(f"Training LogReg (C={probe_cfg['logreg_C']})...")
    clf.fit(X_train_s, y_train)

    # Evaluate on test
    y_pred = clf.predict(X_test_s)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    logger.info(f"Probe test macro-F1: {macro_f1:.4f}")
    logger.info(
        "\n" + classification_report(
            y_test, y_pred, target_names=["macro", "meso", "micro"]
        )
    )

    metrics = {
        "test_macro_f1": float(macro_f1),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }

    return scaler, clf, metrics


def load_scibert(config: dict):
    """Load SciBERT model and tokenizer.

    Returns (model, tokenizer).
    """
    from transformers import AutoModel, AutoTokenizer

    model_name = config["probe"]["scibert_model"]
    logger.info(f"Loading SciBERT: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def embed_steps(
    steps: list[str],
    model,
    tokenizer,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Embed text steps with SciBERT [CLS] token.

    Returns np.ndarray of shape (len(steps), 768).
    """
    if not steps:
        return np.zeros((0, 768), dtype=np.float32)

    # Sort by length for efficient batching
    sort_idx = np.argsort([len(t.split()) for t in steps])
    sorted_steps = [steps[i] for i in sort_idx]

    all_embeddings = []
    for i in tqdm(
        range(0, len(sorted_steps), batch_size),
        desc="Embedding CoT steps",
        disable=len(sorted_steps) < 100,
    ):
        batch = sorted_steps[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls)

    sorted_emb = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Restore original order
    unsort_idx = np.argsort(sort_idx)
    return sorted_emb[unsort_idx]


def tag_all_steps(config: dict) -> Path:
    """Orchestrate Stage C: load CoT traces, embed steps, classify SLoD levels.

    Returns path to saved output.
    """
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    output_path = data_dir / "cot_slod_tags.jsonl"

    if output_path.exists():
        existing = load_jsonl(output_path)
        logger.info(
            f"SLoD tags already exist at {output_path} ({len(existing)} records), skipping."
        )
        return output_path

    # Train probe
    scaler, clf, probe_metrics = train_probe(config)
    logger.info(f"Probe metrics: {probe_metrics}")

    # Load SciBERT
    model, tokenizer = load_scibert(config)
    batch_size = config["probe"]["scibert_batch_size"]
    max_length = config["probe"]["scibert_max_length"]

    # Collect all steps from all conditions
    conditions = config["retrieval"]["conditions"]
    all_step_records = []  # (question_id, condition, step_index, step_text)

    for condition in conditions:
        traces_path = data_dir / "cot_traces" / f"{condition}.jsonl"
        if not traces_path.exists():
            logger.warning(f"Traces file not found: {traces_path}")
            continue

        traces = load_jsonl(traces_path)
        for trace in traces:
            for step_idx, step_text in enumerate(trace["cot_steps"]):
                all_step_records.append({
                    "question_id": trace["question_id"],
                    "condition": condition,
                    "step_index": step_idx,
                    "step_text": step_text,
                })

    logger.info(f"Total CoT steps to embed and classify: {len(all_step_records)}")

    if not all_step_records:
        logger.warning("No steps found to tag!")
        save_jsonl([], output_path)
        return output_path

    # Embed all steps
    step_texts = [r["step_text"] for r in all_step_records]
    logger.info(f"Embedding {len(step_texts)} steps with SciBERT...")
    embeddings = embed_steps(step_texts, model, tokenizer, batch_size, max_length)

    # Scale and classify
    embeddings_scaled = scaler.transform(embeddings)
    predictions = clf.predict(embeddings_scaled)
    probabilities = clf.predict_proba(embeddings_scaled)

    # Build output records
    output_records = []
    for i, record in enumerate(all_step_records):
        pred_class = int(predictions[i])
        probs = probabilities[i].tolist()
        output_records.append({
            "question_id": record["question_id"],
            "condition": record["condition"],
            "step_index": record["step_index"],
            "step_text": record["step_text"],
            "predicted_slod": pred_class,
            "predicted_label": SLOD_MAP[pred_class],
            "probabilities": {
                "macro": round(probs[0], 4),
                "meso": round(probs[1], 4),
                "micro": round(probs[2], 4),
            },
        })

    save_jsonl(output_records, output_path)
    logger.info(f"Saved {len(output_records)} SLoD tags to {output_path}")

    # Report distribution
    from collections import Counter
    label_dist = Counter(r["predicted_label"] for r in output_records)
    total = len(output_records)
    logger.info("SLoD label distribution:")
    for label in ["macro", "meso", "micro"]:
        count = label_dist.get(label, 0)
        pct = 100 * count / total if total else 0
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    # Check for degenerate distribution
    for label in ["macro", "meso", "micro"]:
        pct = 100 * label_dist.get(label, 0) / total if total else 0
        if pct < 10:
            logger.warning(
                f"WARNING: {label} class is underrepresented ({pct:.1f}%). "
                "Domain shift from SH1 training data may be significant."
            )

    return output_path
