#!/usr/bin/env python3
"""Rerun SH3 Steps 3-6 with LLM-derived SLoD probe.

Retrains the query classifier on LLM labels instead of SH0 section-derived
labels, then re-runs retrieval + evaluation + analysis.

Reuses cached index (Step 1) and embeddings (Step 2).

Usage:
    cd playground/SLoD-SH3
    python scripts/rerun_sh3_llm.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import LABEL_MAP, LABEL_NAMES, load_config, load_jsonl, save_json, setup_logging
from src.query_classifier import embed_queries_scibert
from src.retrieve import run_all_retrieval
from src.evaluate import evaluate_all

SH1_ROOT = Path(__file__).resolve().parent.parent.parent / "SLoD-SH1"


def train_llm_probe(config, project_root):
    """Train LogReg probe on LLM-labeled SH1 spans."""
    logging.info("Training SLoD probe on LLM labels...")

    sh1_dir = project_root / config["paths"]["sh1_data_dir"]
    emb_path = sh1_dir / config["paths"]["sh1_embeddings"]

    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]

    # Load LLM labels
    relabel_dir = SH1_ROOT / "data" / "relabel"
    llm_labels = {}
    for p in sorted(relabel_dir.glob("result_*.json")):
        with open(p) as f:
            batch = json.load(f)
        for item in batch:
            if isinstance(item, dict) and "id" in item and "label" in item:
                label = item["label"].lower().strip()
                if label in LABEL_MAP:
                    llm_labels[int(item["id"])] = LABEL_MAP[label]

    logging.info(f"LLM labels loaded: {len(llm_labels)} spans")

    # Load metadata to separate train/val
    meta_path = relabel_dir / "samples_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    train_indices, train_labels = [], []
    val_indices, val_labels = [], []
    for sample in metadata:
        idx = sample["id"]
        if idx in llm_labels:
            if sample["split"] == "train":
                train_indices.append(idx)
                train_labels.append(llm_labels[idx])
            elif sample["split"] == "val":
                val_indices.append(idx)
                val_labels.append(llm_labels[idx])

    train_indices = np.array(train_indices)
    train_labels = np.array(train_labels)
    val_indices = np.array(val_indices)
    val_labels = np.array(val_labels)

    X_train = embeddings[train_indices]
    X_val = embeddings[val_indices]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Hyperparameter sweep
    best_c, best_f1 = 1.0, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_train_s, train_labels)
        vf1 = f1_score(val_labels, clf.predict(X_val_s), average="macro")
        if vf1 > best_f1:
            best_f1 = vf1
            best_c = C

    clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train_s, train_labels)

    logging.info(f"LLM probe: best C={best_c}, val F1={best_f1:.4f}, train n={len(train_indices)}")
    return scaler, clf


def classify_queries_llm(config, project_root, scaler, clf):
    """Classify queries with LLM-trained probe, save predictions."""
    data_dir = project_root / config["paths"]["data_dir"]
    out_path = data_dir / "query_slod_predictions_llm.json"

    questions = load_jsonl(data_dir / "questions.jsonl")
    logging.info(f"Classifying {len(questions)} queries with LLM probe...")

    query_embeddings = embed_queries_scibert(questions, config)
    X_scaled = scaler.transform(query_embeddings)
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)

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

    logging.info("LLM probe SLoD distribution:")
    for name, count in class_counts.items():
        pct = count / len(questions) * 100
        logging.info(f"  {name}: {count} ({pct:.1f}%)")

    save_json(results, out_path)
    logging.info(f"LLM predictions saved to {out_path}")

    # Compare with original predictions
    orig_path = data_dir / "query_slod_predictions.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        agree = sum(
            1 for qid in results
            if qid in orig and results[qid]["predicted_slod"] == orig[qid]["predicted_slod"]
        )
        total = len(results)
        logging.info(f"Agreement with original predictions: {agree}/{total} ({agree/total:.1%})")

    return out_path


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    data_dir = Path(config["paths"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    results_dir = data_dir / "results"

    print("\n" + "=" * 60)
    print("SH3 RERUN WITH LLM-DERIVED SLoD PROBE")
    print("=" * 60)

    # Step 3: Retrain probe + classify queries
    print("\n--- Step 3: Query classification (LLM probe) ---")
    scaler, clf = train_llm_probe(config, project_root)
    pred_path = classify_queries_llm(config, project_root, scaler, clf)

    # Swap predictions file and move existing results aside
    config["retrieval"]["active_predictions_file"] = "query_slod_predictions_llm.json"

    retrieval_path = results_dir / "retrieval_results.json"
    metrics_path = results_dir / "evaluation_metrics.json"
    bootstrap_path = results_dir / "bootstrap_tests.json"

    # Back up original results
    backup_suffix = ".orig_sh0"
    for p in [retrieval_path, metrics_path, bootstrap_path]:
        backup = p.with_suffix(p.suffix + backup_suffix)
        if p.exists() and not backup.exists():
            p.rename(backup)
            logging.info(f"Backed up {p.name} → {backup.name}")
        elif p.exists():
            p.unlink()
            logging.info(f"Removed {p.name} (backup already exists)")

    # Step 4: Retrieval
    print("\n--- Step 4: Retrieval (13 conditions) ---")
    from src.retrieve import run_all_retrieval
    run_all_retrieval(config, project_root=project_root)

    # Step 5: Evaluation
    print("\n--- Step 5: Evaluation ---")
    evaluate_all(config, project_root=project_root)

    # Step 6: Compare
    print("\n" + "=" * 60)
    print("COMPARISON: SH0 vs LLM probe")
    print("=" * 60)

    orig_metrics_path = metrics_path.with_suffix(metrics_path.suffix + backup_suffix)
    if orig_metrics_path.exists() and metrics_path.exists():
        with open(orig_metrics_path) as f:
            orig_metrics = json.load(f)
        with open(metrics_path) as f:
            llm_metrics = json.load(f)

        # Use primary threshold (0.5) results
        print(f"\n{'Condition':<40} {'SH0 soft-F1':>12} {'LLM soft-F1':>12} {'Delta':>8}")
        print("-" * 75)

        for cond in config["retrieval"]["conditions"]:
            k = "5"
            # Navigate the metrics structure
            orig_cond = orig_metrics.get(cond, {})
            llm_cond = llm_metrics.get(cond, {})

            # Try different key formats
            for key in [f"k={k}", k, f"k{k}"]:
                if key in orig_cond:
                    orig_k = orig_cond[key]
                    llm_k = llm_cond.get(key, {})
                    break
            else:
                orig_k = {}
                llm_k = {}

            orig_f1 = orig_k.get("soft_attribution_f1", orig_k.get("soft_f1", 0))
            llm_f1 = llm_k.get("soft_attribution_f1", llm_k.get("soft_f1", 0))
            delta = llm_f1 - orig_f1
            marker = " *" if abs(delta) > 0.01 else ""
            print(f"  {cond:<38} {orig_f1:>12.4f} {llm_f1:>12.4f} {delta:>+8.4f}{marker}")

        # Rename LLM results with suffix
        for p in [retrieval_path, metrics_path, bootstrap_path]:
            if p.exists():
                llm_copy = p.with_suffix(p.suffix + ".llm")
                if llm_copy.exists():
                    llm_copy.unlink()
                import shutil
                shutil.copy2(p, llm_copy)

        # Restore originals
        for p in [retrieval_path, metrics_path, bootstrap_path]:
            backup = p.with_suffix(p.suffix + backup_suffix)
            if backup.exists():
                if p.exists():
                    p.unlink()
                backup.rename(p)
    else:
        print("  Cannot compare: original metrics not found.")

    # Generate report
    report_path = project_root / "reports" / "SH3_LLM_RERUN_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# SH3 Rerun with LLM-Derived SLoD Probe\n\n"
        "> Re-ran retrieval pipeline with probe trained on LLM labels.\n\n"
        f"See comparison output above. Results saved to:\n"
        f"- `{results_dir}/retrieval_results.json.llm`\n"
        f"- `{results_dir}/evaluation_metrics.json.llm`\n"
    )
    print(f"\nReport: {report_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
