#!/usr/bin/env python3
"""Rerun SH4 Stages D-F with LLM-derived SLoD probe.

Monkey-patches the SLoD probe to train on LLM labels instead of SH0 labels,
then re-runs feature engineering, model training, and analysis.

Usage:
    cd playground/SLoD-SH4
    python scripts/rerun_sh4_llm.py
"""
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, setup_logging, ensure_dirs, read_json

SH1_ROOT = project_root.parent / "SLoD-SH1"

LABEL_MAP = {"macro": 0, "meso": 1, "micro": 2}


def load_llm_labels():
    """Load LLM labels from SH1 relabeling results."""
    relabel_dir = SH1_ROOT / "data" / "relabel"
    labels = {}
    for p in sorted(relabel_dir.glob("result_*.json")):
        with open(p) as f:
            batch = json.load(f)
        for item in batch:
            if isinstance(item, dict) and "id" in item and "label" in item:
                label = item["label"].lower().strip()
                if label in LABEL_MAP:
                    labels[int(item["id"])] = LABEL_MAP[label]
    return labels


def patch_slod_probe():
    """Monkey-patch SLoDProbe._train_probe to use LLM labels."""
    from src.feature_engineering import SLoDProbe

    original_train = SLoDProbe._train_probe

    def _train_probe_llm(self):
        logging.info("Training SLoD probe from LLM labels (patched)...")

        emb_path = self.sh1_data_dir / "embeddings" / "scibert_length_matched.npz"
        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"]

        # Load LLM labels
        llm_labels = load_llm_labels()
        logging.info(f"LLM labels: {len(llm_labels)} spans")

        # Load metadata for train/val split
        meta_path = SH1_ROOT / "data" / "relabel" / "samples_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        train_idx, train_lab = [], []
        val_idx, val_lab = [], []
        for sample in metadata:
            idx = sample["id"]
            if idx in llm_labels:
                if sample["split"] == "train":
                    train_idx.append(idx)
                    train_lab.append(llm_labels[idx])
                elif sample["split"] == "val":
                    val_idx.append(idx)
                    val_lab.append(llm_labels[idx])

        train_idx = np.array(train_idx)
        train_lab = np.array(train_lab)
        val_idx = np.array(val_idx)
        val_lab = np.array(val_lab)

        X_train = embeddings[train_idx]
        X_val = embeddings[val_idx]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # Sweep C
        best_c, best_f1 = 1.0, -1
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
            clf.fit(X_train_s, train_lab)
            vf1 = f1_score(val_lab, clf.predict(X_val_s), average="macro")
            if vf1 > best_f1:
                best_f1 = vf1
                best_c = C

        self.clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=42)
        self.clf.fit(X_train_s, train_lab)

        val_acc = self.clf.score(X_val_s, val_lab)
        logging.info(f"LLM SLoD probe: C={best_c}, val acc={val_acc:.4f}, val F1={best_f1:.4f}")

    SLoDProbe._train_probe = _train_probe_llm


def main():
    setup_logging()
    config = load_config()
    ensure_dirs(config, project_root)
    data_dir = project_root / config["data_dir"]

    print("\n" + "=" * 60)
    print("SH4 RERUN WITH LLM-DERIVED SLoD PROBE")
    print("=" * 60)

    # Patch the probe
    patch_slod_probe()

    # Back up and remove cached features/models/results to force recomputation
    feat_cfg = config["features"]
    model_cfg = config["model"]
    feature_path = data_dir / feat_cfg["output"]
    metrics_path = data_dir / model_cfg["output"]["metrics"]
    model_path = data_dir / model_cfg["output"]["model"]
    results_dir = data_dir / "results"

    backup_suffix = ".orig_sh0"
    for p in [feature_path,
              data_dir / "features" / "feature_info.json",
              metrics_path,
              model_path,
              results_dir / "predictions.json"]:
        if p.exists():
            backup = p.with_suffix(p.suffix + backup_suffix)
            if not backup.exists():
                shutil.copy2(p, backup)
                logging.info(f"Backed up {p.name}")
            p.unlink()

    # Stage D: Feature engineering with LLM probe
    print("\n--- Stage D: Feature engineering (LLM probe) ---")
    from src.feature_engineering import run_feature_engineering
    run_feature_engineering(config, project_root)

    # Stage E: Model training
    print("\n--- Stage E: Model training ---")
    from src.model_training import run_training
    run_training(config, project_root)

    # Load results and compare
    print("\n" + "=" * 60)
    print("COMPARISON: SH0 vs LLM probe")
    print("=" * 60)

    if metrics_path.exists():
        with open(metrics_path) as f:
            llm_metrics = json.load(f)

        print(f"\n  LLM probe results:")
        for model_name, metrics in llm_metrics.items():
            if isinstance(metrics, dict) and "auroc" in metrics:
                print(f"    {model_name}: AUROC={metrics['auroc']:.4f}")

    orig_path = metrics_path.with_suffix(metrics_path.suffix + backup_suffix)
    if orig_path.exists():
        with open(orig_path) as f:
            orig_metrics = json.load(f)
        print(f"\n  Original (SH0) results:")
        for model_name, metrics in orig_metrics.items():
            if isinstance(metrics, dict) and "auroc" in metrics:
                print(f"    {model_name}: AUROC={metrics['auroc']:.4f}")

    # Copy LLM results with suffix, restore originals
    for p in [feature_path,
              data_dir / "features" / "feature_info.json",
              metrics_path,
              model_path,
              results_dir / "predictions.json"]:
        if p.exists():
            llm_copy = p.with_suffix(p.suffix + ".llm")
            if llm_copy.exists():
                llm_copy.unlink()
            shutil.copy2(p, llm_copy)

        backup = p.with_suffix(p.suffix + backup_suffix)
        if backup.exists():
            if p.exists():
                p.unlink()
            shutil.copy2(backup, p)
            backup.unlink()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
