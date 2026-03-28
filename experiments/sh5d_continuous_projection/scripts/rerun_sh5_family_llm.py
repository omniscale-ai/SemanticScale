#!/usr/bin/env python3
"""Rerun SH5d + SH5a + SH5c with LLM-derived SLoD probe/axis.

1. Reclassify 6101 CoT steps with LLM-trained probe
2. Recompute SLoD axis from LLM labels
3. Rerun SH5d (axis projections), SH5a (transition matrices), SH5c (alignment)

Usage:
    cd playground/SLoD-SH5d
    python scripts/rerun_sh5_family_llm.py
"""
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SH1_ROOT = PROJECT_ROOT.parent / "SLoD-SH1"
SH5A_ROOT = PROJECT_ROOT.parent / "SLoD-SH5a"
SH5C_ROOT = PROJECT_ROOT.parent / "SLoD-SH5c"

LABEL_MAP = {"macro": 0, "meso": 1, "micro": 2}
LABEL_NAMES = ["macro", "meso", "micro"]

_SLOD_DATA_ROOT = Path(os.environ.get('SLOD_DATA_ROOT', str(Path(__file__).resolve().parent.parent.parent.parent / 'data')))
SHARED_TAGS = _SLOD_DATA_ROOT / "SH5" / "cot_slod_tags.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_llm_labels():
    """Load LLM labels from SH1 relabeling."""
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


def train_llm_probe():
    """Train SciBERT LogReg probe on LLM labels."""
    logging.info("Training LLM probe...")

    emb_path = SH1_ROOT / "data_sh0" / ".." / "data" / "embeddings" / "scibert_length_matched.npz"
    if not emb_path.exists():
        emb_path = _SLOD_DATA_ROOT / "SH1" / "embeddings" / "scibert_length_matched.npz"

    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]

    llm_labels = load_llm_labels()
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

    X_train = embeddings[np.array(train_idx)]
    X_val = embeddings[np.array(val_idx)]
    y_train = np.array(train_lab)
    y_val = np.array(val_lab)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_c, best_f1 = 1.0, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_train_s, y_train)
        vf1 = f1_score(y_val, clf.predict(X_val_s), average="macro")
        if vf1 > best_f1:
            best_f1 = vf1
            best_c = C

    clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train_s, y_train)
    logging.info(f"LLM probe: C={best_c}, val F1={best_f1:.4f}")
    return scaler, clf


def reclassify_cot_steps(scaler, clf):
    """Reclassify all 6101 CoT steps with LLM probe."""
    logging.info("Reclassifying CoT steps...")

    # Load existing tags
    with open(SHARED_TAGS) as f:
        tags = [json.loads(line) for line in f]
    logging.info(f"Loaded {len(tags)} CoT steps")

    # Extract texts and embed with SciBERT
    from transformers import AutoModel, AutoTokenizer

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    texts = [t["step_text"] for t in tags]
    batch_size = 64

    sort_idx = np.argsort([len(t.split()) for t in texts])
    sorted_texts = [texts[i] for i in sort_idx]

    all_embs = []
    for i in tqdm(range(0, len(sorted_texts), batch_size), desc="Embedding CoT steps"):
        batch = sorted_texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls_emb)

    sorted_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    unsort_idx = np.argsort(sort_idx)
    embeddings = sorted_embs[unsort_idx]

    # Classify
    X_scaled = scaler.transform(embeddings)
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)

    # Update tags
    from collections import Counter
    dist = Counter()
    for tag, pred, probs in zip(tags, predictions, probabilities):
        pred_int = int(pred)
        tag["predicted_slod"] = pred_int
        tag["predicted_label"] = LABEL_NAMES[pred_int]
        tag["probabilities"] = {
            name: float(p) for name, p in zip(LABEL_NAMES, probs)
        }
        dist[LABEL_NAMES[pred_int]] += 1

    logging.info(f"LLM probe distribution: {dict(dist)}")

    # Save as new file
    llm_tags_path = SHARED_TAGS.parent / "cot_slod_tags_llm.jsonl"
    with open(llm_tags_path, "w") as f:
        for tag in tags:
            f.write(json.dumps(tag) + "\n")
    logging.info(f"Saved {len(tags)} reclassified tags to {llm_tags_path}")

    return llm_tags_path, embeddings


def recompute_slod_axis():
    """Compute new SLoD axis from LLM labels for SH5d."""
    logging.info("Computing LLM SLoD axis for SH5d...")

    emb_path = _SLOD_DATA_ROOT / "SH1" / "embeddings" / "scibert_length_matched.npz"
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]

    llm_labels = load_llm_labels()
    indices = sorted(llm_labels.keys())
    llm_emb = embeddings[indices]
    llm_lab = np.array([llm_labels[i] for i in indices])

    # Centroid axis
    macro_mask = llm_lab == 0
    micro_mask = llm_lab == 2
    macro_centroid = llm_emb[macro_mask].mean(axis=0)
    micro_centroid = llm_emb[micro_mask].mean(axis=0)
    axis = micro_centroid - macro_centroid
    axis = axis / np.linalg.norm(axis)

    logging.info(f"LLM centroid axis: macro n={macro_mask.sum()}, micro n={micro_mask.sum()}")
    return axis


def backup_and_swap(data_dir, filenames, suffix=".orig_sh0"):
    """Back up files before overwriting."""
    for fname in filenames:
        p = data_dir / fname
        backup = Path(str(p) + suffix)
        if p.exists() and not backup.exists():
            shutil.copy2(p, backup)

def restore_originals(data_dir, filenames, suffix=".orig_sh0"):
    """Save LLM copies and restore originals."""
    for fname in filenames:
        p = data_dir / fname
        if p.exists():
            shutil.copy2(p, Path(str(p) + ".llm"))
        backup = Path(str(p) + suffix)
        if backup.exists():
            if p.exists():
                p.unlink()
            shutil.copy2(backup, p)
            backup.unlink()

def run_sh5d(llm_axis):
    """Rerun SH5d with LLM axis."""
    logging.info("\n" + "=" * 60)
    logging.info("RERUNNING SH5d")
    logging.info("=" * 60)

    data_dir = PROJECT_ROOT / "data"
    files_to_backup = ["slod_axis.npz", "analysis_results.json"]
    backup_and_swap(data_dir, files_to_backup)

    # Save LLM axis
    axis_path = data_dir / "slod_axis.npz"
    np.savez(str(axis_path), centroid_axis=llm_axis, lda_axis=llm_axis)

    # Run stages 4 + 5
    python = sys.executable
    subprocess.run([python, "scripts/04_features.py", "--force"], cwd=str(PROJECT_ROOT), check=True)
    subprocess.run([python, "scripts/05_analysis.py", "--force"], cwd=str(PROJECT_ROOT), check=True)

    # Print results
    results_path = data_dir / "analysis_results.json"
    orig_path = Path(str(results_path) + ".orig_sh0")

    if results_path.exists():
        with open(results_path) as f:
            llm_res = json.load(f)
        print("\n  SH5d LLM — key attr-F1 correlations:")
        # Navigate the results structure
        corrs = llm_res.get("correlations", llm_res)
        if isinstance(corrs, dict):
            pairs = []
            for feat, targets in corrs.items():
                if isinstance(targets, dict):
                    for target, stats in targets.items():
                        if "attr" in target.lower() and isinstance(stats, dict):
                            rho = stats.get("spearman_rho", stats.get("rho", 0))
                            p = stats.get("p_value", stats.get("p", 1))
                            pairs.append((feat, target, rho, p))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for feat, target, rho, p in pairs[:5]:
                print(f"    {feat} ↔ {target}: ρ={rho:+.4f} (p={p:.2e})")

    # Restore
    restore_originals(data_dir, files_to_backup)


def swap_tags(llm_tags_path):
    """Swap shared tags file to LLM version."""
    backup = Path(str(SHARED_TAGS) + ".orig_sh0")
    if not backup.exists():
        shutil.copy2(SHARED_TAGS, backup)
    shutil.copy2(llm_tags_path, SHARED_TAGS)
    return backup

def restore_tags(backup):
    """Restore original tags file."""
    shutil.copy2(backup, SHARED_TAGS)
    backup.unlink()

def run_sh5a(llm_tags_path):
    """Rerun SH5a with LLM tags."""
    logging.info("\n" + "=" * 60)
    logging.info("RERUNNING SH5a")
    logging.info("=" * 60)

    data_dir = SH5A_ROOT / "data"
    tags_backup = swap_tags(llm_tags_path)

    python = sys.executable
    subprocess.run([python, "scripts/02_build_matrices.py", "--force"], cwd=str(SH5A_ROOT), check=True)
    subprocess.run([python, "scripts/03_extract_features.py", "--force"], cwd=str(SH5A_ROOT), check=True)
    subprocess.run([python, "scripts/04_analyze.py", "--force"], cwd=str(SH5A_ROOT), check=True)

    # Print key results
    corr_path = data_dir / "results" / "correlation_results.json"
    if corr_path.exists():
        with open(corr_path) as f:
            res = json.load(f)
        print("\n  SH5a LLM — top attr-F1 correlations:")
        pairs = []
        for feat, targets in res.items():
            if isinstance(targets, dict):
                for target, stats in targets.items():
                    if "attr" in target.lower() and isinstance(stats, dict):
                        rho = stats.get("spearman_rho", stats.get("rho", 0))
                        p = stats.get("p_value", stats.get("p", 1))
                        pairs.append((feat, rho, p))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, rho, p in pairs[:5]:
            print(f"    {feat}: ρ={rho:+.4f} (p={p:.2e})")

    restore_tags(tags_backup)


def run_sh5c(llm_tags_path):
    """Rerun SH5c with LLM tags."""
    logging.info("\n" + "=" * 60)
    logging.info("RERUNNING SH5c")
    logging.info("=" * 60)

    data_dir = SH5C_ROOT / "data"
    tags_backup = swap_tags(llm_tags_path)

    python = sys.executable
    subprocess.run([python, "scripts/01_load_and_extract.py", "--force"], cwd=str(SH5C_ROOT), check=True)
    subprocess.run([python, "scripts/02_alignment_features.py", "--force"], cwd=str(SH5C_ROOT), check=True)
    subprocess.run([python, "scripts/03_statistical_analysis.py", "--force"], cwd=str(SH5C_ROOT), check=True)

    # Print key results
    corr_path = data_dir / "results" / "correlations.json"
    if corr_path.exists():
        with open(corr_path) as f:
            res = json.load(f)
        print("\n  SH5c LLM — top attr-F1 correlations:")
        pairs = []
        for feat, targets in res.items():
            if isinstance(targets, dict):
                for target, stats in targets.items():
                    if "attr" in target.lower() and isinstance(stats, dict):
                        rho = stats.get("spearman_rho", stats.get("rho", 0))
                        p = stats.get("p_value", stats.get("p", 1))
                        pairs.append((feat, rho, p))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, rho, p in pairs[:5]:
            print(f"    {feat}: ρ={rho:+.4f} (p={p:.2e})")

    restore_tags(tags_backup)


def main():
    print("\n" + "=" * 60)
    print("SH5 FAMILY RERUN WITH LLM-DERIVED SLoD PROBE")
    print("=" * 60)

    # Step 1: Train LLM probe
    scaler, clf = train_llm_probe()

    # Step 2: Reclassify CoT steps
    llm_tags_path, step_embeddings = reclassify_cot_steps(scaler, clf)

    # Step 3: Compute LLM axis
    llm_axis = recompute_slod_axis()

    # Step 4: Run experiments
    run_sh5d(llm_axis)
    run_sh5a(llm_tags_path)
    run_sh5c(llm_tags_path)

    print("\n" + "=" * 60)
    print("ALL SH5 FAMILY RERUNS COMPLETE")
    print("=" * 60)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
