#!/usr/bin/env python3
"""Fast confound check (Step 6) using stratified subsampling of the full dataset.

Instead of embedding all 83K spans (~4 hours on CPU), subsample ~10K spans
stratified by label, embed those, and compare F1 with the length-matched result.
This gives a reliable estimate in ~30 minutes instead of ~4 hours.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    LABEL_MAP,
    LABEL_NAMES,
    load_config,
    load_spans,
    setup_logging,
)


def stratified_subsample(labels, n_per_class=3333, seed=42):
    """Return indices for a stratified subsample."""
    rng = np.random.RandomState(seed)
    indices = []
    for label_val in sorted(set(labels)):
        class_idx = np.where(labels == label_val)[0]
        n = min(n_per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=n, replace=False)
        indices.extend(chosen.tolist())
    return sorted(indices)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config(PROJECT_ROOT / "config.yaml")
    data_dir = PROJECT_ROOT / config["data_dir"]
    sh0_dir = PROJECT_ROOT / config["sh0_data_dir"]
    results_path = data_dir / "results" / "probe_results.json"

    # Load existing results
    with open(results_path) as f:
        all_results = json.load(f)

    if "confound_check" in all_results:
        print("Confound check already in results, skipping.")
        cc = all_results["confound_check"]
        for k, v in cc.items():
            print(f"  {k}: {v}")
        return

    best_config = all_results.get("best_config", {})
    best_model_key = best_config.get("model", "scibert")
    best_test_f1 = best_config.get("test_f1", 0.72)

    print(f"Best model: {best_model_key}, best test F1: {best_test_f1:.4f}")

    # Load full dataset
    print("Loading full dataset...")
    full_spans = load_spans(sh0_dir / config["full_dataset"])
    full_labels = np.array([LABEL_MAP[s["label"]] for s in full_spans])
    full_word_counts = np.array([s.get("word_count", len(s["text"].split())) for s in full_spans])

    print(f"Full dataset: {len(full_spans)} spans")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {(full_labels == i).sum()}")

    # Stratified subsample: 3333 per class = ~10K total
    N_PER_CLASS = 3333
    sub_idx = stratified_subsample(full_labels, n_per_class=N_PER_CLASS, seed=42)
    print(f"\nSubsampled {len(sub_idx)} spans ({N_PER_CLASS} per class)")

    sub_texts = [full_spans[i]["text"] for i in sub_idx]
    sub_labels = full_labels[sub_idx]
    sub_word_counts = full_word_counts[sub_idx]

    # Check for cached subsample embeddings
    embeddings_dir = data_dir / "embeddings"
    sub_npz_path = embeddings_dir / f"{best_model_key}_full_subsample.npz"

    if sub_npz_path.exists():
        print(f"Loading cached subsample embeddings from {sub_npz_path}")
        data = np.load(sub_npz_path, allow_pickle=True)
        sub_embeddings = data["embeddings"]
    else:
        # Embed subsample
        print(f"Embedding {len(sub_texts)} texts with {best_model_key}...")
        model_cfg = config["models"][best_model_key]

        from src.embed_spans import embed_with_transformers
        t0 = time.time()
        sub_embeddings = embed_with_transformers(
            sub_texts,
            model_name=model_cfg["name"],
            batch_size=model_cfg.get("batch_size", 64),
            max_length=model_cfg.get("max_length", 512),
        )
        elapsed = time.time() - t0
        print(f"Embedding done in {elapsed:.1f}s")

        # Save
        np.savez(sub_npz_path, embeddings=sub_embeddings, labels=sub_labels)
        print(f"Saved to {sub_npz_path}")

    # Split subsample: 70/15/15
    n = len(sub_labels)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # Train probe
    print("\nTraining confound probe on subsample...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    scaler = StandardScaler()
    X_train = scaler.fit_transform(sub_embeddings[train_idx])
    X_test = scaler.transform(sub_embeddings[test_idx])
    y_train = sub_labels[train_idx]
    y_test = sub_labels[test_idx]

    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    full_f1 = float(f1_score(y_test, y_pred, average="macro"))

    # Word-count baseline
    wc_train = sub_word_counts[train_idx].reshape(-1, 1)
    wc_test = sub_word_counts[test_idx].reshape(-1, 1)
    scaler_wc = StandardScaler()
    wc_train_s = scaler_wc.fit_transform(wc_train)
    wc_test_s = scaler_wc.transform(wc_test)
    clf_wc = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf_wc.fit(wc_train_s, y_train)
    y_wc_pred = clf_wc.predict(wc_test_s)
    full_wc_f1 = float(f1_score(y_test, y_wc_pred, average="macro"))

    confound_results = {
        "full_dataset_f1": full_f1,
        "full_word_count_f1": full_wc_f1,
        "length_matched_f1": best_test_f1,
        "gap": abs(full_f1 - best_test_f1),
        "note": f"Based on stratified subsample of {len(sub_idx)} spans (not full 83K)",
    }

    print(f"\n{'='*60}")
    print("CONFOUND CHECK RESULTS")
    print(f"{'='*60}")
    print(f"  Full dataset (subsample) F1: {full_f1:.4f}")
    print(f"  Length-matched F1:           {best_test_f1:.4f}")
    print(f"  Gap:                         {abs(full_f1 - best_test_f1):.4f}")
    print(f"  Full word-count-only F1:     {full_wc_f1:.4f}")

    if abs(full_f1 - best_test_f1) > 0.10:
        print("  WARNING: Gap > 0.10 -- length may be a significant confound!")
    else:
        print("  OK: Gap <= 0.10 -- length confound appears controlled.")

    # Save to results
    all_results["confound_check"] = confound_results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
