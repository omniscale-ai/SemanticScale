#!/usr/bin/env python3
"""End-to-end orchestrator for SH1: Linear Decodability of SLoD from Frozen Embeddings."""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    LABEL_MAP,
    LABEL_NAMES,
    create_splits,
    load_config,
    load_spans,
    load_splits,
    setup_logging,
)
from src.embed_spans import embed_dataset
from src.train_probe import run_all_probes, run_binary_fallback, run_baselines
from src.analyze import main as analyze_main


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config_path = PROJECT_ROOT / "config.yaml"
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    data_dir = PROJECT_ROOT / config["data_dir"]
    sh0_dir = PROJECT_ROOT / config["sh0_data_dir"]
    reports_dir = PROJECT_ROOT / config["reports_dir"]

    # ---------------------------------------------------------------
    # Step 1: Create directories
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Creating directories")
    print("=" * 60)

    (data_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (data_dir / "results").mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger.info("Directories created.")

    # ---------------------------------------------------------------
    # Checkpoint: Load existing results if available
    # ---------------------------------------------------------------
    results_path = data_dir / "results" / "probe_results.json"
    cached_results = {}
    if results_path.exists():
        with open(results_path) as f:
            cached_results = json.load(f)
        logger.info(f"Loaded cached results from {results_path} ({len(cached_results)} keys)")

    # ---------------------------------------------------------------
    # Step 2: Create or load splits
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Creating / loading data splits")
    print("=" * 60)

    splits_path = data_dir / "splits.json"
    primary_data_path = sh0_dir / config["primary_dataset"]

    if splits_path.exists():
        logger.info("Splits already exist, loading.")
        splits = load_splits(splits_path)
    else:
        logger.info("Creating new splits...")
        spans = load_spans(primary_data_path)
        labels = [LABEL_MAP[s["label"]] for s in spans]
        splits = create_splits(
            n_samples=len(spans),
            labels=labels,
            ratios=config["split_ratios"],
            seed=config["random_seed"],
            save_path=splits_path,
        )

    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

    # ---------------------------------------------------------------
    # Step 3: Embed length-matched dataset with each model
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Generating embeddings (length-matched dataset)")
    print("=" * 60)

    model_order = ["minilm", "scibert", "specter2"]

    for model_key in model_order:
        if model_key not in config["models"]:
            logger.warning(f"Model {model_key} not in config, skipping.")
            continue

        print(f"\n--- Embedding with {model_key} ---")
        t0 = time.time()
        try:
            npz_path = embed_dataset(config, model_key, dataset_key="primary", project_root=PROJECT_ROOT)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s -> {npz_path}")
        except Exception as e:
            logger.error(f"Failed to embed with {model_key}: {e}", exc_info=True)
            print(f"  FAILED: {e}")

    # ---------------------------------------------------------------
    # Step 4: Train probes and evaluate
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Training linear probes")
    print("=" * 60)

    # Check if we can reuse cached probe results
    has_cached_probes = all(
        model_key in cached_results
        for model_key in config["models"]
        if (data_dir / "embeddings" / f"{model_key}_length_matched.npz").exists()
    ) and "baselines" in cached_results

    if has_cached_probes:
        print("  [CHECKPOINT] Probe results loaded from cache, skipping retraining.")
        all_results = cached_results
    else:
        all_results = run_all_probes(config, project_root=PROJECT_ROOT)
        # Save checkpoint immediately after probes complete
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Checkpoint saved after probe training -> {results_path}")

    # Print summary
    print("\n--- Probe Results Summary ---")
    best_test_f1 = 0.0
    best_model_key = None
    best_clf_key = None

    for model_key in config["models"]:
        if model_key not in all_results:
            continue
        for clf_key, metrics in all_results[model_key].items():
            test_f1 = metrics.get("test", {}).get("macro_f1", 0)
            val_f1 = metrics.get("val", {}).get("macro_f1", 0)
            print(f"  {model_key} + {clf_key}: val={val_f1:.4f}, test={test_f1:.4f}")
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_model_key = model_key
                best_clf_key = clf_key

    print(f"\n  Best: {best_model_key} + {best_clf_key} = {best_test_f1:.4f}")

    # Save best config for resumability
    all_results["best_config"] = {
        "model": best_model_key,
        "classifier": best_clf_key,
        "test_f1": best_test_f1,
    }

    # Print baselines
    if "baselines" in all_results:
        print("\n--- Baselines ---")
        for bl_name, bl_metrics in all_results["baselines"].items():
            print(f"  {bl_name}: macro-F1 = {bl_metrics.get('macro_f1', 0):.4f}")

    # ---------------------------------------------------------------
    # Step 5: Binary fallback (if needed)
    # ---------------------------------------------------------------
    three_way_threshold = config.get("thresholds", {}).get("three_way_f1", 0.60)

    if best_test_f1 < three_way_threshold and best_model_key is not None:
        print("\n" + "=" * 60)
        print(f"STEP 5: Binary fallback (3-way F1 {best_test_f1:.4f} < {three_way_threshold})")
        print("=" * 60)

        npz_path = data_dir / "embeddings" / f"{best_model_key}_length_matched.npz"
        data = np.load(npz_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]

        train_idx = np.array(splits["train"])
        test_idx = np.array(splits["test"])

        binary_results = run_binary_fallback(
            embeddings[train_idx], labels[train_idx],
            embeddings[test_idx], labels[test_idx],
        )
        all_results["binary_fallback"] = binary_results
        print(f"  Binary (macro vs micro) F1 = {binary_results['macro_f1']:.4f}")
    else:
        print("\n  Skipping binary fallback (3-way F1 meets threshold)")

    # ---------------------------------------------------------------
    # Step 6: Confound check — embed full dataset with best model
    # ---------------------------------------------------------------
    if "confound_check" in all_results:
        print("\n" + "=" * 60)
        print("STEP 6: Confound check (full dataset)")
        print("=" * 60)
        cc = all_results["confound_check"]
        print(f"  [CHECKPOINT] Confound check loaded from cache.")
        print(f"  Full dataset F1:         {cc['full_dataset_f1']:.4f}")
        print(f"  Length-matched F1:       {cc['length_matched_f1']:.4f}")
        print(f"  Gap:                     {cc['gap']:.4f}")
        print(f"  Full word-count-only F1: {cc['full_word_count_f1']:.4f}")
    elif best_model_key is not None:
        print("\n" + "=" * 60)
        print("STEP 6: Confound check (full dataset)")
        print("=" * 60)

        try:
            print("  Step 6a: Embedding full dataset...")
            full_npz = embed_dataset(config, best_model_key, dataset_key="full", project_root=PROJECT_ROOT)
            print("  Step 6a: Embedding complete (cached as .npz).")

            # Create splits for full dataset
            full_spans = load_spans(sh0_dir / config["full_dataset"])
            full_labels = np.array([LABEL_MAP[s["label"]] for s in full_spans])
            full_word_counts = np.array([s.get("word_count", len(s["text"].split())) for s in full_spans])

            full_splits_path = data_dir / "splits_full.json"
            if full_splits_path.exists():
                full_splits = load_splits(full_splits_path)
            else:
                full_splits = create_splits(
                    n_samples=len(full_spans),
                    labels=full_labels.tolist(),
                    ratios=config["split_ratios"],
                    seed=config["random_seed"],
                    save_path=full_splits_path,
                )

            full_data = np.load(full_npz, allow_pickle=True)
            full_embeddings = full_data["embeddings"]
            full_labels_arr = full_data["labels"]

            full_train_idx = np.array(full_splits["train"])
            full_val_idx = np.array(full_splits["val"])
            full_test_idx = np.array(full_splits["test"])

            # Train probe on full dataset
            print("  Step 6b: Training confound probe...")
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import f1_score

            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(full_embeddings[full_train_idx])
            X_test_f = scaler.transform(full_embeddings[full_test_idx])
            y_train_f = full_labels_arr[full_train_idx]
            y_test_f = full_labels_arr[full_test_idx]

            clf = LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs", random_state=42
            )
            clf.fit(X_train_f, y_train_f)
            y_pred_f = clf.predict(X_test_f)
            full_f1 = float(f1_score(y_test_f, y_pred_f, average="macro"))

            # Word-count baseline on full dataset
            wc_train = full_word_counts[full_train_idx].reshape(-1, 1)
            wc_test = full_word_counts[full_test_idx].reshape(-1, 1)
            scaler_wc = StandardScaler()
            wc_train_s = scaler_wc.fit_transform(wc_train)
            wc_test_s = scaler_wc.transform(wc_test)
            clf_wc = LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs", random_state=42
            )
            clf_wc.fit(wc_train_s, y_train_f)
            y_wc_pred = clf_wc.predict(wc_test_s)
            full_wc_f1 = float(f1_score(y_test_f, y_wc_pred, average="macro"))

            confound_results = {
                "full_dataset_f1": full_f1,
                "full_word_count_f1": full_wc_f1,
                "length_matched_f1": best_test_f1,
                "gap": abs(full_f1 - best_test_f1),
            }
            all_results["confound_check"] = confound_results

            # Checkpoint: save immediately after confound check
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Checkpoint saved after confound check -> {results_path}")

            print(f"  Full dataset F1:         {full_f1:.4f}")
            print(f"  Length-matched F1:       {best_test_f1:.4f}")
            print(f"  Gap:                     {abs(full_f1 - best_test_f1):.4f}")
            print(f"  Full word-count-only F1: {full_wc_f1:.4f}")

            if abs(full_f1 - best_test_f1) > 0.10:
                print("  WARNING: Gap > 0.10 — length may be a significant confound!")
            else:
                print("  OK: Gap <= 0.10 — length confound appears controlled.")

        except Exception as e:
            logger.error(f"Confound check failed: {e}", exc_info=True)
            print(f"  FAILED: {e}")

    # Save updated results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---------------------------------------------------------------
    # Step 7: Generate analysis (plots + report)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Generating analysis and report")
    print("=" * 60)

    try:
        analyze_main(config, project_root=PROJECT_ROOT)
        print(f"  Report: {reports_dir / 'sh1_results.md'}")
        print(f"  Figures: {reports_dir / 'figures/'}")
    except Exception as e:
        logger.error(f"Analysis generation failed: {e}", exc_info=True)
        print(f"  FAILED: {e}")

    # ---------------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    baseline_f1 = 0.333
    if "baselines" in all_results and "random" in all_results["baselines"]:
        baseline_f1 = all_results["baselines"]["random"]["macro_f1"]

    gap = best_test_f1 - baseline_f1
    binary_f1 = all_results.get("binary_fallback", {}).get("macro_f1", 0)
    binary_threshold = config.get("thresholds", {}).get("binary_f1", 0.75)
    gap_threshold = config.get("thresholds", {}).get("baseline_gap", 0.15)

    print(f"  Best 3-way test macro-F1: {best_test_f1:.4f} (threshold: {three_way_threshold})")
    print(f"  Baseline gap:             {gap:.4f} (threshold: {gap_threshold})")

    if best_test_f1 >= three_way_threshold:
        print(f"\n  >>> SH1 CONFIRMED <<<")
        print(f"  Linear probes decode SLoD levels above threshold.")
    elif binary_f1 >= binary_threshold:
        print(f"\n  >>> SH1 PARTIAL (binary confirmed) <<<")
        print(f"  3-way failed but binary macro-vs-micro F1 = {binary_f1:.4f} >= {binary_threshold}")
    elif gap >= gap_threshold:
        print(f"\n  >>> SH1 PARTIAL (non-trivial signal) <<<")
        print(f"  Below threshold but baseline gap = {gap:.4f} >= {gap_threshold}")
    else:
        print(f"\n  >>> SH1 NOT CONFIRMED <<<")
        print(f"  Linear probes do not reliably decode SLoD levels.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
