"""Stage E: Model training with paper-level splits, C sweep, ablation study."""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import ALL_FEATURES, CONFIDENCE_FEATURES, DRIFT_FEATURES, SURFACE_FEATURES
from src.utils import get_project_root, load_config, read_json, write_json


def create_paper_level_splits(
    paper_ids: np.ndarray,
    y: np.ndarray,
    split_ratios: list[float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Create train/val/test splits at the paper level.

    All extractions from one paper go into the same split.
    Stratified by paper-level majority label.

    Returns dict with 'train', 'val', 'test' keys, each an array of sample indices.
    """
    unique_papers = np.unique(paper_ids)
    n_papers = len(unique_papers)
    logging.info(f"Creating paper-level splits for {n_papers} papers")

    # Compute paper-level majority label for stratification
    paper_labels = []
    for paper in unique_papers:
        mask = paper_ids == paper
        paper_y = y[mask]
        # Majority label
        paper_labels.append(int(np.round(np.mean(paper_y))))
    paper_labels = np.array(paper_labels)

    train_ratio, val_ratio, test_ratio = split_ratios

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    try:
        train_papers, valtest_papers, train_plabels, valtest_plabels = train_test_split(
            unique_papers,
            paper_labels,
            test_size=val_test_ratio,
            stratify=paper_labels,
            random_state=seed,
        )
    except ValueError:
        # Not enough samples for stratification, fall back to non-stratified
        logging.warning("Stratification failed, using non-stratified split")
        train_papers, valtest_papers = train_test_split(
            unique_papers,
            test_size=val_test_ratio,
            random_state=seed,
        )
        valtest_plabels = paper_labels[np.isin(unique_papers, valtest_papers)]

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    try:
        val_papers, test_papers = train_test_split(
            valtest_papers,
            test_size=relative_test,
            stratify=valtest_plabels,
            random_state=seed,
        )
    except ValueError:
        val_papers, test_papers = train_test_split(
            valtest_papers,
            test_size=relative_test,
            random_state=seed,
        )

    # Convert paper-level splits to sample-level indices
    train_set = set(train_papers)
    val_set = set(val_papers)
    test_set = set(test_papers)

    train_idx = np.where(np.isin(paper_ids, list(train_set)))[0]
    val_idx = np.where(np.isin(paper_ids, list(val_set)))[0]
    test_idx = np.where(np.isin(paper_ids, list(test_set)))[0]

    logging.info(
        f"Paper-level splits: "
        f"train={len(train_papers)} papers ({len(train_idx)} samples), "
        f"val={len(val_papers)} papers ({len(val_idx)} samples), "
        f"test={len(test_papers)} papers ({len(test_idx)} samples)"
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "train_papers": list(train_papers),
        "val_papers": list(val_papers),
        "test_papers": list(test_papers),
    }


def get_feature_indices(feature_names: list[str], selected: list[str] | str) -> list[int]:
    """Get column indices for selected features."""
    if selected == "all":
        return list(range(len(feature_names)))
    return [feature_names.index(f) for f in selected if f in feature_names]


def train_logreg_with_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list[float],
    max_iter: int = 2000,
    solver: str = "lbfgs",
) -> tuple[LogisticRegression, StandardScaler, float]:
    """Train LogReg with C sweep on validation set.

    Returns (best_model, scaler, best_C).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_auc = -1.0
    best_C = C_values[0]
    best_model = None

    for C in C_values:
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver, random_state=42
        )
        clf.fit(X_train_s, y_train)

        if len(np.unique(y_val)) < 2:
            # Can't compute AUROC with single class
            y_proba = clf.predict_proba(X_val_s)[:, 1]
            auc = 0.5
        else:
            y_proba = clf.predict_proba(X_val_s)[:, 1]
            auc = roc_auc_score(y_val, y_proba)

        logging.info(f"  C={C}: val AUROC = {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_C = C
            best_model = clf

    logging.info(f"  Best C={best_C} with val AUROC={best_auc:.4f}")
    return best_model, scaler, best_C


def evaluate_model(
    clf,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    precision_at_k_thresholds: list[float] = None,
) -> dict:
    """Evaluate a trained model on test data.

    Returns dict with AUROC, precision@k, Brier score, etc.
    """
    if precision_at_k_thresholds is None:
        precision_at_k_thresholds = [0.25, 0.50, 0.75]

    X_test_s = scaler.transform(X_test)
    y_proba = clf.predict_proba(X_test_s)[:, 1]  # P(correct)
    y_pred = clf.predict(X_test_s)

    results = {}

    # AUROC
    if len(np.unique(y_test)) >= 2:
        results["auroc"] = float(roc_auc_score(y_test, y_proba))
    else:
        results["auroc"] = 0.5
        logging.warning("Single class in test set, AUROC set to 0.5")

    # Brier score
    results["brier_score"] = float(brier_score_loss(y_test, y_proba))

    # Precision@k
    results["precision_at_k"] = {}
    sorted_indices = np.argsort(-y_proba)  # highest confidence first
    for k in precision_at_k_thresholds:
        n_retain = max(1, int(len(y_test) * k))
        top_indices = sorted_indices[:n_retain]
        top_correct = y_test[top_indices]
        prec = float(np.mean(top_correct))
        results["precision_at_k"][f"k={k:.2f}"] = prec
        logging.info(f"  Precision@{k:.0%}: {prec:.4f} (retain {n_retain}/{len(y_test)})")

    # Overall precision and accuracy
    results["accuracy"] = float(np.mean(y_pred == y_test))
    if np.sum(y_pred) > 0:
        results["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    else:
        results["precision"] = 0.0

    # Feature importances (coefficients for LogReg)
    if hasattr(clf, "coef_"):
        results["coefficients"] = clf.coef_[0].tolist()

    # Predictions for downstream analysis
    results["y_proba"] = y_proba.tolist()
    results["y_pred"] = y_pred.tolist()
    results["y_true"] = y_test.tolist()

    return results


def run_training(config: dict = None, project_root: Path = None) -> Path:
    """Run the full model training pipeline with ablation study.

    Returns path to the metrics JSON file.
    """
    if config is None:
        config = load_config()
    if project_root is None:
        project_root = get_project_root()

    model_cfg = config["model"]
    data_dir = project_root / config["data_dir"]

    # Output paths
    metrics_path = data_dir / model_cfg["output"]["metrics"]
    model_path = data_dir / model_cfg["output"]["model"]

    if metrics_path.exists():
        logging.info(f"Model metrics already exist at {metrics_path}, skipping.")
        return metrics_path

    # Load feature matrix
    feat_path = data_dir / config["features"]["output"]
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {feat_path}. Run Stage D first.")

    data = np.load(feat_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    paper_ids = data["paper_ids"]
    extraction_ids = data["extraction_ids"]

    logging.info(f"Loaded feature matrix: {X.shape}, {np.sum(y)} correct / {np.sum(1-y)} incorrect")

    # Paper-level splits
    splits = create_paper_level_splits(
        paper_ids=paper_ids,
        y=y,
        split_ratios=model_cfg["split_ratios"],
        seed=model_cfg["seed"],
    )

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logging.info(f"Train: {len(y_train)} ({np.mean(y_train):.1%} correct)")
    logging.info(f"Val: {len(y_val)} ({np.mean(y_val):.1%} correct)")
    logging.info(f"Test: {len(y_test)} ({np.mean(y_test):.1%} correct)")

    # Precision@k thresholds
    prec_k = config.get("analysis", {}).get("precision_at_k", [0.25, 0.50, 0.75])

    # --- Ablation Study ---
    ablation_cfg = model_cfg["ablation"]
    logreg_cfg = model_cfg["logreg"]
    all_results = {"splits": {
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "train_correct_rate": float(np.mean(y_train)),
        "val_correct_rate": float(np.mean(y_val)),
        "test_correct_rate": float(np.mean(y_test)),
    }}

    best_combined_model = None
    best_combined_scaler = None

    for ablation_name, abl_cfg in ablation_cfg.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Ablation: {ablation_name}")
        logging.info(f"{'='*60}")

        # Get feature indices
        selected_features = abl_cfg["features"]
        feat_idx = get_feature_indices(feature_names, selected_features)

        if not feat_idx:
            logging.warning(f"No features selected for {ablation_name}, skipping.")
            continue

        selected_names = [feature_names[i] for i in feat_idx]
        logging.info(f"Features ({len(feat_idx)}): {selected_names}")

        X_train_abl = X_train[:, feat_idx]
        X_val_abl = X_val[:, feat_idx]
        X_test_abl = X_test[:, feat_idx]

        # Train with C sweep
        clf, scaler, best_C = train_logreg_with_sweep(
            X_train_abl, y_train,
            X_val_abl, y_val,
            C_values=logreg_cfg["C_search"],
            max_iter=logreg_cfg["max_iter"],
            solver=logreg_cfg["solver"],
        )

        # Evaluate
        test_results = evaluate_model(
            clf, scaler, X_test_abl, y_test,
            precision_at_k_thresholds=prec_k,
        )
        test_results["best_C"] = best_C
        test_results["feature_names"] = selected_names
        test_results["n_features"] = len(feat_idx)

        all_results[ablation_name] = test_results
        logging.info(f"  Test AUROC: {test_results['auroc']:.4f}")

        if ablation_name == "combined":
            best_combined_model = clf
            best_combined_scaler = scaler

    # --- Optional: Gradient Boosted Trees ---
    gbt_cfg = model_cfg.get("gbt", {})
    if gbt_cfg.get("enabled", False):
        logging.info("\n--- Training GBT (fallback) ---")
        feat_idx = get_feature_indices(feature_names, "all")
        X_train_all = X_train[:, feat_idx]
        X_test_all = X_test[:, feat_idx]

        scaler_gbt = StandardScaler()
        X_train_s = scaler_gbt.fit_transform(X_train_all)
        X_test_s = scaler_gbt.transform(X_test_all)

        gbt = GradientBoostingClassifier(
            n_estimators=gbt_cfg.get("n_estimators", 100),
            max_depth=gbt_cfg.get("max_depth", 3),
            learning_rate=gbt_cfg.get("learning_rate", 0.1),
            random_state=42,
        )
        gbt.fit(X_train_s, y_train)

        gbt_results = evaluate_model(
            gbt, scaler_gbt, X_test_all, y_test,
            precision_at_k_thresholds=prec_k,
        )
        gbt_results["feature_importances"] = gbt.feature_importances_.tolist()
        gbt_results["feature_names"] = [feature_names[i] for i in feat_idx]
        all_results["gbt_combined"] = gbt_results
        logging.info(f"  GBT Test AUROC: {gbt_results['auroc']:.4f}")

    # Check if GBT should be enabled based on LogReg performance
    combined_auroc = all_results.get("combined", {}).get("auroc", 0.0)
    if combined_auroc < 0.65 and not gbt_cfg.get("enabled", False):
        logging.info("\nLogReg AUROC < 0.65, training GBT as fallback...")
        feat_idx = get_feature_indices(feature_names, "all")
        X_train_all = X_train[:, feat_idx]
        X_val_all = X_val[:, feat_idx]
        X_test_all = X_test[:, feat_idx]

        scaler_gbt = StandardScaler()
        X_train_s = scaler_gbt.fit_transform(X_train_all)
        X_test_s = scaler_gbt.transform(X_test_all)

        gbt = GradientBoostingClassifier(
            n_estimators=gbt_cfg.get("n_estimators", 100),
            max_depth=gbt_cfg.get("max_depth", 3),
            learning_rate=gbt_cfg.get("learning_rate", 0.1),
            random_state=42,
        )
        gbt.fit(X_train_s, y_train)

        gbt_results = evaluate_model(
            gbt, scaler_gbt, X_test_all, y_test,
            precision_at_k_thresholds=prec_k,
        )
        gbt_results["feature_importances"] = gbt.feature_importances_.tolist()
        gbt_results["feature_names"] = [feature_names[i] for i in feat_idx]
        all_results["gbt_combined"] = gbt_results
        logging.info(f"  GBT Test AUROC: {gbt_results['auroc']:.4f}")

    # Save metrics
    # Remove y_proba/y_pred/y_true from the saved metrics (they're large)
    # but keep them in a separate file for analysis
    analysis_data = {}
    metrics_clean = {}
    for key, val in all_results.items():
        if isinstance(val, dict) and "y_proba" in val:
            analysis_data[key] = {
                "y_proba": val.pop("y_proba"),
                "y_pred": val.pop("y_pred"),
                "y_true": val.pop("y_true"),
            }
        metrics_clean[key] = val

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics_clean, metrics_path)

    # Save analysis data (predictions) separately
    analysis_path = data_dir / "results" / "predictions.json"
    write_json(analysis_data, analysis_path)

    # Save split info
    split_info = {
        "train_papers": splits.get("train_papers", []),
        "val_papers": splits.get("val_papers", []),
        "test_papers": splits.get("test_papers", []),
    }
    write_json(split_info, data_dir / "results" / "paper_splits.json")

    # Save best combined model
    if best_combined_model is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": best_combined_model,
                "scaler": best_combined_scaler,
                "feature_names": feature_names,
            }, f)
        logging.info(f"Combined model saved to {model_path}")

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("=== Training Summary ===")
    for name in ["drift_only", "surface_only", "combined"]:
        if name in all_results and "auroc" in all_results[name]:
            auroc = all_results[name]["auroc"]
            logging.info(f"  {name}: AUROC = {auroc:.4f}")
    if "gbt_combined" in all_results:
        logging.info(f"  gbt_combined: AUROC = {all_results['gbt_combined']['auroc']:.4f}")

    # Check exit criteria
    thresholds = config.get("thresholds", {})
    drift_auroc = all_results.get("drift_only", {}).get("auroc", 0)
    surface_auroc = all_results.get("surface_only", {}).get("auroc", 0)
    drift_advantage = drift_auroc - surface_auroc

    logging.info(f"\n  Drift advantage: {drift_advantage:+.4f} "
                 f"(threshold: {thresholds.get('drift_advantage', 0.03)})")

    return metrics_path
