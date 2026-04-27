#!/usr/bin/env python
"""SH6 — Stage 7: Advanced failure-prediction analyses.

Two complementary analyses on top of stage 05's logistic baselines:

1. **Length-residualized trajectory features.** Each shape feature is
   residualized against ``[reasoning_n_chunks, answer_n_chunks]`` per CV
   fold, then a logistic regression is fit on the residuals. If a model on
   residualized shape features cannot beat ``length_only``, the trajectory
   signal on this dataset is just length in disguise.

2. **Gradient-boosted models with subject as a feature.** A
   ``HistGradientBoostingClassifier`` captures non-linearities and per-domain
   regimes that the logistic baseline misses. Compares:
     - GBM on length features only
     - GBM on length + trajectory shape
     - GBM on length + trajectory shape + subject (one-hot)

All models share the stage-05 cross-validation setup (stratified 5-fold,
``random_state=42``) so AUCs are directly comparable.

Inputs:
    {data_dir}/{dataset}/{slug}/trajectory_features.csv  (from stage 05)

Outputs:
    {reports_dir}/{dataset}/{slug}/advanced_failure_prediction.md
    {reports_dir}/{dataset}/{slug}/advanced_failure_prediction_summary.json

Usage:
    python scripts/07_advanced_failure_analysis.py --config <config>.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from semanticscale.sh6 import datasets as ds
from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

LENGTH_COLS = ["reasoning_n_chunks", "answer_n_chunks", "total_n_chunks"]
META_COLS = {
    "id", "dataset", "run_slug", "subject", "generator", "model",
    "target", "target_label", "is_correct", "final_answer_correct",
    "error_step_index", "error_step_position", "has_answer_chunks",
    "exit_status",
}
DENSITY_COLS = {"reasoning_pair_density", "answer_pair_density"}


class LengthResidualizer(BaseEstimator, TransformerMixin):
    """Replace shape columns with their residuals after OLS regression on length.

    Drops the length columns from the output. Per-fold fitting prevents
    leakage. ``length_idx`` indexes columns of the imputed input matrix.
    """

    def __init__(self, length_idx: list[int]):
        self.length_idx = length_idx

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        length_idx = list(self.length_idx)
        L = X[:, length_idx]
        n_cols = X.shape[1]
        length_set = set(length_idx)
        self.shape_idx_ = [j for j in range(n_cols) if j not in length_set]
        self.coef_ = np.zeros((len(self.shape_idx_), L.shape[1]))
        self.intercept_ = np.zeros(len(self.shape_idx_))
        for k, j in enumerate(self.shape_idx_):
            y_j = X[:, j]
            reg = LinearRegression().fit(L, y_j)
            self.coef_[k] = reg.coef_
            self.intercept_[k] = reg.intercept_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        L = X[:, list(self.length_idx)]
        S = X[:, self.shape_idx_]
        pred = L @ self.coef_.T + self.intercept_
        return S - pred


def _select_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (length_cols, shape_cols) actually present and usable."""
    length = [c for c in LENGTH_COLS if c in df.columns]
    shape = []
    for col in df.columns:
        if col in META_COLS or col in DENSITY_COLS or col in length:
            continue
        if df[col].dtype == object:
            continue
        nonnull = df[col].dropna()
        if nonnull.empty or nonnull.nunique() < 2:
            continue
        shape.append(col)
    return length, shape


def _logreg(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=random_state
    )


def _gbm(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=random_state,
    )


def _logreg_pipeline(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", _logreg(random_state)),
    ])


def _logreg_residualized_pipeline(
    n_total: int, length_idx: list[int], random_state: int
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("residualizer", LengthResidualizer(length_idx)),
        ("scaler", StandardScaler()),
        ("logreg", _logreg(random_state)),
    ])


def _gbm_pipeline_numeric(random_state: int) -> HistGradientBoostingClassifier:
    # HGB handles NaN natively, so no imputer is needed.
    return _gbm(random_state)


def _gbm_pipeline_with_subject(numeric_cols: list[str], random_state: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "subj",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["subject"],
            ),
        ]
    )
    return Pipeline([("pre", pre), ("gbm", _gbm(random_state))])


def _cv_score(estimator, X, y, cv) -> tuple[float, float]:
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    dataset_name = ds.dataset_name(config)
    slug = ds.run_slug(config)

    fa_cfg = config.get("failure_analysis", {})
    cv_folds = int(fa_cfg.get("cv_folds", 5))
    random_state = int(fa_cfg.get("random_state", 42))

    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve() / dataset_name / slug
    features_path = reports_dir / "trajectory_features.csv"
    if not features_path.exists():
        logger.error("Missing %s — run stage 05 first", features_path)
        raise SystemExit(1)

    df = pd.read_csv(features_path)
    if "target" not in df.columns:
        logger.error("trajectory_features.csv has no 'target' column")
        raise SystemExit(1)

    df = df.dropna(subset=["target"]).copy()
    y = df["target"].astype(int).to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < cv_folds or n_neg < cv_folds:
        cv_folds = max(2, min(n_pos, n_neg))
        logger.warning("Shrinking cv_folds to %d due to small class counts", cv_folds)

    length_cols, shape_cols = _select_columns(df)
    logger.info(
        "Items: %d (pos=%d, neg=%d). Length cols: %d. Shape cols: %d.",
        len(df), n_pos, n_neg, len(length_cols), len(shape_cols),
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: dict[str, dict] = {}

    # --- Baselines (logreg) ---
    X_len = df[length_cols].to_numpy(dtype=float)
    mean, std = _cv_score(_logreg_pipeline(random_state), X_len, y, cv)
    results["logreg__length_only"] = {
        "auc_mean": mean, "auc_std": std,
        "features": length_cols, "n_features": len(length_cols),
        "model": "logreg",
    }

    X_full_cols = length_cols + shape_cols
    X_full = df[X_full_cols].to_numpy(dtype=float)
    mean, std = _cv_score(_logreg_pipeline(random_state), X_full, y, cv)
    results["logreg__length_plus_shape"] = {
        "auc_mean": mean, "auc_std": std,
        "features": X_full_cols, "n_features": len(X_full_cols),
        "model": "logreg",
    }

    # --- Analysis 1: length-residualized shape ---
    length_idx = list(range(len(length_cols)))
    n_total_cols = len(X_full_cols)
    pipe_resid = _logreg_residualized_pipeline(n_total_cols, length_idx, random_state)
    mean, std = _cv_score(pipe_resid, X_full, y, cv)
    results["logreg__residualized_shape"] = {
        "auc_mean": mean, "auc_std": std,
        "features": shape_cols, "n_features": len(shape_cols),
        "model": "logreg",
        "note": "shape features residualized against length per-fold; length cols dropped",
    }

    # --- Analysis 2: gradient boosting ---
    mean, std = _cv_score(_gbm_pipeline_numeric(random_state), X_len, y, cv)
    results["gbm__length_only"] = {
        "auc_mean": mean, "auc_std": std,
        "features": length_cols, "n_features": len(length_cols),
        "model": "hist_gradient_boosting",
    }

    mean, std = _cv_score(_gbm_pipeline_numeric(random_state), X_full, y, cv)
    results["gbm__length_plus_shape"] = {
        "auc_mean": mean, "auc_std": std,
        "features": X_full_cols, "n_features": len(X_full_cols),
        "model": "hist_gradient_boosting",
    }

    if "subject" in df.columns and df["subject"].nunique() > 1:
        df_subj = df[X_full_cols + ["subject"]].copy()
        df_subj["subject"] = df_subj["subject"].fillna("unknown").astype(str)
        pipe = _gbm_pipeline_with_subject(X_full_cols, random_state)
        mean, std = _cv_score(pipe, df_subj, y, cv)
        results["gbm__length_plus_shape_plus_subject"] = {
            "auc_mean": mean, "auc_std": std,
            "features": X_full_cols + ["subject(one_hot)"],
            "n_features": len(X_full_cols) + 1,
            "model": "hist_gradient_boosting",
            "subjects": sorted(df_subj["subject"].unique().tolist()),
        }
    else:
        logger.info("No usable subject column; skipping subject-augmented GBM")

    # --- Write outputs ---
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset_name,
        "run_slug": slug,
        "n_items": int(len(df)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "n_length_features": len(length_cols),
        "n_shape_features": len(shape_cols),
        "results": results,
    }
    summary_path = reports_dir / "advanced_failure_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary JSON to %s", summary_path)

    md_lines = [
        f"# SH6 {dataset_name}/{slug} — Advanced Failure Prediction",
        "",
        "## Setup",
        "",
        f"- Items: {len(df)} (positive={n_pos}, negative={n_neg})",
        f"- Cross-validation: stratified {cv_folds}-fold, random_state={random_state}",
        f"- Length features ({len(length_cols)}): {', '.join(length_cols) or '_none_'}",
        f"- Shape features: {len(shape_cols)}",
        "",
        "## Results",
        "",
        "| Model | Features | # Features | ROC-AUC (mean ± std) |",
        "|---|---|---|---|",
    ]
    label_for = {
        "logreg__length_only": ("logreg", "length only"),
        "logreg__length_plus_shape": ("logreg", "length + shape"),
        "logreg__residualized_shape": (
            "logreg", "length-residualized shape (length dropped)"
        ),
        "gbm__length_only": ("gbm (HGB)", "length only"),
        "gbm__length_plus_shape": ("gbm (HGB)", "length + shape"),
        "gbm__length_plus_shape_plus_subject": (
            "gbm (HGB)", "length + shape + subject"
        ),
    }
    for key, res in results.items():
        model, feats = label_for.get(key, (res["model"], key))
        md_lines.append(
            f"| {model} | {feats} | {res['n_features']} "
            f"| {res['auc_mean']:.3f} ± {res['auc_std']:.3f} |"
        )

    # Interpretation — derived directly from the AUCs.
    base = results["logreg__length_only"]["auc_mean"]
    resid = results["logreg__residualized_shape"]["auc_mean"]
    full = results["logreg__length_plus_shape"]["auc_mean"]
    gbm_full = results["gbm__length_plus_shape"]["auc_mean"]
    gbm_subj = results.get("gbm__length_plus_shape_plus_subject", {}).get(
        "auc_mean"
    )

    md_lines += [
        "",
        "## Interpretation",
        "",
        f"- **Length baseline (logreg)**: AUC {base:.3f}. This is the bar.",
        (
            f"- **Length-residualized shape**: AUC {resid:.3f}. "
            + (
                f"Shape carries Δ = {resid - 0.5:+.3f} above chance after "
                "removing length."
                if resid > 0.5
                else "Residual shape is at or below chance — the SLoD "
                "trajectory adds no information beyond length on this run."
            )
        ),
        (
            f"- **Gradient boosting (length + shape)**: AUC {gbm_full:.3f} "
            f"vs logreg {full:.3f}. "
            + (
                f"Δ = {gbm_full - full:+.3f}: non-linearity helps."
                if gbm_full > full + 0.01
                else "Non-linearity does not lift the model meaningfully on "
                "this run."
            )
        ),
    ]
    if gbm_subj is not None:
        md_lines.append(
            f"- **GBM + subject**: AUC {gbm_subj:.3f}. "
            + (
                f"Δ vs GBM-without-subject = {gbm_subj - gbm_full:+.3f}: "
                "domain stratification helps."
                if gbm_subj > gbm_full + 0.01
                else "Subject stratification does not lift the GBM "
                "meaningfully on this run."
            )
        )
    md_lines += [
        "",
        "## Caveats",
        "",
        "- Residualization is performed per fold to avoid leakage; on small "
        "samples, fold-to-fold residualizer fits add variance, so the "
        "residualized AUC has slightly wider standard deviation than a "
        "naive whole-dataset residualization would.",
        "- HistGradientBoostingClassifier uses native NaN handling, no "
        "imputation is applied to GBM inputs.",
    ]
    md_path = reports_dir / "advanced_failure_prediction.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    logger.info("Saved markdown report to %s", md_path)


if __name__ == "__main__":
    main()
