#!/usr/bin/env python
"""SH6 — Stage 10: FrontierScience Olympiad coverage--risk curve.

Casts the existing best-of-5 SLoD-LightGBM scorer (Stage 09) as a selective
prediction signal: at coverage c, retain the top-c fraction of attempts ranked
by score and report the residual error rate.

Two scorers are compared:

* SLoD: the per-attempt OOF probabilities from
  ``oof_predictions_lightgbm.parquet`` produced by Stage 09.
* Length: a ``length_only`` logistic regression
  (``[reasoning_n_chunks, answer_n_chunks, total_n_chunks]``) fit OOF inline
  with the same StratifiedKFold(5, shuffle=True, random_state=42) splits the
  paper uses elsewhere, then pooled across seeds.

The curves are pooled across all five DeepSeek seed runs (~480 attempts total),
giving a single coverage--risk plot for the abstract's selective-sampling
framing. Random-retention is a horizontal line at the base error rate; we note
this in the caption rather than plot a third curve.

Outputs:
    SLoD-ICML-2026-Workshop/figures/fig_coverage_risk.pdf
    experiments/sh6_llm-pairwise-slod/reports/_cross_dataset/coverage_risk.json

Usage:
    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/10_selective_prediction_curve.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "experiments/sh6_llm-pairwise-slod/reports"
PAPER_FIGURES = REPO_ROOT / "SLoD-ICML-2026-Workshop/figures"
CROSS_DATASET_DIR = REPORTS_ROOT / "_cross_dataset"

RUNS = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s2",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s3",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s4",
]
ORIGIN = "olympiad"
LENGTH_FEATURES = ["reasoning_n_chunks", "answer_n_chunks", "total_n_chunks"]


def _slod_oof_path(run: str) -> Path:
    return REPORTS_ROOT / run / f"by-origin/{ORIGIN}/artifacts/oof_predictions_lightgbm.parquet"


def _features_path(run: str) -> Path:
    return REPORTS_ROOT / run / f"by-origin/{ORIGIN}/trajectory_features.csv"


def load_slod_scores() -> pd.DataFrame:
    """Pool the 5 per-seed OOF parquets into a single attempt table."""
    frames = []
    for run in RUNS:
        path = _slod_oof_path(run)
        df = pd.read_parquet(path)
        if df["id"].nunique() != len(df):
            raise ValueError(
                f"OOF parquet at {path} has multiple rows per id; pooling assumes one OOF prob per id per seed"
            )
        df = df[["id", "target", "prob"]].copy()
        df["run"] = run
        frames.append(df)
        logger.info("loaded %d SLoD OOF rows from %s", len(df), run)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def fit_length_oof(features_df: pd.DataFrame) -> pd.DataFrame:
    """OOF logistic regression over LENGTH_FEATURES with the canonical CV split."""
    y = features_df["is_correct"].astype(int).to_numpy()
    X = features_df[LENGTH_FEATURES].to_numpy()
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.full(len(features_df), np.nan)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        pipeline.fit(X[train_idx], y[train_idx])
        probs[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
    out = features_df[["id", "is_correct"]].copy()
    out = out.rename(columns={"is_correct": "target"})
    out["target"] = out["target"].astype(int)
    out["prob"] = probs
    return out


def load_length_scores() -> pd.DataFrame:
    """OOF length-only logreg scores pooled across the 5 seeds."""
    frames = []
    for run in RUNS:
        path = _features_path(run)
        feats = pd.read_csv(path)
        if not all(col in feats.columns for col in LENGTH_FEATURES + ["is_correct", "id"]):
            missing = [c for c in LENGTH_FEATURES + ["is_correct", "id"] if c not in feats.columns]
            raise KeyError(f"Missing required columns in {path}: {missing}")
        scored = fit_length_oof(feats)
        scored["run"] = run
        frames.append(scored)
        logger.info("fit length-only OOF on %d rows from %s", len(scored), run)
    return pd.concat(frames, ignore_index=True)


def coverage_risk_curve(scores: pd.DataFrame, n_steps: int = 20) -> dict:
    """Sweep coverage from 1/N up to 1.0; report retained-accuracy and risk.

    `scores` columns: target (0/1), prob.
    """
    df = scores.dropna(subset=["prob"]).copy()
    if df.empty:
        raise ValueError("No non-null scores")
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    n = len(df)
    coverages = np.linspace(1 / n, 1.0, n_steps)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        retained = df.iloc[:k]
        risk = float(1.0 - retained["target"].mean())
        acc = float(retained["target"].mean())
        rows.append({"coverage": float(c), "retained_n": int(k), "risk": risk, "accuracy": acc})
    base_acc = float(df["target"].mean())
    return {
        "n_attempts": int(n),
        "base_accuracy": base_acc,
        "base_risk": 1.0 - base_acc,
        "points": rows,
    }


def render_figure(slod_curve: dict, length_curve: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slod_pts = slod_curve["points"]
    length_pts = length_curve["points"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    ax.plot(
        [p["coverage"] for p in length_pts],
        [p["accuracy"] for p in length_pts],
        marker="s",
        markersize=3,
        linewidth=1.2,
        color="#888888",
        label="Length-only (logreg)",
    )
    ax.plot(
        [p["coverage"] for p in slod_pts],
        [p["accuracy"] for p in slod_pts],
        marker="o",
        markersize=3,
        linewidth=1.6,
        color="#0d4f8b",
        label="SLoD (LightGBM)",
    )
    ax.axhline(
        slod_curve["base_accuracy"],
        linestyle=":",
        linewidth=1.0,
        color="#444444",
        label=f"No selection ({100 * slod_curve['base_accuracy']:.1f}%)",
    )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Pass@1 on retained")
    ax.set_xlim(0.0, 1.02)
    ymin = min(slod_curve["base_accuracy"], length_curve["base_accuracy"]) - 0.05
    ax.set_ylim(max(0.5, ymin), 1.02)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> None:
    setup_logging()

    slod_pool = load_slod_scores()
    length_pool = load_length_scores()

    slod_curve = coverage_risk_curve(slod_pool)
    length_curve = coverage_risk_curve(length_pool)

    sidecar = {
        "n_runs": len(RUNS),
        "runs": RUNS,
        "slod": slod_curve,
        "length": length_curve,
    }
    CROSS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    sidecar_path = CROSS_DATASET_DIR / "coverage_risk.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    logger.info("wrote %s", sidecar_path)

    render_figure(slod_curve, length_curve, PAPER_FIGURES / "fig_coverage_risk.pdf")

    # Summary print for spot-checking
    base = slod_curve["base_accuracy"]
    for c_target in (0.2, 0.4, 0.6, 0.8, 1.0):
        slod_pt = min(slod_curve["points"], key=lambda p: abs(p["coverage"] - c_target))
        length_pt = min(length_curve["points"], key=lambda p: abs(p["coverage"] - c_target))
        logger.info(
            "coverage=%.2f  SLoD acc=%.3f  length acc=%.3f  base=%.3f",
            slod_pt["coverage"],
            slod_pt["accuracy"],
            length_pt["accuracy"],
            base,
        )


if __name__ == "__main__":
    main()
