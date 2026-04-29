"""Stage-5 model-comparison driver.

Loads one run's ``trajectory_features.csv``, runs the requested models from
``failure_models.MODEL_REGISTRY`` under the protocol fixed by
``DESIGN-stage5-models.md``, and persists OOF predictions plus diagnostic
metadata under ``reports/{dataset}/{run_slug}/artifacts/``.

The driver supports two task types:

- ``task_type="classification"`` (default) preserves the original behaviour
  byte-for-byte: ``StratifiedKFold``, classifier minority-class feasibility
  check, ``oof_predictions_<model>.parquet`` with columns
  ``(id, row_index, fold, target, prob)``.
- ``task_type="regression"`` resolves the target to ``rubric_score``, uses
  plain ``KFold``, drops rows with missing target, and writes
  ``oof_regression_<model>.parquet`` with columns
  ``(id, row_index, fold, score, pred)``.

This driver is deliberately separate from ``failure_analysis.py``:

- ``failure_analysis.py`` is the **incumbent** Stage-5 pipeline. The classifier
  path here must not change its outputs.
- This module produces *additional* artifacts that the comparison and
  regression scripts depend on.

The two paths share the same feature-set logic (``choose_feature_sets``) and
classifier feasibility check so logreg numbers from this driver are
byte-identical to the existing report.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from semanticscale.sh6.failure_analysis import (
    META_COLUMNS,
    MEASUREMENT_COLUMNS,
    choose_feature_sets,
    prediction_feasibility,
    target_stats,
)
from semanticscale.sh6.failure_models import (
    OOFResult,
    TaskType,
    default_cv,
    get_spec,
    run_oof,
)

logger = logging.getLogger(__name__)


def _git_sha(repo_root: Path) -> str | None:
    """Best-effort short SHA. Returns None if not in a repo or git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _resolve_classification_target(df: pd.DataFrame, target_label: str) -> pd.Series:
    """Pick the binary target column, mirroring failure_analysis logic.

    Auto-resolution prefers ``final_answer_correct`` so SH6 reports stay
    consistent with the existing per-run failure_prediction.md.
    """
    if target_label == "auto":
        for candidate in ("final_answer_correct", "is_correct"):
            if candidate in df.columns and df[candidate].notna().any():
                target_label = candidate
                break
        else:
            msg = "No usable target column (final_answer_correct / is_correct)."
            raise ValueError(msg)
    if target_label not in df.columns:
        msg = f"Requested target '{target_label}' not in columns."
        raise ValueError(msg)
    return df[target_label].astype(float)


def _resolve_regression_target(df: pd.DataFrame, target_label: str) -> pd.Series:
    """Pick the continuous target column for regression.

    Default is ``rubric_score`` — the FrontierScience numeric-score target.
    Any column present in ``df`` may be passed explicitly; the values are
    coerced to float and rows with NaN are filtered downstream.
    """
    if target_label == "auto":
        target_label = "rubric_score"
    if target_label not in df.columns:
        msg = f"Requested regression target '{target_label}' not in columns."
        raise ValueError(msg)
    return pd.to_numeric(df[target_label], errors="coerce")


def _save_oof_classification(result: OOFResult, df: pd.DataFrame, artifact_dir: Path) -> Path:
    """Persist OOF probabilities (classification schema, unchanged)."""
    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy() if "id" in df.columns else np.arange(len(df)),
            "row_index": np.arange(len(df)),
            "fold": result.fold_assignments,
            "target": df["target"].astype(int).to_numpy(),
            "prob": result.predictions,
        }
    )
    path = artifact_dir / f"oof_predictions_{result.model_name}.parquet"
    out.to_parquet(path, index=False)
    return path


def _save_oof_regression(result: OOFResult, df: pd.DataFrame, artifact_dir: Path) -> Path:
    """Persist OOF predictions (regression schema)."""
    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy() if "id" in df.columns else np.arange(len(df)),
            "row_index": np.arange(len(df)),
            "fold": result.fold_assignments,
            "score": df["target"].astype(float).to_numpy(),
            "pred": result.predictions.astype(float),
        }
    )
    path = artifact_dir / f"oof_regression_{result.model_name}.parquet"
    out.to_parquet(path, index=False)
    return path


def _save_importance(result: OOFResult, artifact_dir: Path) -> Path | None:
    if result.feature_importance is None:
        return None
    suffix = "regression_importance" if result.task_type == "regression" else "feature_importance"
    path = artifact_dir / f"{suffix}_{result.model_name}.csv"
    result.feature_importance.to_csv(path, index=False)
    return path


def _save_metadata(
    result: OOFResult,
    feature_set_name: str,
    cv_folds: int,
    random_state: int,
    target_label: str,
    target_summary: dict,
    duration_s: float,
    git_sha: str | None,
    artifact_dir: Path,
) -> Path:
    payload: dict = {
        "model_name": result.model_name,
        "feature_set": feature_set_name,
        "n_features": result.n_features,
        "feature_cols": result.feature_cols,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "target_label": target_label,
    }
    # Preserve the original key name for the classifier metadata so existing
    # cv_metadata_*.json files remain byte-identical when re-generated.
    if result.task_type == "classification":
        payload["target_counts"] = target_summary
    else:
        payload["task_type"] = result.task_type
        payload["target_summary"] = target_summary
    payload["fold_metrics"] = result.fold_metrics
    if result.task_type == "classification":
        payload["confusion_matrix"] = result.confusion_matrix
    payload.update(
        {
            "duration_seconds": duration_s,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha,
            "python_version": sys.version.split()[0],
            "library_versions": _library_versions(),
        }
    )
    suffix = "regression_metadata" if result.task_type == "regression" else "cv_metadata"
    path = artifact_dir / f"{suffix}_{result.model_name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _library_versions() -> dict[str, str | None]:
    """Capture key library versions for reproducibility metadata."""
    versions: dict[str, str | None] = {}
    for module_name in ("sklearn", "lightgbm", "numpy", "pandas", "scipy"):
        try:
            mod = __import__(module_name)
            versions[module_name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[module_name] = None
    return versions


def _regression_target_summary(y: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(y.size),
        "mean": float(np.mean(y)) if y.size else 0.0,
        "std": float(np.std(y, ddof=0)) if y.size else 0.0,
        "min": float(np.min(y)) if y.size else 0.0,
        "max": float(np.max(y)) if y.size else 0.0,
    }


def _regression_feasibility(n: int, cv_folds: int) -> tuple[bool, int, str]:
    """Lightweight feasibility check for regression — no minority class to shrink."""
    if n < cv_folds * 2:
        return False, cv_folds, (
            f"Skipping regression: only {n} items with target — need ≥{cv_folds * 2} "
            f"for {cv_folds}-fold CV with two items per fold."
        )
    return True, cv_folds, ""


def run_models_on_run(
    features_csv: Path,
    artifact_dir: Path,
    models: list[str],
    *,
    feature_set: str = "trajectory_full",
    target_label: str = "auto",
    task_type: TaskType = "classification",
    cv_folds: int = 5,
    random_state: int = 42,
    repo_root: Path | None = None,
    feature_cols_override: list[str] | None = None,
) -> dict[str, dict]:
    """Run the requested models on one Stage-5 feature CSV.

    Returns a per-model summary dict (used by the orchestration script for its
    terminal summary). Heavy artifacts go to ``artifact_dir``.

    ``task_type`` selects the CV strategy, target resolver, scoring metrics,
    and OOF parquet schema. Defaults to ``"classification"`` so existing
    callers are unaffected.

    ``feature_cols_override`` lets the caller pass an explicit feature list
    (e.g. for the length-only baseline) in lieu of ``feature_set``.
    """
    import time

    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)

    if task_type == "classification":
        df["target"] = _resolve_classification_target(df, target_label).astype(float)
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        df["target"] = df["target"].astype(int)
        feasible, used_folds, note = prediction_feasibility(df, cv_folds)
        if not feasible:
            logger.warning("Skipping prediction: %s", note)
            return {"_skipped": {"reason": note}}
        if note:
            logger.info(note)
        target_summary = target_stats(df)
    elif task_type == "regression":
        df["target"] = _resolve_regression_target(df, target_label).astype(float)
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        feasible, used_folds, note = _regression_feasibility(len(df), cv_folds)
        if not feasible:
            logger.warning("Skipping regression: %s", note)
            return {"_skipped": {"reason": note}}
        target_summary = _regression_target_summary(df["target"].to_numpy())
    else:
        msg = f"Unknown task_type={task_type!r}"
        raise ValueError(msg)

    if feature_cols_override is not None:
        feature_cols = list(feature_cols_override)
    else:
        feature_sets = choose_feature_sets(df)
        if feature_set not in feature_sets:
            msg = f"Feature set '{feature_set}' not in {sorted(feature_sets)}"
            raise ValueError(msg)
        feature_cols = feature_sets[feature_set]
    if not feature_cols:
        msg = f"Feature set '{feature_set}' is empty for this run."
        raise ValueError(msg)

    X = df[feature_cols]
    y_arr = df["target"].to_numpy()
    cv = default_cv(task_type, used_folds, random_state)

    git_sha = _git_sha(repo_root or Path.cwd())
    summary: dict[str, dict] = {}

    for name in models:
        spec = get_spec(name)
        spec_task = getattr(spec, "task_type", "classification")
        if spec_task != task_type:
            msg = (
                f"Model '{name}' has task_type='{spec_task}' but driver was "
                f"called with task_type='{task_type}'."
            )
            raise ValueError(msg)
        logger.info("Fitting %s on %d features (%d items)", name, len(feature_cols), len(df))
        t0 = time.perf_counter()
        result = run_oof(spec, X, y_arr, cv, random_state)
        duration = time.perf_counter() - t0

        if task_type == "classification":
            oof_path = _save_oof_classification(result, df, artifact_dir)
        else:
            oof_path = _save_oof_regression(result, df, artifact_dir)
        imp_path = _save_importance(result, artifact_dir)
        meta_path = _save_metadata(
            result,
            feature_set_name=feature_set if feature_cols_override is None else "custom",
            cv_folds=used_folds,
            random_state=random_state,
            target_label=(target_label if target_label != "auto" else "auto-resolved"),
            target_summary=target_summary,
            duration_s=duration,
            git_sha=git_sha,
            artifact_dir=artifact_dir,
        )

        if task_type == "classification":
            roc = result.fold_metrics["roc_auc"]
            summary[name] = {
                "auc_mean": roc["mean"],
                "auc_std": roc["std"],
                "duration_s": duration,
                "oof_path": str(oof_path),
                "importance_path": str(imp_path) if imp_path else None,
                "metadata_path": str(meta_path),
            }
        else:
            r2 = result.fold_metrics["r2"]
            rho = result.fold_metrics.get("spearman", {"mean": float("nan"), "std": float("nan")})
            summary[name] = {
                "r2_mean": r2["mean"],
                "r2_std": r2["std"],
                "spearman_mean": rho["mean"],
                "spearman_std": rho["std"],
                "duration_s": duration,
                "oof_path": str(oof_path),
                "importance_path": str(imp_path) if imp_path else None,
                "metadata_path": str(meta_path),
            }
    return summary
