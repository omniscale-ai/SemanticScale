"""Stage-5 model-comparison driver.

Loads one run's ``trajectory_features.csv``, runs the requested models from
``failure_models.MODEL_REGISTRY`` under the protocol fixed by
``DESIGN-stage5-models.md``, and persists OOF predictions plus diagnostic
metadata under ``reports/{dataset}/{run_slug}/artifacts/``.

This driver is deliberately separate from ``failure_analysis.py``:

- ``failure_analysis.py`` is the **incumbent** Stage-5 pipeline. Step 3
  must not change its outputs.
- This module produces *additional* artifacts that the comparison
  scripts (Step 4–5) depend on.

The two paths share the same feature-set logic (``choose_feature_sets``)
and feasibility check so logreg numbers from this driver are byte-identical
to the existing report.
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
from sklearn.model_selection import StratifiedKFold

from semanticscale.sh6.failure_analysis import (
    META_COLUMNS,
    MEASUREMENT_COLUMNS,
    choose_feature_sets,
    prediction_feasibility,
    target_stats,
)
from semanticscale.sh6.failure_models import OOFResult, get_spec, run_oof

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


def _resolve_target(df: pd.DataFrame, target_label: str) -> pd.Series:
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


def _save_oof(result: OOFResult, df: pd.DataFrame, artifact_dir: Path) -> Path:
    """Persist OOF probabilities + fold assignments + target as parquet.

    Schema is identical across models so the cross-dataset aggregator can
    do straight `pd.concat`. We always include `id` (when available) so the
    paired bootstrap in Step 5 can join across model files even if the row
    order somehow drifts.
    """
    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy() if "id" in df.columns else np.arange(len(df)),
            "row_index": np.arange(len(df)),
            "fold": result.fold_assignments,
            "target": df["target"].astype(int).to_numpy(),
            "prob": result.probabilities,
        }
    )
    path = artifact_dir / f"oof_predictions_{result.model_name}.parquet"
    out.to_parquet(path, index=False)
    return path


def _save_importance(result: OOFResult, artifact_dir: Path) -> Path | None:
    if result.feature_importance is None:
        return None
    path = artifact_dir / f"feature_importance_{result.model_name}.csv"
    result.feature_importance.to_csv(path, index=False)
    return path


def _save_metadata(
    result: OOFResult,
    feature_set_name: str,
    cv_folds: int,
    random_state: int,
    target_label: str,
    target_counts: dict[str, int],
    duration_s: float,
    git_sha: str | None,
    artifact_dir: Path,
) -> Path:
    payload = {
        "model_name": result.model_name,
        "feature_set": feature_set_name,
        "n_features": result.n_features,
        "feature_cols": result.feature_cols,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "target_label": target_label,
        "target_counts": target_counts,
        "fold_metrics": result.fold_metrics,
        "confusion_matrix": result.confusion_matrix,
        "duration_seconds": duration_s,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(),
    }
    path = artifact_dir / f"cv_metadata_{result.model_name}.json"
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


def run_models_on_run(
    features_csv: Path,
    artifact_dir: Path,
    models: list[str],
    *,
    feature_set: str = "trajectory_full",
    target_label: str = "auto",
    cv_folds: int = 5,
    random_state: int = 42,
    repo_root: Path | None = None,
) -> dict[str, dict]:
    """Run the requested models on one Stage-5 feature CSV.

    Returns a per-model summary dict (used by the orchestration script for
    its terminal summary). Heavy artifacts go to ``artifact_dir``.
    """
    import time

    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    df["target"] = _resolve_target(df, target_label).astype(float)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    feasible, used_folds, note = prediction_feasibility(df, cv_folds)
    if not feasible:
        logger.warning("Skipping prediction: %s", note)
        return {"_skipped": {"reason": note}}
    if note:
        logger.info(note)

    feature_sets = choose_feature_sets(df)
    if feature_set not in feature_sets:
        msg = f"Feature set '{feature_set}' not in {sorted(feature_sets)}"
        raise ValueError(msg)
    feature_cols = feature_sets[feature_set]
    if not feature_cols:
        msg = f"Feature set '{feature_set}' is empty for this run."
        raise ValueError(msg)

    X = df[feature_cols]
    y = df["target"].to_numpy()
    cv = StratifiedKFold(n_splits=used_folds, shuffle=True, random_state=random_state)

    git_sha = _git_sha(repo_root or Path.cwd())
    counts = target_stats(df)
    summary: dict[str, dict] = {}

    for name in models:
        spec = get_spec(name)
        logger.info("Fitting %s on %d features (%d items)", name, len(feature_cols), len(df))
        t0 = time.perf_counter()
        result = run_oof(spec, X, y, cv, random_state)
        duration = time.perf_counter() - t0

        oof_path = _save_oof(result, df, artifact_dir)
        imp_path = _save_importance(result, artifact_dir)
        meta_path = _save_metadata(
            result,
            feature_set_name=feature_set,
            cv_folds=used_folds,
            random_state=random_state,
            target_label=(target_label if target_label != "auto" else "auto-resolved"),
            target_counts=counts,
            duration_s=duration,
            git_sha=git_sha,
            artifact_dir=artifact_dir,
        )

        roc = result.fold_metrics["roc_auc"]
        summary[name] = {
            "auc_mean": roc["mean"],
            "auc_std": roc["std"],
            "duration_s": duration,
            "oof_path": str(oof_path),
            "importance_path": str(imp_path) if imp_path else None,
            "metadata_path": str(meta_path),
        }
    return summary
