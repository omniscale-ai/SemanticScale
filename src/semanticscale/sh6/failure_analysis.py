"""Failure-mode feature extraction and prediction for SH6 trajectories.

This module turns SH6 trace-level SLoD rankings into a tabular prediction
problem. The intended workflow is:

1. Merge Stage 1 trace records with Stage 2 SLoD rankings by ``id``.
2. Convert each reasoning / answer trajectory into interpretable summary
   features such as range, monotonicity, timing of peaks/troughs, and
   volatility.
3. Build three feature sets:
   ``length_only`` for trivial structural baselines,
   ``trajectory_shape`` for actual trajectory-derived signals, and
   ``trajectory_full`` for both together.
4. Run cross-validated logistic models and emit reports that are easy to
   inspect in research notes.

Two implementation details are worth calling out because they affect how
results should be interpreted:

- Trajectories are mean-centered before feature extraction. That keeps the
  analysis focused on within-trace shape rather than absolute Bradley-Terry
  offsets, which are only comparable within a single problem.
- Pair-density fields are retained in the CSV output as diagnostics, but they
  are excluded from the predictive models because they describe tournament
  coverage, not reasoning behavior.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .failure_models import get_spec, run_oof

logger = logging.getLogger(__name__)

EPS = 1e-8
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}

META_COLUMNS = {
    "id",
    "dataset",
    "run_slug",
    "subject",
    "generator",
    "model",
    "target",
    "target_label",
    "is_correct",
    "final_answer_correct",
    "rubric_score",
    "error_step_index",
    "error_step_position",
    "has_answer_chunks",
    "exit_status",
}


def _extract_rubric_score(item: dict) -> float | None:
    """Return rubric fractional credit in [0, 1] if available, else None.

    Prefers an already-populated ``rubric_score`` (written by grading.py for new
    runs). Falls back to deriving from ``item['grade']`` so already-graded data
    on disk works without re-running the grader.
    """
    explicit = item.get("rubric_score")
    if explicit is not None:
        return float(explicit)
    grade = item.get("grade")
    if not isinstance(grade, dict) or grade.get("type") != "rubric":
        return None
    total_max = grade.get("total_max") or 0.0
    if total_max <= 0:
        return None
    return float(grade.get("total_awarded", 0.0)) / float(total_max)

MEASUREMENT_COLUMNS = {
    "reasoning_pair_density",
    "answer_pair_density",
}

FEATURE_FAMILY_RULES = [
    ("direction_changes", "thrashing"),
    ("total_variation", "thrashing"),
    ("curvature_abs_mean", "thrashing"),
    ("zero_crossings", "thrashing"),
    ("late_mean", "landing"),
    ("end", "landing"),
    ("late_minus_early", "transition"),
    ("max_drop", "derailment"),
    ("fall_from_peak", "derailment"),
    ("trough_pos", "timing"),
    ("peak_pos", "timing"),
    ("max_drop_pos", "timing"),
    ("max_rise_pos", "timing"),
    ("monotonicity", "commitment"),
    ("range", "commitment"),
    ("n_chunks", "length"),
    ("answer_minus_reasoning", "answer_alignment"),
]

MODEL_DISPLAY_NAMES = {
    "length_only": "length_only (logreg)",
    "lenght_abort": "lenght_abort (logreg)",
    "trajectory_shape": "trajectory_shape (logreg)",
    "trajectory_full": "trajectory_full (logreg)",
    "mode_stack": "mode_stack (logreg)",
    "lightgbm_trajectory_full": "trajectory_full (lightgbm)",
    "lightgbm_mode_stack": "mode_stack (lightgbm)",
}


def merge_traces_and_rankings(traces: list[dict], rankings: list[dict]) -> list[dict]:
    """Merge Stage 1 and Stage 2 records by id."""
    rank_by_id = {record["id"]: record for record in rankings}
    merged = []
    missing = 0
    for trace in traces:
        rank = rank_by_id.get(trace["id"])
        if rank is None:
            missing += 1
            continue
        merged.append({**trace, **rank})
    if missing:
        logger.warning("Skipped %d trace(s) with no ranking record", missing)
    return merged


def _interpolate(values: np.ndarray, n: int) -> np.ndarray:
    """Resample a trajectory onto a fixed grid so lengths become comparable."""
    if len(values) == 0:
        return np.array([], dtype=float)
    if len(values) == 1:
        return np.repeat(values, n).astype(float)
    x_orig = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_orig, values)


def _mean_center(values: np.ndarray) -> np.ndarray:
    """Remove within-trace mean so features describe shape, not absolute level."""
    if len(values) == 0:
        return values.astype(float)
    return values.astype(float) - float(values.mean())


def _segment_means(values: np.ndarray) -> tuple[float, float, float]:
    """Summarise early / middle / late trajectory phases."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    parts = np.array_split(values, 3)
    return tuple(float(np.mean(part)) for part in parts)


def _signed_series(values: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Convert a series to {-1, 0, +1} signs with a small dead zone around zero."""
    signs = np.sign(values)
    signs[np.abs(values) <= tol] = 0
    return signs


def _direction_changes(values: np.ndarray) -> int:
    """Count slope sign flips after removing zero-slope intervals."""
    if len(values) == 0:
        return 0
    signs = _signed_series(values)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(nonzero[1:] != nonzero[:-1]))


def _zero_crossings(values: np.ndarray) -> int:
    """Count how often the trajectory crosses its own mean-centered zero line."""
    if len(values) < 2:
        return 0
    signs = _signed_series(values)
    crossings = 0
    prev = signs[0]
    for sign in signs[1:]:
        if sign == 0:
            continue
        if prev != 0 and sign != prev:
            crossings += 1
        prev = sign
    return crossings


def _normalised_index(index: int, total: int) -> float:
    """Map a step index onto [0, 1] so timing is length-invariant."""
    if total <= 1:
        return 0.0
    return float(index / (total - 1))


def _all_feature_names(prefix: str) -> list[str]:
    """Return the full schema for one trajectory family.

    This is mainly used so empty reasoning or answer traces still produce a
    rectangular feature table with explicit ``NaN`` placeholders.
    """
    return [
        f"{prefix}_start",
        f"{prefix}_end",
        f"{prefix}_end_minus_start",
        f"{prefix}_std",
        f"{prefix}_range",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_early_mean",
        f"{prefix}_mid_mean",
        f"{prefix}_late_mean",
        f"{prefix}_late_minus_early",
        f"{prefix}_total_variation",
        f"{prefix}_monotonicity",
        f"{prefix}_direction_changes",
        f"{prefix}_zero_crossings",
        f"{prefix}_peak_pos",
        f"{prefix}_trough_pos",
        f"{prefix}_rebound_from_trough",
        f"{prefix}_fall_from_peak",
        f"{prefix}_max_rise",
        f"{prefix}_max_drop",
        f"{prefix}_max_rise_pos",
        f"{prefix}_max_drop_pos",
        f"{prefix}_curvature_abs_mean",
        f"{prefix}_positive_mass",
        f"{prefix}_negative_mass",
        f"{prefix}_time_positive",
        f"{prefix}_time_negative",
    ]


def _trajectory_features(prefix: str, params: list[float], interp_points: int) -> dict:
    """Extract interpretable summary features from a trajectory.

    The features are designed to capture a few broad failure-mode families:

    - commitment / landing: where the trace starts and ends, and whether it
      settles cleanly
    - thrashing: oscillation, curvature, and total variation
    - timing: when major peaks, troughs, rises, and drops occur
    - derailment / recovery: how sharply the trace falls and whether it
      rebounds afterward
    """
    values = np.array(params, dtype=float)
    features = {f"{prefix}_n_chunks": int(len(values))}
    if len(values) == 0:
        for name in _all_feature_names(prefix):
            features[name] = np.nan
        return features

    centred = _mean_center(values)
    interp = _interpolate(centred, interp_points)
    diffs = np.diff(interp)
    second_diffs = np.diff(interp, n=2)
    early_mean, mid_mean, late_mean = _segment_means(interp)

    start = float(interp[0])
    end = float(interp[-1])
    max_value = float(np.max(interp))
    min_value = float(np.min(interp))
    peak_idx = int(np.argmax(interp))
    trough_idx = int(np.argmin(interp))
    max_rise_idx = int(np.argmax(diffs)) if len(diffs) else 0
    max_drop_idx = int(np.argmin(diffs)) if len(diffs) else 0
    total_variation = float(np.abs(diffs).sum())

    features.update(
        {
            f"{prefix}_start": start,
            f"{prefix}_end": end,
            f"{prefix}_end_minus_start": end - start,
            f"{prefix}_std": float(np.std(interp)),
            f"{prefix}_range": max_value - min_value,
            f"{prefix}_min": min_value,
            f"{prefix}_max": max_value,
            f"{prefix}_early_mean": early_mean,
            f"{prefix}_mid_mean": mid_mean,
            f"{prefix}_late_mean": late_mean,
            f"{prefix}_late_minus_early": late_mean - early_mean,
            f"{prefix}_total_variation": total_variation,
            f"{prefix}_monotonicity": float(abs(end - start) / max(total_variation, EPS)),
            f"{prefix}_direction_changes": _direction_changes(diffs),
            f"{prefix}_zero_crossings": _zero_crossings(interp),
            f"{prefix}_peak_pos": _normalised_index(peak_idx, len(interp)),
            f"{prefix}_trough_pos": _normalised_index(trough_idx, len(interp)),
            f"{prefix}_rebound_from_trough": end - min_value,
            f"{prefix}_fall_from_peak": max_value - end,
            f"{prefix}_max_rise": float(np.max(diffs)) if len(diffs) else 0.0,
            f"{prefix}_max_drop": float(max(0.0, -np.min(diffs))) if len(diffs) else 0.0,
            f"{prefix}_max_rise_pos": _normalised_index(max_rise_idx, len(diffs)) if len(diffs) else np.nan,
            f"{prefix}_max_drop_pos": _normalised_index(max_drop_idx, len(diffs)) if len(diffs) else np.nan,
            f"{prefix}_curvature_abs_mean": float(np.mean(np.abs(second_diffs))) if len(second_diffs) else 0.0,
            f"{prefix}_positive_mass": float(np.clip(interp, 0.0, None).mean()),
            f"{prefix}_negative_mass": float(np.clip(-interp, 0.0, None).mean()),
            f"{prefix}_time_positive": float(np.mean(interp > 0.0)),
            f"{prefix}_time_negative": float(np.mean(interp < 0.0)),
        }
    )
    return features


def _cross_features(reasoning_params: list[float], answer_params: list[float]) -> dict:
    """Compare reasoning and answer trajectories on the shared SLoD scale.

    These are only meaningful because Stage 2 ranks reasoning and answer
    chunks jointly for a single item, putting both traces onto one local
    Bradley-Terry scale.
    """
    names = [
        "answer_minus_reasoning_mean",
        "answer_start_minus_reasoning_end",
        "answer_end_minus_reasoning_end",
        "answer_range_minus_reasoning_range",
    ]
    if not reasoning_params or not answer_params:
        return {name: np.nan for name in names}

    reasoning = np.array(reasoning_params, dtype=float)
    answer = np.array(answer_params, dtype=float)
    return {
        "answer_minus_reasoning_mean": float(answer.mean() - reasoning.mean()),
        "answer_start_minus_reasoning_end": float(answer[0] - reasoning[-1]),
        "answer_end_minus_reasoning_end": float(answer[-1] - reasoning[-1]),
        "answer_range_minus_reasoning_range": float(
            (answer.max() - answer.min()) - (reasoning.max() - reasoning.min())
        ),
    }


def _comparison_density(
    comparisons: list[list[int]],
    n_chunks: int,
    ties: list[list[int]] | None = None,
) -> float:
    """Compute how much of the possible pairwise tournament was actually observed."""
    if n_chunks < 2:
        return np.nan
    possible = n_chunks * (n_chunks - 1) / 2
    seen_pairs = {
        frozenset(pair)
        for pair in (comparisons or [])
        if len(pair) == 2
    }
    seen_pairs.update(
        frozenset(pair)
        for pair in (ties or [])
        if len(pair) == 2
    )
    return float(len(seen_pairs) / max(possible, 1.0))


def _resolve_target_label(merged: list[dict], requested: str) -> str:
    """Choose the prediction target, preferring final-answer correctness when available."""
    if requested != "auto":
        available = any(item.get(requested) is not None for item in merged)
        if not available:
            raise ValueError(f"Requested target label '{requested}' is unavailable in this run")
        return requested
    for candidate in ("final_answer_correct", "is_correct"):
        if any(item.get(candidate) is not None for item in merged):
            return candidate
    raise ValueError("No usable target label found; expected final_answer_correct or is_correct")


def build_feature_table(
    merged: list[dict],
    target_label: str = "auto",
    interp_points: int = 20,
) -> tuple[pd.DataFrame, str]:
    """Convert merged trace records into a per-item feature table.

    Each row corresponds to one problem instance. Besides the derived
    trajectory features, the table preserves enough metadata to support
    follow-up slicing by subject, generator, dataset, or error-step
    annotations.
    """
    resolved_target = _resolve_target_label(merged, target_label)
    rows = []
    for item in merged:
        label_value = item.get(resolved_target)
        if label_value is None:
            continue

        reasoning_params = item.get("reasoning_params") or []
        answer_params = item.get("answer_params") or []
        if not reasoning_params and not answer_params:
            continue

        row = {
            "id": item.get("id"),
            "dataset": item.get("dataset"),
            "run_slug": item.get("run_slug"),
            "subject": item.get("subject"),
            "generator": item.get("generator"),
            "model": item.get("model"),
            "target": int(bool(label_value)),
            "target_label": resolved_target,
            "is_correct": item.get("is_correct"),
            "final_answer_correct": item.get("final_answer_correct"),
            "rubric_score": _extract_rubric_score(item),
            "has_answer_chunks": bool(answer_params),
            "error_step_index": item.get("error_step_index"),
            "error_step_position": np.nan,
            "exit_status": item.get("exit_status"),
        }

        if item.get("error_step_index") is not None and len(reasoning_params) > 1:
            row["error_step_position"] = float(
                item["error_step_index"] / max(len(reasoning_params) - 1, 1)
            )

        row.update(_trajectory_features("reasoning", reasoning_params, interp_points))
        row.update(_trajectory_features("answer", answer_params, interp_points))
        row.update(_cross_features(reasoning_params, answer_params))

        row["total_n_chunks"] = row["reasoning_n_chunks"] + row["answer_n_chunks"]
        row["reasoning_pair_density"] = _comparison_density(
            item.get("reasoning_comparisons") or [],
            row["reasoning_n_chunks"],
            item.get("reasoning_ties") or [],
        )
        row["answer_pair_density"] = _comparison_density(
            item.get("answer_comparisons") or [],
            row["answer_n_chunks"],
            item.get("answer_ties") or [],
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, resolved_target


def choose_feature_sets(
    df: pd.DataFrame,
    extra_meta_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    """Build model-ready feature sets.

    The split between ``length_only`` and ``trajectory_shape`` is deliberate:
    it lets us test whether trajectory-derived features carry signal beyond
    simple structural cues such as answer length or number of reasoning steps.

    ``extra_meta_columns`` lets the caller mark detector-score columns as
    meta — they should not leak into ``trajectory_shape`` / ``trajectory_full``
    because the caller will register them as their own ``mode_stack`` set.
    """
    extra_meta = set(extra_meta_columns or [])

    def dedupe(cols: list[str]) -> list[str]:
        unique: list[str] = []
        for col in cols:
            if any(df[col].equals(df[other]) for other in unique):
                continue
            unique.append(col)
        return unique

    def usable(cols: list[str]) -> list[str]:
        good = []
        for col in cols:
            if (
                col not in df.columns
                or col in META_COLUMNS
                or col in MEASUREMENT_COLUMNS
                or col in extra_meta
            ):
                continue
            series = df[col]
            nonnull = series.dropna()
            if nonnull.empty:
                continue
            if nonnull.nunique() < 2:
                continue
            good.append(col)
        return dedupe(good)

    length = usable(
        [
            "reasoning_n_chunks",
            "answer_n_chunks",
            "total_n_chunks",
        ]
    )
    shape_candidates = [
        col
        for col in df.columns
        if col not in META_COLUMNS and col not in extra_meta and col not in length
    ]
    shape = usable(shape_candidates)

    return {
        "length_only": length,
        "trajectory_shape": shape,
        "trajectory_full": usable(length + shape),
    }


def _make_estimator(random_state: int) -> Pipeline:
    """Standard predictive baseline used across all SH6 failure analyses."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def target_stats(df: pd.DataFrame) -> dict[str, int]:
    """Return class counts used for feasibility checks and report headers."""
    y = df["target"].to_numpy()
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return {
        "n_items": int(len(df)),
        "n_positive": positives,
        "n_negative": negatives,
    }


def prediction_feasibility(df: pd.DataFrame, requested_cv_folds: int) -> tuple[bool, int | None, str | None]:
    """Check whether cross-validated prediction is statistically possible.

    The helper prevents the script from silently producing meaningless metrics
    on one-class or tiny runs. When possible it also shrinks the requested
    number of folds to match the smallest class.
    """
    stats = target_stats(df)
    positives = stats["n_positive"]
    negatives = stats["n_negative"]
    n_items = stats["n_items"]

    if positives == 0 or negatives == 0:
        return (
            False,
            None,
            "Prediction is not identifiable because the run contains only one target class.",
        )

    min_class = min(positives, negatives)
    cv_folds = min(requested_cv_folds, min_class, n_items)
    if cv_folds < 2:
        return (
            False,
            None,
            "Prediction is not identifiable because there are too few examples per class for cross-validation.",
        )

    note = None
    if cv_folds != requested_cv_folds:
        note = f"Reduced cross-validation folds from {requested_cv_folds} to {cv_folds} due to class counts."
    return True, cv_folds, note


def _summarise_scores(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Collapse per-fold sklearn scores into mean/std report entries."""
    summary: dict[str, dict[str, float]] = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.removeprefix("test_")
        arr = np.array(values, dtype=float)
        summary[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
        }
    return summary


def _evaluate_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
    cv_folds: int,
) -> dict:
    """Evaluate one feature set and keep both fold metrics and OOF predictions."""
    y = df["target"].to_numpy()
    estimator = _make_estimator(random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    X = df[feature_cols]
    scores = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, n_jobs=None)
    probs = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)

    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    return {
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "metrics": _summarise_scores(scores),
        "oof_probabilities": probs,
        "oof_predictions": preds,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _evaluate_registered_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    random_state: int,
    cv_folds: int,
) -> dict:
    """Evaluate one registered non-incumbent model on a fixed feature set."""
    y = df["target"].to_numpy()
    X = df[feature_cols]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    spec = get_spec(model_name)
    result = run_oof(spec, X, y, cv, random_state)
    return {
        "n_features": result.n_features,
        "feature_cols": result.feature_cols,
        "metrics": result.fold_metrics,
        "oof_probabilities": result.probabilities,
        "oof_predictions": (result.probabilities >= 0.5).astype(int),
        "confusion_matrix": result.confusion_matrix,
        "model_name": result.model_name,
        "feature_importance": result.feature_importance,
    }


def evaluate_prediction_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    random_state: int,
    cv_folds: int,
    extra_models: dict[str, tuple[str, list[str]]] | None = None,
) -> dict[str, dict]:
    """Run the predictive baseline for every requested feature set."""
    results = {}
    for name, cols in feature_sets.items():
        if not cols:
            logger.warning("Skipping %s: no usable features", name)
            continue
        logger.info("Evaluating %s with %d feature(s)", name, len(cols))
        results[name] = _evaluate_feature_set(df, cols, random_state, cv_folds)

    for result_name, (model_name, cols) in (extra_models or {}).items():
        if not cols:
            logger.warning("Skipping %s: no usable features", result_name)
            continue
        logger.info(
            "Evaluating %s with model=%s and %d feature(s)",
            result_name,
            model_name,
            len(cols),
        )
        try:
            results[result_name] = _evaluate_registered_model(
                df,
                cols,
                model_name,
                random_state,
                cv_folds,
            )
        except ImportError as exc:
            logger.warning("Skipping %s: %s", result_name, exc)
    return results


def score_univariate_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
    cv_folds: int,
) -> list[dict]:
    """Rank individual features by standalone predictive signal.

    ``signal_auc`` is directional-agnostic: a feature that strongly predicts
    either success or failure should rank high even if the raw ROC-AUC is
    below 0.5.
    """
    y = df["target"].to_numpy()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows: list[dict] = []

    for feature in feature_cols:
        X = df[[feature]]
        series = X[feature].dropna()
        if series.empty or series.nunique() < 2:
            continue
        estimator = _make_estimator(random_state)
        scores = cross_validate(estimator, X, y, cv=cv, scoring={"roc_auc": "roc_auc"})
        roc_auc = float(np.mean(scores["test_roc_auc"]))
        rows.append(
            {
                "feature": feature,
                "roc_auc": roc_auc,
                "signal_auc": max(roc_auc, 1.0 - roc_auc),
                "direction": "higher -> correct" if roc_auc >= 0.5 else "higher -> wrong",
                "family": feature_family(feature),
            }
        )

    rows.sort(key=lambda row: row["signal_auc"], reverse=True)
    return rows


def fit_full_model_coefficients(
    df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
) -> list[dict]:
    """Fit one full logistic model for coefficient-level interpretation.

    These coefficients are descriptive rather than inferential. They are most
    useful for naming tentative failure-mode families after we have already
    established that the feature set predicts the target out of sample.
    """
    estimator = _make_estimator(random_state)
    estimator.fit(df[feature_cols], df["target"].to_numpy())
    coefs = estimator.named_steps["logreg"].coef_[0]

    rows = []
    for feature, coef in zip(feature_cols, coefs):
        rows.append(
            {
                "feature": feature,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "direction": "higher -> correct" if coef >= 0 else "higher -> wrong",
                "family": feature_family(feature),
            }
        )
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows


def feature_family(feature: str) -> str:
    """Map a raw feature name to a coarse failure-mode family for reports."""
    for needle, family in FEATURE_FAMILY_RULES:
        if needle in feature:
            return family
    return "shape"


def write_feature_table(df: pd.DataFrame, path: Path) -> None:
    """Persist the per-item feature matrix for follow-up analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved feature table to %s", path)


def _serialise_summary(obj: object) -> object:
    """Convert numpy-heavy structures into plain JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {key: _serialise_summary(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialise_summary(value) for value in obj]
    if isinstance(obj, pd.DataFrame):
        return [_serialise_summary(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _serialise_summary(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_summary_json(summary: dict, path: Path) -> None:
    """Write a machine-readable summary alongside the markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_serialise_summary(summary), f, indent=2)
    logger.info("Saved summary JSON to %s", path)


def plot_roc_curves(
    y_true: np.ndarray,
    model_results: dict[str, dict],
    out_path: Path,
) -> None:
    """Plot out-of-fold ROC curves for the available feature-set baselines."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, color in [
        ("length_only", "#8c8c8c"),
        ("lenght_abort", "#8c8c8c"),
        ("trajectory_shape", "#2166ac"),
        ("trajectory_full", "#b2182b"),
        ("lightgbm_trajectory_full", "#4d9221"),
        ("mode_stack", "#1b7837"),
        ("lightgbm_mode_stack", "#762a83"),
    ]:
        result = model_results.get(name)
        if result is None:
            continue
        probs = np.asarray(result["oof_probabilities"], dtype=float)
        fpr, tpr, _ = roc_curve(y_true, probs)
        score = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=color,
            label=f"{MODEL_DISPLAY_NAMES.get(name, name)} (AUC={score:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cross-validated ROC curves")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC figure to %s", out_path)


def plot_top_coefficients(coefficients: list[dict], out_path: Path, top_k: int = 12) -> None:
    """Visualise the strongest positive and negative coefficients from the full model."""
    top = coefficients[:top_k]
    if not top:
        logger.warning("No coefficient rows available; skipping coefficient plot")
        return

    labels = [row["feature"] for row in reversed(top)]
    values = [row["coefficient"] for row in reversed(top)]
    colors = ["#2166ac" if value >= 0 else "#b2182b" for value in values]

    fig_height = max(4, 0.4 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(range(len(top)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Standardised logistic coefficient")
    ax.set_title("Strongest multivariate trajectory features")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved coefficient figure to %s", out_path)


def plot_mode_detector_summary(mode_rows: list[dict], out_path: Path) -> None:
    """Render a one-panel bar chart of detector score AUC with 95% bootstrap CIs.

    Bars are colour-coded by verdict so a reader sees at a glance which
    detectors were confirmed, inverted, or inconclusive on this run. The
    dashed line at 0.5 marks the null; a CI error bar that does not cross it
    corresponds to a confirmed or inverted verdict.
    """
    from .failure_modes import (
        VERDICT_CONFIRMED,
        VERDICT_INCONCLUSIVE,
        VERDICT_INSUFFICIENT,
        VERDICT_INVERTED,
    )

    usable = [row for row in mode_rows if row.get("n_scored") and not row.get("reason_skipped")]
    if not usable:
        logger.warning("No scored detectors available; skipping mode-detector plot")
        return

    colour_map = {
        VERDICT_CONFIRMED: "#2166ac",
        VERDICT_INVERTED: "#b2182b",
        VERDICT_INCONCLUSIVE: "#bdbdbd",
        VERDICT_INSUFFICIENT: "#d9d9d9",
    }

    names = [row["mode"] for row in usable]
    y_pos = np.arange(len(names))
    aucs = np.array(
        [row["roc_auc"] if row["roc_auc"] == row["roc_auc"] else 0.5 for row in usable]
    )
    lo = np.array(
        [
            row["roc_auc_ci_lo"]
            if row["roc_auc_ci_lo"] == row["roc_auc_ci_lo"]
            else row["roc_auc"] if row["roc_auc"] == row["roc_auc"] else 0.5
            for row in usable
        ]
    )
    hi = np.array(
        [
            row["roc_auc_ci_hi"]
            if row["roc_auc_ci_hi"] == row["roc_auc_ci_hi"]
            else row["roc_auc"] if row["roc_auc"] == row["roc_auc"] else 0.5
            for row in usable
        ]
    )
    colors = [colour_map.get(row.get("verdict", VERDICT_INSUFFICIENT), "#d9d9d9") for row in usable]
    err_low = np.clip(aucs - lo, 0.0, None)
    err_high = np.clip(hi - aucs, 0.0, None)

    fig_height = max(3.5, 0.55 * len(usable) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(y_pos, aucs, color=colors, edgecolor="white")
    ax.errorbar(
        aucs,
        y_pos,
        xerr=[err_low, err_high],
        fmt="none",
        ecolor="#404040",
        elinewidth=1.2,
        capsize=3,
    )
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="null (AUC=0.5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Failure-score ROC-AUC with 95% bootstrap CI")
    ax.set_title("Falsifiable failure-mode detectors")
    ax.invert_yaxis()

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour_map[VERDICT_CONFIRMED], label="confirmed"),
        plt.Rectangle((0, 0), 1, 1, color=colour_map[VERDICT_INVERTED], label="inverted"),
        plt.Rectangle((0, 0), 1, 1, color=colour_map[VERDICT_INCONCLUSIVE], label="inconclusive"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mode-detector figure to %s", out_path)


def _metric_line(name: str, result: dict) -> str:
    """Format one markdown table row of cross-validated metrics."""
    metrics = result["metrics"]
    return (
        f"| {MODEL_DISPLAY_NAMES.get(name, name)} "
        f"| {result['n_features']} "
        f"| {metrics['roc_auc']['mean']:.3f} +/- {metrics['roc_auc']['std']:.3f} "
        f"| {metrics['average_precision']['mean']:.3f} +/- {metrics['average_precision']['std']:.3f} "
        f"| {metrics['balanced_accuracy']['mean']:.3f} +/- {metrics['balanced_accuracy']['std']:.3f} "
        f"| {metrics['accuracy']['mean']:.3f} +/- {metrics['accuracy']['std']:.3f} "
        f"| {metrics['f1']['mean']:.3f} +/- {metrics['f1']['std']:.3f} |"
    )


def compute_mode_coverage(
    df: pd.DataFrame,
    mode_scores: pd.DataFrame,
    mode_rows: list[dict] | None,
) -> dict:
    """Coverage = fraction of failures flagged by at least one detector.

    Two variants are reported:

    - ``any``: union of every detector's flag, ignoring whether the directional
      claim was confirmed on this run.
    - ``confirmed``: union restricted to detectors whose verdict was
      ``confirmed`` — the principled coverage number, since flags from
      inverted/inconclusive detectors carry no validated meaning.

    Each variant reports recall (the fraction of failures covered),
    precision (of items flagged, what fraction are real failures), and the
    raw counts.
    """
    if "target" not in df.columns:
        raise ValueError("df must contain a 'target' column")

    fail = (df["target"].to_numpy() == 0)
    n_fail = int(fail.sum())
    if n_fail == 0:
        return {"any": None, "confirmed": None, "n_failures": 0}

    flag_cols = [c for c in mode_scores.columns if c.endswith("_flag")]
    confirmed_modes = {
        row["mode"] for row in (mode_rows or []) if row.get("verdict") == "confirmed"
    }
    confirmed_flag_cols = [f"{m}_flag" for m in confirmed_modes if f"{m}_flag" in mode_scores.columns]

    def _coverage(cols: list[str]) -> dict | None:
        if not cols:
            return None
        matrix = mode_scores[cols].fillna(0).to_numpy().astype(bool)
        any_flag = matrix.any(axis=1)
        n_flagged = int(any_flag.sum())
        n_caught = int(np.logical_and(any_flag, fail).sum())
        return {
            "n_modes": len(cols),
            "modes": [c.removesuffix("_flag") for c in cols],
            "n_flagged": n_flagged,
            "n_caught": n_caught,
            "n_failures": n_fail,
            "recall": float(n_caught / n_fail) if n_fail else float("nan"),
            "precision": float(n_caught / n_flagged) if n_flagged else float("nan"),
        }

    return {
        "any": _coverage(flag_cols),
        "confirmed": _coverage(confirmed_flag_cols),
        "n_failures": n_fail,
    }


def compute_capture_ratio(model_results: dict[str, dict]) -> dict | None:
    """Compare the mode-stack LR's AUC against the full-feature LR.

    Returns the raw ROC-AUCs for ``mode_stack`` and ``trajectory_full`` plus
    the captured-fraction ``(AUC_modes - 0.5) / (AUC_full - 0.5)``. Useful
    interpretation: 1.0 means the named modes carry as much above-chance
    discrimination as the full feature set; 0.5 means they capture roughly
    half. Returns ``None`` when either model is missing.

    Caveat: on FrontierScience the captured-fraction is **inflated** because
    the answer-side detectors were selected for predictive power on FS. The
    SWE-agent number is unbiased because the reasoning-side detectors were
    pre-registered. See ``failure_modes`` module docstring for the full
    methodological note.
    """
    modes_result = model_results.get("mode_stack")
    full_result = model_results.get("trajectory_full") or model_results.get("trajectory_shape")
    if modes_result is None or full_result is None:
        return None
    auc_modes = float(modes_result["metrics"]["roc_auc"]["mean"])
    auc_full = float(full_result["metrics"]["roc_auc"]["mean"])
    above_chance_full = auc_full - 0.5
    if above_chance_full <= 0:
        captured = float("nan")
    else:
        captured = float((auc_modes - 0.5) / above_chance_full)
    return {
        "mode_stack_auc": auc_modes,
        "full_auc": auc_full,
        "captured_fraction": captured,
        "comparison_set": "trajectory_full"
        if "trajectory_full" in model_results
        else "trajectory_shape",
    }


def _label_alignment(df: pd.DataFrame) -> dict | None:
    """Compare reasoning-cleanliness labels against final-answer correctness when both exist."""
    if "is_correct" not in df.columns or "final_answer_correct" not in df.columns:
        return None
    if df["is_correct"].isna().all() or df["final_answer_correct"].isna().all():
        return None

    clean = df.dropna(subset=["is_correct", "final_answer_correct"]).copy()
    if clean.empty:
        return None

    reasoning_clean = clean["is_correct"].astype(int)
    final_clean = clean["final_answer_correct"].astype(int)
    agreement = float(np.mean(reasoning_clean == final_clean))
    return {
        "n_items": int(len(clean)),
        "agreement": agreement,
        "final_correct_but_reasoning_wrong": int(np.sum((final_clean == 1) & (reasoning_clean == 0))),
        "final_wrong_but_reasoning_clean": int(np.sum((final_clean == 0) & (reasoning_clean == 1))),
    }


def write_markdown_report(
    df: pd.DataFrame,
    target_label: str,
    model_results: dict[str, dict],
    univariate_rows: list[dict],
    coefficient_rows: list[dict],
    out_path: Path,
    dataset_name: str,
    run_slug: str,
    analysis_note: str | None = None,
    mode_rows: list[dict] | None = None,
    coverage: dict | None = None,
    capture: dict | None = None,
) -> None:
    """Write the human-readable experiment report.

    The markdown report is meant to be the first artifact a researcher opens.
    It includes enough context to interpret the metrics without re-reading the
    code or the raw feature CSV.
    """
    stats = target_stats(df)
    label_alignment = _label_alignment(df)

    lines = [
        f"# SH6 {dataset_name}/{run_slug} — Failure Prediction",
        "",
        "## Setup",
        "",
        f"- Target label: `{target_label}`",
        f"- Items analysed: {stats['n_items']}",
        f"- Positive class (`{target_label}=true`): {stats['n_positive']}",
        f"- Negative class (`{target_label}=false`): {stats['n_negative']}",
    ]

    if label_alignment:
        lines.extend(
            [
                f"- Label agreement (`is_correct` vs `final_answer_correct`): {100 * label_alignment['agreement']:.1f}% "
                f"over {label_alignment['n_items']} items",
                f"- Final answer correct but reasoning wrong: {label_alignment['final_correct_but_reasoning_wrong']}",
                f"- Final answer wrong but reasoning clean: {label_alignment['final_wrong_but_reasoning_clean']}",
            ]
        )

    if analysis_note:
        lines.extend(["", "## Status", "", f"- {analysis_note}"])

    lines.extend(
        [
            "",
            "## Cross-Validated Prediction",
            "",
            "| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    wrote_model_row = False
    for name in (
        "length_only",
        "lenght_abort",
        "trajectory_shape",
        "trajectory_full",
        "lightgbm_trajectory_full",
        "mode_stack",
        "lightgbm_mode_stack",
    ):
        result = model_results.get(name)
        if result is not None:
            lines.append(_metric_line(name, result))
            wrote_model_row = True
    if not wrote_model_row:
        lines.append("| Not run | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Top Single Features",
            "",
            "| Feature | Family | Signal ROC-AUC | Direction |",
            "|---|---|---|---|",
        ]
    )
    for row in univariate_rows[:12]:
        lines.append(
            f"| {row['feature']} | {row['family']} | {row['signal_auc']:.3f} | {row['direction']} |"
        )
    if not univariate_rows:
        lines.append("| Not available | - | - | - |")

    lines.extend(
        [
            "",
            "## Strongest Multivariate Coefficients",
            "",
            "| Feature | Family | Coefficient | Direction |",
            "|---|---|---|---|",
        ]
    )
    for row in coefficient_rows[:12]:
        lines.append(
            f"| {row['feature']} | {row['family']} | {row['coefficient']:.3f} | {row['direction']} |"
        )
    if not coefficient_rows:
        lines.append("| Not available | - | - | - |")

    if mode_rows:
        from .failure_modes import (
            VERDICT_CONFIRMED,
            VERDICT_INCONCLUSIVE,
            VERDICT_INVERTED,
            render_mode_descriptions_markdown,
            render_mode_table_markdown,
        )

        lines.extend(
            [
                "",
                "## Interpretable Failure-Mode Detectors",
                "",
                "Each detector encodes a pre-registered hypothesis: *higher detector score implies a higher probability of failure*.",
                "For every run we compute a 95% percentile-bootstrap CI on the score's failure-AUC and assign one of four verdicts:",
                "",
                "- `confirmed` — CI lower bound above 0.5; the directional claim holds.",
                "- `inverted` — CI upper bound below 0.5; on this run the score actually predicts *success*. The hypothesis is falsified in the opposite direction.",
                "- `inconclusive` — CI straddles 0.5; there is no evidence either way on this run.",
                "- `insufficient_data` — too few scored rows or only one class present.",
                "",
                "Flag-level metrics (precision / recall / F1 / lift) are reported only when the verdict is `confirmed`. When the hypothesis is falsified or unclear, those numbers would be actively misleading, so they render as `-`. The raw AUC and CI are always reported so the call can be audited.",
                "",
                "### What each detector catches",
                "",
            ]
        )
        lines.extend(render_mode_descriptions_markdown(mode_rows))
        lines.extend(
            [
                "",
                "### Detector performance",
                "",
            ]
        )
        lines.extend(render_mode_table_markdown(mode_rows))

        inverted = [row["mode"] for row in mode_rows if row.get("verdict") == VERDICT_INVERTED]
        inconclusive = [
            row["mode"] for row in mode_rows if row.get("verdict") == VERDICT_INCONCLUSIVE
        ]
        confirmed = [row["mode"] for row in mode_rows if row.get("verdict") == VERDICT_CONFIRMED]
        callouts: list[str] = []
        if confirmed:
            callouts.append(f"- **Confirmed on this run**: {', '.join(confirmed)}.")
        if inverted:
            callouts.append(
                f"- **Hypothesis falsified (inverted)**: {', '.join(inverted)}. "
                "The score direction flipped — higher values predict success, not failure, on this dataset. "
                "This is a real negative result, not a detector failure."
            )
        if inconclusive:
            callouts.append(
                f"- **Inconclusive**: {', '.join(inconclusive)}. "
                "The bootstrap CI spans 0.5, so we cannot reject the null on this run."
            )
        if callouts:
            lines.append("")
            lines.append("### Verdict summary")
            lines.append("")
            lines.extend(callouts)

        if capture:
            lines.append("")
            lines.append("### Signal capture")
            lines.append("")
            lines.append(
                "How much of the failure-prediction signal does the named-mode "
                "taxonomy actually carry? The `mode_stack` model is a logistic "
                "regression on the detector scores only; the comparison set is the "
                f"`{capture['comparison_set']}` model fit on all trajectory features."
            )
            lines.append("")
            lines.append(
                f"- `mode_stack` ROC-AUC: **{capture['mode_stack_auc']:.3f}**"
            )
            lines.append(
                f"- `{capture['comparison_set']}` ROC-AUC: **{capture['full_auc']:.3f}**"
            )
            captured = capture["captured_fraction"]
            captured_txt = (
                f"{captured:.1%}"
                if captured == captured and not np.isinf(captured)
                else "n/a"
            )
            lines.append(
                f"- Above-chance discrimination preserved by the mode stack: "
                f"**{captured_txt}** "
                f"((AUC_modes − 0.5) ÷ (AUC_full − 0.5))"
            )
            if dataset_name == "frontierscience":
                lines.append("")
                lines.append(
                    "> Caveat: the FrontierScience capture number is **inflated** "
                    "because the answer-side detectors (`answer_meandering`, "
                    "`answer_volatility`, `answer_uncommitted`, `answer_overrange`) "
                    "were selected post-hoc by ranking univariate AUCs on this "
                    "dataset. Treat this number as a descriptive upper bound. "
                    "The SWE-agent capture number, where the reasoning-side "
                    "detectors were pre-registered, is the unbiased estimate."
                )

        if coverage and coverage.get("n_failures"):
            lines.append("")
            lines.append("### Failure coverage")
            lines.append("")
            lines.append(
                "What fraction of failures get flagged by at least one detector? "
                "`any` uses the union of every detector's flag; `confirmed` "
                "restricts to detectors whose directional hypothesis was confirmed "
                "on this run, which is the principled coverage number."
            )
            lines.append("")
            lines.append(
                "| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |"
            )
            lines.append("|---|---|---|---|---|---|")
            for variant_name in ("confirmed", "any"):
                row = coverage.get(variant_name)
                if not row:
                    lines.append(
                        f"| {variant_name} | 0 | 0 / {coverage.get('n_failures', 0)} | 0 | - | - |"
                    )
                    continue
                lines.append(
                    f"| {variant_name} | {row['n_modes']} "
                    f"| {row['n_caught']} / {row['n_failures']} "
                    f"| {row['n_flagged']} "
                    f"| {row['recall']:.3f} "
                    f"| {row['precision']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.",
            "- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.",
            "- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.",
            "- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.",
            "- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved markdown report to %s", out_path)


def build_summary_payload(
    df: pd.DataFrame,
    dataset_name: str,
    run_slug: str,
    target_label: str,
    model_results: dict[str, dict],
    univariate_rows: list[dict],
    coefficient_rows: list[dict],
    analysis_note: str | None = None,
    mode_rows: list[dict] | None = None,
    coverage: dict | None = None,
    capture: dict | None = None,
) -> dict:
    """Build the machine-readable summary that backs the markdown report."""
    stats = target_stats(df)
    return {
        "dataset": dataset_name,
        "run_slug": run_slug,
        "target_label": target_label,
        "n_items": stats["n_items"],
        "n_positive": stats["n_positive"],
        "n_negative": stats["n_negative"],
        "analysis_note": analysis_note,
        "label_alignment": _label_alignment(df),
        "model_results": model_results,
        "top_univariate_features": univariate_rows[:20],
        "top_coefficients": coefficient_rows[:20],
        "mode_detectors": mode_rows or [],
        "mode_coverage": coverage,
        "mode_stack_capture": capture,
    }
