"""AgentHallu SLoD-only judgment and attribution baseline.

This module evaluates a non-LLM baseline that uses SH6 SLoD trajectories only:

* Judgment: cross-validated trajectory-level classification from the existing
  SH6 trajectory feature table.
* Attribution: cross-validated step-level classification over per-step SLoD
  features, selecting the highest-scoring step in each hallucinated trace as the
  predicted responsible step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from semanticscale.sh6 import datasets as ds
from semanticscale.sh6.failure_analysis import (
    build_feature_table,
    choose_feature_sets,
    merge_traces_and_rankings,
)
from semanticscale.sh6.failure_models import default_cv, get_spec, run_oof
from semanticscale.utils import load_jsonl

logger = logging.getLogger(__name__)


def _cfg(config: dict) -> dict:
    return config.get("agenthallu_slod_prediction") or {}


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _safe_slug(config: dict, run_slug: str | None) -> tuple[str, Path, Path]:
    project_root = Path(config["_project_root"])
    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    requested = run_slug or ds.run_slug(config)
    resolved_slug, run_dir = ds.resolve_run_dir(
        config,
        data_dir,
        requested,
        required_files=("traces.jsonl", "chunk_rankings.jsonl"),
    )
    reports_root = (project_root / config["paths"]["reports_dir"]).resolve()
    return resolved_slug, run_dir, reports_root / ds.dataset_name(config) / resolved_slug


def load_agenthallu_run(
    config: dict,
    *,
    run_slug: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame, str, Path]:
    """Load one AgentHallu run and derive the trajectory feature table."""
    resolved_slug, run_dir, reports_dir = _safe_slug(config, run_slug)
    traces = load_jsonl(run_dir / "traces.jsonl")
    rankings = load_jsonl(run_dir / "chunk_rankings.jsonl")
    merged = merge_traces_and_rankings(traces, rankings)
    feature_df, _ = build_feature_table(
        merged,
        target_label="final_answer_correct",
        interp_points=int(config.get("failure_analysis", {}).get("interp_points", 20)),
    )
    feature_df["target"] = 1 - feature_df["target"].astype(int)
    feature_df["target_label"] = "is_hallucination"
    return traces, rankings, merged, feature_df, resolved_slug, reports_dir


def evaluate_judgment(
    feature_df: pd.DataFrame,
    *,
    feature_set: str,
    model_name: str,
    cv_folds: int,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    """OOF trajectory-level hallucination judgment from SLoD features."""
    feature_sets = choose_feature_sets(feature_df)
    if feature_set not in feature_sets:
        raise ValueError(
            f"Unknown feature_set={feature_set!r}. Available: {sorted(feature_sets)}"
        )
    cols = feature_sets[feature_set]
    X = feature_df[cols]
    y = feature_df["target"].astype(int).to_numpy()
    cv = default_cv("classification", cv_folds, random_state)
    spec = get_spec(model_name)
    result = run_oof(spec, X, y, cv, random_state)
    probs = result.predictions.astype(float)
    hard = (probs >= 0.5).astype(int)

    oof = pd.DataFrame(
        {
            "id": feature_df["id"].to_numpy(),
            "row_index": np.arange(len(feature_df)),
            "fold": result.fold_assignments,
            "target": y,
            "prob_hallucination": probs,
            "pred_hallucination": hard,
            "subject": feature_df["subject"].astype(str).to_numpy(),
            "hallucination_category": feature_df.get(
                "hallucination_category",
                pd.Series(["Unknown Type"] * len(feature_df)),
            ).astype(str).to_numpy(),
        }
    )

    overall = _metric_block(y, hard)
    by_subject = {}
    for subject, sub in oof.groupby("subject", sort=True):
        by_subject[str(subject)] = {
            **_metric_block(
                sub["target"].to_numpy(dtype=int),
                sub["pred_hallucination"].to_numpy(dtype=int),
            ),
            "n_items": int(len(sub)),
            "n_hallucinated": int(sub["target"].sum()),
        }

    return (
        {
            "model": model_name,
            "feature_set": feature_set,
            "n_items": int(len(feature_df)),
            "n_hallucinated": int(y.sum()),
            "n_clean": int(len(y) - y.sum()),
            "metrics": overall,
            "by_subject": by_subject,
            "confusion_matrix": result.confusion_matrix,
        },
        oof,
    )


def _step_rows(
    merged: list[dict],
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_lookup = feature_df.set_index("id")
    rows: list[dict] = []
    global_cols = [
        "reasoning_range",
        "reasoning_monotonicity",
        "reasoning_total_variation",
        "reasoning_direction_changes",
        "reasoning_curvature_abs_mean",
        "reasoning_n_chunks",
        "reasoning_peak_pos",
        "reasoning_trough_pos",
    ]

    for item in merged:
        params = [float(v) for v in (item.get("reasoning_params") or [])]
        error_step_index = item.get("error_step_index")
        if error_step_index is None or not params:
            continue
        if error_step_index >= len(params):
            error_step_index = len(params) - 1

        values = np.asarray(params, dtype=float)
        mean_value = float(values.mean())
        total_steps = len(values)
        feature_row = feature_lookup.loc[item["id"]] if item["id"] in feature_lookup.index else None

        running_peak = np.maximum.accumulate(values)
        running_trough = np.minimum.accumulate(values)

        for idx, value in enumerate(values):
            prev_value = values[idx - 1] if idx > 0 else value
            next_value = values[idx + 1] if idx + 1 < total_steps else value
            delta_prev = float(value - prev_value)
            delta_next = float(next_value - value)
            rows.append(
                {
                    "id": item["id"],
                    "subject": str(item.get("subject") or "unknown"),
                    "hallucination_category": str(
                        item.get("hallucination_category") or "Unknown Type"
                    ),
                    "true_step_index": int(error_step_index),
                    "step_index": idx,
                    "target": int(idx == error_step_index),
                    "n_steps": total_steps,
                    "step_pos": float(idx / max(total_steps - 1, 1)),
                    "remaining_steps": float(total_steps - idx - 1),
                    "slod_value": float(value),
                    "slod_centered": float(value - mean_value),
                    "slod_from_start": float(value - values[0]),
                    "slod_to_end": float(values[-1] - value),
                    "delta_prev": delta_prev,
                    "delta_next": delta_next,
                    "abs_delta_prev": abs(delta_prev),
                    "abs_delta_next": abs(delta_next),
                    "curvature": float(delta_next - delta_prev),
                    "fall_from_peak": float(running_peak[idx] - value),
                    "rebound_from_trough": float(value - running_trough[idx]),
                    "is_peak_so_far": float(value >= running_peak[idx] - 1e-8),
                    "is_trough_so_far": float(value <= running_trough[idx] + 1e-8),
                }
            )
            if feature_row is not None:
                for col in global_cols:
                    rows[-1][col] = feature_row.get(col, np.nan)

    return pd.DataFrame(rows)


def _step_estimator(random_state: int) -> Pipeline:
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


def evaluate_attribution(
    merged: list[dict],
    feature_df: pd.DataFrame,
    *,
    cv_folds: int,
    random_state: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """OOF responsible-step attribution from per-step SLoD features."""
    step_df = _step_rows(merged, feature_df)
    if step_df.empty:
        raise ValueError("No hallucinated traces with usable reasoning_params were found.")

    feature_cols = [
        col
        for col in step_df.columns
        if col
        not in {
            "id",
            "subject",
            "hallucination_category",
            "true_step_index",
            "step_index",
            "target",
        }
    ]
    X = step_df[feature_cols]
    y = step_df["target"].astype(int).to_numpy()
    groups = step_df["id"].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(cv_folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two hallucinated traces are required for attribution CV.")

    gkf = GroupKFold(n_splits=n_splits)
    estimator = _step_estimator(random_state)

    probs = np.zeros(len(step_df), dtype=float)
    folds = np.full(len(step_df), -1, dtype=int)
    trace_rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        estimator.fit(X.iloc[train_idx], y[train_idx])
        fold_probs = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        probs[test_idx] = fold_probs
        folds[test_idx] = fold_idx

        scored = step_df.iloc[test_idx].copy()
        scored["prob_responsible"] = fold_probs
        for trace_id, sub in scored.groupby("id", sort=False):
            best = sub.sort_values(
                ["prob_responsible", "step_index"],
                ascending=[False, True],
            ).iloc[0]
            trace_rows.append(
                {
                    "id": trace_id,
                    "fold": fold_idx,
                    "subject": best["subject"],
                    "hallucination_category": best["hallucination_category"],
                    "true_step_index": int(best["true_step_index"]),
                    "predicted_step_index": int(best["step_index"]),
                    "predicted_step_probability": float(best["prob_responsible"]),
                    "n_steps": int(best["n_steps"]),
                    "correct": int(best["true_step_index"] == best["step_index"]),
                }
            )

    step_oof = step_df.copy()
    step_oof["fold"] = folds
    step_oof["prob_responsible"] = probs

    trace_oof = pd.DataFrame(trace_rows).sort_values("id").reset_index(drop=True)
    overall_accuracy = float(trace_oof["correct"].mean()) if not trace_oof.empty else 0.0

    by_category = {}
    for category, sub in trace_oof.groupby("hallucination_category", sort=True):
        by_category[str(category)] = {
            "step_localization_accuracy": float(sub["correct"].mean()),
            "n_hallucinated": int(len(sub)),
        }

    by_subject = {}
    for subject, sub in trace_oof.groupby("subject", sort=True):
        by_subject[str(subject)] = {
            "step_localization_accuracy": float(sub["correct"].mean()),
            "n_hallucinated": int(len(sub)),
        }

    return (
        {
            "model": "logreg",
            "feature_cols": feature_cols,
            "n_hallucinated": int(trace_oof.shape[0]),
            "n_step_rows": int(step_oof.shape[0]),
            "cv_folds": int(n_splits),
            "step_localization_accuracy": overall_accuracy,
            "by_hallucination_category": by_category,
            "by_subject": by_subject,
        },
        trace_oof,
        step_oof,
    )


def evaluate_run(
    config: dict,
    *,
    run_slug: str | None = None,
) -> tuple[dict, dict[str, pd.DataFrame], Path]:
    """Run the full SLoD-only AgentHallu baseline for one SH6 run."""
    cfg = _cfg(config)
    _, _, merged, feature_df, resolved_slug, reports_dir = load_agenthallu_run(
        config,
        run_slug=run_slug,
    )

    random_state = int(cfg.get("random_state", 42))
    cv_folds = int(cfg.get("cv_folds", 5))
    judgment_model = str(cfg.get("judgment_model", "logreg"))
    judgment_feature_set = str(cfg.get("judgment_feature_set", "trajectory_full"))

    judgment_summary, judgment_oof = evaluate_judgment(
        feature_df,
        feature_set=judgment_feature_set,
        model_name=judgment_model,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    attribution_summary, attribution_trace_oof, attribution_step_oof = evaluate_attribution(
        merged,
        feature_df,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    summary = {
        "dataset": ds.dataset_name(config),
        "run_slug": resolved_slug,
        "judgment": judgment_summary,
        "attribution": attribution_summary,
    }
    outputs = {
        "judgment_oof": judgment_oof,
        "attribution_trace_oof": attribution_trace_oof,
        "attribution_step_oof": attribution_step_oof,
    }
    return summary, outputs, reports_dir


def write_outputs(
    summary: dict,
    outputs: dict[str, pd.DataFrame],
    reports_dir: Path,
) -> dict[str, Path]:
    """Persist JSON/Markdown summaries plus OOF artifacts."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "slod_prediction_summary.json"
    md_path = reports_dir / "slod_prediction.md"
    judgment_oof_path = reports_dir / "slod_judgment_oof.csv"
    attribution_oof_path = reports_dir / "slod_attribution_oof.csv"
    attribution_step_path = reports_dir / "slod_attribution_step_scores.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["judgment_oof"].to_csv(judgment_oof_path, index=False)
    outputs["attribution_trace_oof"].to_csv(attribution_oof_path, index=False)
    outputs["attribution_step_oof"].to_csv(attribution_step_path, index=False)

    judgment = summary["judgment"]
    attribution = summary["attribution"]
    lines = [
        "# AgentHallu SLoD Trajectory Baseline",
        "",
        f"**Run:** `{summary['run_slug']}`",
        "",
        "## Judgment",
        "",
        "| Model | Feature set | Macro-F1 | Macro-recall | Accuracy | n hallucinated | n clean |",
        "|---|---|---|---|---|---|---|",
        (
            f"| {judgment['model']} | {judgment['feature_set']} "
            f"| {100 * judgment['metrics']['macro_f1']:.1f}% "
            f"| {100 * judgment['metrics']['macro_recall']:.1f}% "
            f"| {100 * judgment['metrics']['accuracy']:.1f}% "
            f"| {judgment['n_hallucinated']} | {judgment['n_clean']} |"
        ),
        "",
        "## Attribution",
        "",
        "| Model | Step localization accuracy | Hallucinated traces | Step rows | CV folds |",
        "|---|---|---|---|---|",
        (
            f"| {attribution['model']} "
            f"| {100 * attribution['step_localization_accuracy']:.1f}% "
            f"| {attribution['n_hallucinated']} "
            f"| {attribution['n_step_rows']} "
            f"| {attribution['cv_folds']} |"
        ),
        "",
        "## Attribution by Hallucination Category",
        "",
        "| Category | Step localization accuracy | Hallucinated traces |",
        "|---|---|---|",
    ]
    for category, metrics in attribution["by_hallucination_category"].items():
        lines.append(
            f"| {category} | {100 * metrics['step_localization_accuracy']:.1f}% | {metrics['n_hallucinated']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_json": summary_path,
        "summary_md": md_path,
        "judgment_oof_csv": judgment_oof_path,
        "attribution_oof_csv": attribution_oof_path,
        "attribution_step_scores_csv": attribution_step_path,
    }
