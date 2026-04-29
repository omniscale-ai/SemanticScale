"""Dataset-agnostic SLoD-only failure attribution for SH6.

For one SH6 run this module evaluates two SLoD-only baselines:

* **Failure location** — per-step classifier picks the trace step responsible
  for the failure, scored against ``error_step_index``. Runs whenever any
  trace in the run has ``error_step_index`` populated (ProcessBench,
  AgentHallu, AgentErrorBench all qualify).
* **Failure type** — trajectory-level multi-class classifier predicts the
  categorical failure type (``hallucination_category`` for AgentHallu,
  ``critical_error_type`` for AgentErrorBench). Skipped when the dataset has
  no type label.

Inputs come from the existing SH6 artifacts ``traces.jsonl`` and
``chunk_rankings.jsonl``; no external LLM is called.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from . import datasets as ds
from .failure_analysis import (
    build_feature_table,
    build_step_feature_table,
    choose_feature_sets,
    classification_metrics_block,
    merge_traces_and_rankings,
    metrics_by_group,
    resolve_run_paths,
)
from .failure_models import get_spec
from ..utils import load_jsonl

logger = logging.getLogger(__name__)


_TYPE_LABEL_BY_DATASET: dict[str, str] = {
    "agenthallu": "hallucination_category",
    "agenterrorbench": "critical_error_type",
}


def _cfg(config: dict) -> dict:
    return config.get("failure_attribution") or {}


def _resolve_type_label_field(dataset_name: str) -> str | None:
    return _TYPE_LABEL_BY_DATASET.get(dataset_name)


def load_run(
    config: dict,
    *,
    run_slug: str | None = None,
) -> tuple[list[dict], pd.DataFrame, str, str, Path]:
    """Load merged traces + trajectory feature table for one SH6 run.

    Returns ``(merged, feature_df, dataset_name, resolved_slug, run_reports_dir)``.
    """
    paths = resolve_run_paths(
        config,
        run_slug=run_slug,
        required_files=("traces.jsonl", "chunk_rankings.jsonl"),
    )
    traces = load_jsonl(paths.run_dir / "traces.jsonl")
    rankings = load_jsonl(paths.run_dir / "chunk_rankings.jsonl")
    merged = merge_traces_and_rankings(traces, rankings)

    interp_points = int(
        config.get("failure_analysis", {}).get("interp_points", 20)
    )
    feature_df, _ = build_feature_table(
        merged,
        target_label="auto",
        interp_points=interp_points,
    )
    return merged, feature_df, paths.dataset_name, paths.run_slug, paths.run_reports_dir


def evaluate_location(
    merged: list[dict],
    feature_df: pd.DataFrame,
    *,
    cv_folds: int,
    random_state: int,
    type_field: str | None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """OOF responsible-step localization from per-step SLoD features."""
    step_df = build_step_feature_table(merged, feature_df, type_field=type_field)
    if step_df.empty:
        return (
            {
                "skipped": True,
                "reason": "No traces with both error_step_index and reasoning_params.",
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    feature_cols = [
        col
        for col in step_df.columns
        if col
        not in {
            "id",
            "subject",
            "failure_type",
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
        return (
            {
                "skipped": True,
                "reason": (
                    f"Need at least 2 trace groups for CV; got {len(unique_groups)}."
                ),
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    gkf = GroupKFold(n_splits=n_splits)
    estimator = get_spec("logreg").build_estimator(random_state)

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
                    "failure_type": best["failure_type"],
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
    overall_accuracy = (
        float(trace_oof["correct"].mean()) if not trace_oof.empty else 0.0
    )

    by_failure_type = {
        str(value): {
            "step_localization_accuracy": float(sub["correct"].mean()),
            "n_failed": int(len(sub)),
        }
        for value, sub in trace_oof.groupby("failure_type", sort=True)
    }
    by_subject = {
        str(value): {
            "step_localization_accuracy": float(sub["correct"].mean()),
            "n_failed": int(len(sub)),
        }
        for value, sub in trace_oof.groupby("subject", sort=True)
    }

    summary = {
        "model": "logreg",
        "feature_cols": feature_cols,
        "n_failed": int(trace_oof.shape[0]),
        "n_step_rows": int(step_oof.shape[0]),
        "cv_folds": int(n_splits),
        "step_localization_accuracy": overall_accuracy,
        "by_failure_type": by_failure_type,
        "by_subject": by_subject,
    }
    return summary, trace_oof, step_oof


def evaluate_type(
    merged: list[dict],
    feature_df: pd.DataFrame,
    *,
    cv_folds: int,
    random_state: int,
    type_field: str,
    min_class_count: int,
) -> tuple[dict, pd.DataFrame]:
    """OOF multi-class classifier for the categorical failure type."""
    label_by_id: dict[str, str] = {}
    for item in merged:
        value = item.get(type_field)
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan"}:
            continue
        label_by_id[item["id"]] = text

    if not label_by_id:
        return (
            {
                "skipped": True,
                "reason": f"No traces carry a non-empty `{type_field}` label.",
            },
            pd.DataFrame(),
        )

    if "id" not in feature_df.columns:
        return (
            {"skipped": True, "reason": "feature_df is missing the `id` column."},
            pd.DataFrame(),
        )

    df = feature_df[feature_df["id"].isin(label_by_id)].copy()
    df["true_type"] = df["id"].map(label_by_id)

    counts = df["true_type"].value_counts()
    keep_classes = counts[counts >= int(min_class_count)].index.tolist()
    if len(keep_classes) < 2:
        return (
            {
                "skipped": True,
                "reason": (
                    f"Fewer than 2 failure-type classes have >= {min_class_count} "
                    f"samples (counts: {counts.to_dict()})."
                ),
                "min_class_count": int(min_class_count),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
            },
            pd.DataFrame(),
        )

    df = df[df["true_type"].isin(keep_classes)].reset_index(drop=True)

    feature_sets = choose_feature_sets(df, extra_meta_columns=["true_type"])
    feature_cols = feature_sets.get("trajectory_full") or feature_sets.get(
        "trajectory_shape"
    ) or feature_sets.get("length_only", [])
    if not feature_cols:
        return (
            {"skipped": True, "reason": "No usable trajectory features for this run."},
            pd.DataFrame(),
        )

    classes = sorted(df["true_type"].unique().tolist())
    class_counts = {
        str(k): int(v) for k, v in df["true_type"].value_counts().items()
    }
    smallest = min(class_counts.values())
    n_splits = min(int(cv_folds), smallest)
    if n_splits < 2:
        return (
            {
                "skipped": True,
                "reason": (
                    f"Smallest class has {smallest} samples; need >= 2 for CV."
                ),
                "class_counts": class_counts,
            },
            pd.DataFrame(),
        )

    X = df[feature_cols]
    y = df["true_type"].to_numpy()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_iter = list(cv.split(X, y))
    estimator = get_spec("logreg").build_estimator(random_state)

    pred_classes = cross_val_predict(estimator, X, y, cv=cv_iter, method="predict")
    proba = cross_val_predict(estimator, X, y, cv=cv_iter, method="predict_proba")

    estimator.fit(X, y)
    proba_classes = list(estimator.named_steps["logreg"].classes_)

    folds = np.full(len(X), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(cv_iter):
        folds[test_idx] = fold_idx

    oof = pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "fold": folds,
            "subject": df.get("subject", pd.Series(["unknown"] * len(df))).astype(str).to_numpy(),
            "true_type": y,
            "pred_type": pred_classes,
            "correct": (pred_classes == y).astype(int),
        }
    )
    for col_idx, cls in enumerate(proba_classes):
        oof[f"prob_{cls}"] = proba[:, col_idx]

    overall = classification_metrics_block(y, pred_classes, labels=classes)

    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y, pred_classes, labels=classes, zero_division=0
    )
    per_class = {
        str(cls): {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(classes)
    }

    cm = sk_confusion_matrix(y, pred_classes, labels=classes)
    confusion = {
        str(true_cls): {
            str(pred_cls): int(cm[i, j]) for j, pred_cls in enumerate(classes)
        }
        for i, true_cls in enumerate(classes)
    }

    by_subject = metrics_by_group(
        oof,
        group_col="subject",
        y_true_col="true_type",
        y_pred_col="pred_type",
        labels=classes,
    )

    summary = {
        "model": "logreg",
        "feature_set": "trajectory_full",
        "type_field": type_field,
        "n_items": int(len(df)),
        "n_classes": int(len(classes)),
        "classes": classes,
        "class_counts": class_counts,
        "cv_folds": int(n_splits),
        "metrics": overall,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "by_subject": by_subject,
    }
    return summary, oof


def evaluate_run(
    config: dict,
    *,
    run_slug: str | None = None,
) -> tuple[dict, dict[str, pd.DataFrame], Path]:
    """Run the full SLoD-only failure-attribution baseline for one SH6 run."""
    cfg = _cfg(config)
    cv_folds = int(cfg.get("cv_folds", 5))
    random_state = int(cfg.get("random_state", 42))
    min_class_count = int(cfg.get("min_class_count", 5))

    merged, feature_df, dataset_name, resolved_slug, reports_dir = load_run(
        config, run_slug=run_slug
    )
    type_field = _resolve_type_label_field(dataset_name)

    summary: dict = {
        "dataset": dataset_name,
        "run_slug": resolved_slug,
        "type_field": type_field,
        "config": {
            "cv_folds": cv_folds,
            "random_state": random_state,
            "min_class_count": min_class_count,
        },
    }
    outputs: dict[str, pd.DataFrame] = {}

    if any(item.get("error_step_index") is not None for item in merged):
        loc_summary, loc_trace_oof, loc_step_oof = evaluate_location(
            merged,
            feature_df,
            cv_folds=cv_folds,
            random_state=random_state,
            type_field=type_field,
        )
        summary["location"] = loc_summary
        if not loc_trace_oof.empty:
            outputs["location_trace_oof"] = loc_trace_oof
        if not loc_step_oof.empty:
            outputs["location_step_oof"] = loc_step_oof
    else:
        summary["location"] = {
            "skipped": True,
            "reason": "No traces in this run carry an `error_step_index` label.",
        }

    if type_field:
        type_summary, type_oof = evaluate_type(
            merged,
            feature_df,
            cv_folds=cv_folds,
            random_state=random_state,
            type_field=type_field,
            min_class_count=min_class_count,
        )
        summary["type"] = type_summary
        if not type_oof.empty:
            outputs["type_oof"] = type_oof
    else:
        summary["type"] = {
            "skipped": True,
            "reason": (
                f"Dataset `{dataset_name}` has no categorical failure-type label."
            ),
        }

    return summary, outputs, reports_dir


def _format_pct(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{100 * float(value):.1f}%"


def _markdown_lines(summary: dict) -> list[str]:
    lines: list[str] = [
        "# SH6 Failure Attribution (SLoD-only)",
        "",
        f"**Dataset:** `{summary['dataset']}`  ",
        f"**Run:** `{summary['run_slug']}`  ",
        f"**Type label field:** `{summary.get('type_field') or '— (none for this dataset)'}`",
        "",
        "## Failure Location",
        "",
    ]
    loc = summary.get("location") or {}
    if loc.get("skipped"):
        lines.append(f"_Skipped: {loc.get('reason', 'no reason given')}_")
    else:
        lines.extend(
            [
                "| Model | Step localization accuracy | Failed traces | Step rows | CV folds |",
                "|---|---|---|---|---|",
                (
                    f"| {loc['model']} "
                    f"| {_format_pct(loc['step_localization_accuracy'])} "
                    f"| {loc['n_failed']} "
                    f"| {loc['n_step_rows']} "
                    f"| {loc['cv_folds']} |"
                ),
            ]
        )
        if loc.get("by_failure_type"):
            lines.extend(
                [
                    "",
                    "### By failure type",
                    "",
                    "| Failure type | Step localization accuracy | Failed traces |",
                    "|---|---|---|",
                ]
            )
            for ft, m in loc["by_failure_type"].items():
                lines.append(
                    f"| {ft} | {_format_pct(m['step_localization_accuracy'])} | {m['n_failed']} |"
                )
        if loc.get("by_subject"):
            lines.extend(
                [
                    "",
                    "### By subject",
                    "",
                    "| Subject | Step localization accuracy | Failed traces |",
                    "|---|---|---|",
                ]
            )
            for sub, m in loc["by_subject"].items():
                lines.append(
                    f"| {sub} | {_format_pct(m['step_localization_accuracy'])} | {m['n_failed']} |"
                )

    lines.extend(["", "## Failure Type", ""])
    typ = summary.get("type") or {}
    if typ.get("skipped"):
        lines.append(f"_Skipped: {typ.get('reason', 'no reason given')}_")
    else:
        m = typ["metrics"]
        lines.extend(
            [
                "| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |",
                "|---|---|---|---|---|---|---|",
                (
                    f"| {typ['model']} "
                    f"| {_format_pct(m['macro_f1'])} "
                    f"| {_format_pct(m['macro_recall'])} "
                    f"| {_format_pct(m['accuracy'])} "
                    f"| {typ['n_items']} "
                    f"| {typ['n_classes']} "
                    f"| {typ['cv_folds']} |"
                ),
                "",
                "### Per class",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---|---|---|---|",
            ]
        )
        for cls, pc in typ["per_class"].items():
            lines.append(
                f"| {cls} | {_format_pct(pc['precision'])} "
                f"| {_format_pct(pc['recall'])} | {_format_pct(pc['f1'])} | {pc['support']} |"
            )
    return lines


def write_outputs(
    summary: dict,
    outputs: dict[str, pd.DataFrame],
    reports_dir: Path,
) -> dict[str, Path]:
    """Persist JSON / Markdown summaries and OOF artifacts."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary_path = reports_dir / "failure_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary_json"] = summary_path

    md_path = reports_dir / "failure_attribution.md"
    md_path.write_text("\n".join(_markdown_lines(summary)) + "\n", encoding="utf-8")
    paths["summary_md"] = md_path

    if "location_trace_oof" in outputs:
        loc_oof_path = reports_dir / "failure_location_oof.csv"
        outputs["location_trace_oof"].to_csv(loc_oof_path, index=False)
        paths["location_oof_csv"] = loc_oof_path
    if "location_step_oof" in outputs:
        loc_step_path = reports_dir / "failure_location_step_scores.csv"
        outputs["location_step_oof"].to_csv(loc_step_path, index=False)
        paths["location_step_scores_csv"] = loc_step_path
    if "type_oof" in outputs:
        type_oof_path = reports_dir / "failure_type_oof.csv"
        outputs["type_oof"].to_csv(type_oof_path, index=False)
        paths["type_oof_csv"] = type_oof_path

    return paths
