#!/usr/bin/env python
"""SH6 — Stage 3c: trajectory failure-mode prediction.

Builds per-trace trajectory features and evaluates whether they predict
final-answer success/failure out of sample.

This stage is intentionally conservative:

- it prefers ``final_answer_correct`` as the target when available
- it compares trajectory-derived features against a chunk-count baseline
- it skips predictive evaluation entirely when the run does not contain enough
  label variation to make the task identifiable

Outputs:
    reports/{dataset}/{run_slug}/trajectory_features.csv
    reports/{dataset}/{run_slug}/failure_prediction_summary.json
    reports/{dataset}/{run_slug}/failure_prediction.md
    reports/{dataset}/{run_slug}/failure_prediction_roc.png
    reports/{dataset}/{run_slug}/failure_feature_coefficients.png

Usage:
    python scripts/05_analyze_failure_modes.py --config config-processbench.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from semanticscale.sh6 import datasets as ds
from semanticscale.sh6.failure_analysis import (
    build_feature_table,
    prediction_feasibility,
    build_summary_payload,
    choose_feature_sets,
    compute_capture_ratio,
    compute_mode_coverage,
    evaluate_prediction_models,
    fit_full_model_coefficients,
    merge_traces_and_rankings,
    plot_mode_detector_summary,
    plot_roc_curves,
    plot_top_coefficients,
    score_univariate_features,
    write_feature_table,
    write_markdown_report,
    write_summary_json,
)
from semanticscale.sh6.failure_modes import (
    MODES,
    compute_mode_scores,
    evaluate_mode_detectors,
    write_mode_table,
)
from semanticscale.utils import load_config, load_jsonl, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--run-slug", default=None, dest="run_slug")
    parser.add_argument(
        "--target",
        default=None,
        choices=["auto", "final_answer_correct", "is_correct"],
        help="Override the configured target label",
    )
    return parser.parse_args()


def _safe_slice_dir(value: str) -> str:
    return value.replace("/", "_")


def _run_failure_analysis(
    *,
    merged: list[dict],
    dataset_name: str,
    run_slug: str,
    reports_dir: Path,
    target_label: str,
    interp_points: int,
    cv_folds: int,
    random_state: int,
    top_k: int,
) -> None:
    """Run the full failure-analysis pipeline on a list of merged traces.

    Writes all reports under ``reports_dir``. Used both for the global per-run
    analysis and for per-slice sub-analyses.
    """
    df, resolved_target = build_feature_table(
        merged,
        target_label=target_label,
        interp_points=interp_points,
    )
    if df.empty:
        logger.warning(
            "No rows available for failure analysis at %s — skipping", reports_dir
        )
        return

    mode_scores_df = compute_mode_scores(df, modes=MODES)
    score_cols = [c for c in mode_scores_df.columns if c.endswith("_score")]
    df_with_scores = pd.concat(
        [df.reset_index(drop=True), mode_scores_df[score_cols].reset_index(drop=True)],
        axis=1,
    )
    feature_sets = choose_feature_sets(df_with_scores, extra_meta_columns=score_cols)
    mode_stack = [
        col
        for col in score_cols
        if df_with_scores[col].notna().any() and df_with_scores[col].dropna().nunique() >= 2
    ]
    if mode_stack:
        feature_sets["mode_stack"] = mode_stack
    can_predict, effective_cv_folds, feasibility_note = prediction_feasibility(df, cv_folds)
    model_results: dict[str, dict] = {}
    univariate_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    full_features = feature_sets.get("trajectory_full") or []
    if not full_features:
        logger.warning(
            "No usable features for multivariate failure analysis at %s — skipping",
            reports_dir,
        )
        return

    if can_predict and effective_cv_folds is not None:
        if feasibility_note:
            logger.info(feasibility_note)
        model_results = evaluate_prediction_models(
            df_with_scores,
            feature_sets,
            random_state=random_state,
            cv_folds=effective_cv_folds,
        )
        univariate_rows = score_univariate_features(
            df,
            full_features,
            random_state=random_state,
            cv_folds=effective_cv_folds,
        )
        coefficient_rows = fit_full_model_coefficients(
            df,
            full_features,
            random_state=random_state,
        )
    else:
        logger.warning(feasibility_note or "Prediction is not feasible for this run")

    mode_rows: list[dict] = []
    if can_predict:
        mode_rows = evaluate_mode_detectors(df, mode_scores_df, modes=MODES)
    else:
        logger.warning("Skipping failure-mode detector evaluation: prediction infeasible")

    coverage = compute_mode_coverage(df, mode_scores_df, mode_rows)
    capture = compute_capture_ratio(model_results)

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_feature_table(df, reports_dir / "trajectory_features.csv")
    write_mode_table(df, mode_scores_df, reports_dir / "failure_modes.csv")
    if model_results:
        plot_roc_curves(df["target"].to_numpy(), model_results, reports_dir / "failure_prediction_roc.png")
        plot_top_coefficients(
            coefficient_rows,
            reports_dir / "failure_feature_coefficients.png",
            top_k=top_k,
        )
    if mode_rows:
        plot_mode_detector_summary(mode_rows, reports_dir / "failure_mode_detectors.png")
    write_markdown_report(
        df,
        target_label=resolved_target,
        model_results=model_results,
        univariate_rows=univariate_rows,
        coefficient_rows=coefficient_rows,
        out_path=reports_dir / "failure_prediction.md",
        dataset_name=dataset_name,
        run_slug=run_slug,
        analysis_note=feasibility_note,
        mode_rows=mode_rows,
        coverage=coverage,
        capture=capture,
    )
    summary = build_summary_payload(
        df,
        dataset_name=dataset_name,
        run_slug=run_slug,
        target_label=resolved_target,
        model_results=model_results,
        univariate_rows=univariate_rows,
        coefficient_rows=coefficient_rows,
        analysis_note=feasibility_note,
        mode_rows=mode_rows,
        coverage=coverage,
        capture=capture,
    )
    write_summary_json(summary, reports_dir / "failure_prediction_summary.json")

    logger.info("Failure analysis complete. Outputs written to %s", reports_dir)


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    dataset_name = ds.dataset_name(config)
    run_slug = args.run_slug or ds.run_slug(config)

    analysis_cfg = config.get("failure_analysis", {})
    target_label = args.target or analysis_cfg.get("target_label", "auto")
    interp_points = int(analysis_cfg.get("interp_points", 20))
    cv_folds = int(analysis_cfg.get("cv_folds", 5))
    random_state = int(analysis_cfg.get("random_state", 42))
    top_k = int(analysis_cfg.get("top_k_features", 12))

    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve() / dataset_name / run_slug
    run_dir = data_dir / dataset_name / run_slug

    traces_path = run_dir / "traces.jsonl"
    rankings_path = run_dir / "chunk_rankings.jsonl"
    if not traces_path.exists():
        logger.error("traces.jsonl not found at %s", traces_path)
        raise SystemExit(1)
    if not rankings_path.exists():
        logger.error("chunk_rankings.jsonl not found at %s — run 02_slod.py first", rankings_path)
        raise SystemExit(1)

    traces = load_jsonl(traces_path)
    rankings = load_jsonl(rankings_path)
    merged = merge_traces_and_rankings(traces, rankings)
    logger.info("Merged %d ranked traces", len(merged))

    _run_failure_analysis(
        merged=merged,
        dataset_name=dataset_name,
        run_slug=run_slug,
        reports_dir=reports_dir,
        target_label=target_label,
        interp_points=interp_points,
        cv_folds=cv_folds,
        random_state=random_state,
        top_k=top_k,
    )

    sl_name = ds.slice_name(config)
    if not sl_name:
        return

    slices: dict[str, list[dict]] = defaultdict(list)
    for item in merged:
        label = ds.slice_label(config, item)
        if label is None:
            continue
        slices[label].append(item)

    for label, items in sorted(slices.items()):
        slice_reports_dir = reports_dir / f"by-{sl_name}" / _safe_slice_dir(label)
        slice_run_slug = f"{run_slug}/by-{sl_name}/{label}"
        logger.info(
            "Per-%s slice %s: %d items → %s",
            sl_name, label, len(items), slice_reports_dir,
        )
        _run_failure_analysis(
            merged=items,
            dataset_name=dataset_name,
            run_slug=slice_run_slug,
            reports_dir=slice_reports_dir,
            target_label=target_label,
            interp_points=interp_points,
            cv_folds=cv_folds,
            random_state=random_state,
            top_k=top_k,
        )


if __name__ == "__main__":
    main()
