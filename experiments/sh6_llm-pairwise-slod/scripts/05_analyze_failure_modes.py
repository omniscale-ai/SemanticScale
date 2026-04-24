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
from pathlib import Path

from semanticscale.sh6 import datasets as ds
from semanticscale.sh6.failure_analysis import (
    build_feature_table,
    prediction_feasibility,
    build_summary_payload,
    choose_feature_sets,
    evaluate_prediction_models,
    fit_full_model_coefficients,
    merge_traces_and_rankings,
    plot_roc_curves,
    plot_top_coefficients,
    score_univariate_features,
    write_feature_table,
    write_markdown_report,
    write_summary_json,
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

    df, resolved_target = build_feature_table(
        merged,
        target_label=target_label,
        interp_points=interp_points,
    )
    if df.empty:
        logger.error("No rows available for failure analysis")
        raise SystemExit(1)

    feature_sets = choose_feature_sets(df)
    can_predict, effective_cv_folds, feasibility_note = prediction_feasibility(df, cv_folds)
    model_results: dict[str, dict] = {}
    univariate_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    full_features = feature_sets.get("trajectory_full") or []
    if not full_features:
        logger.error("No usable features available for multivariate failure analysis")
        raise SystemExit(1)

    if can_predict and effective_cv_folds is not None:
        if feasibility_note:
            logger.info(feasibility_note)
        model_results = evaluate_prediction_models(
            df,
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

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_feature_table(df, reports_dir / "trajectory_features.csv")
    if model_results:
        plot_roc_curves(df["target"].to_numpy(), model_results, reports_dir / "failure_prediction_roc.png")
        plot_top_coefficients(
            coefficient_rows,
            reports_dir / "failure_feature_coefficients.png",
            top_k=top_k,
        )
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
    )
    write_summary_json(summary, reports_dir / "failure_prediction_summary.json")

    logger.info("Failure analysis complete. Outputs written to %s", reports_dir)


if __name__ == "__main__":
    main()
