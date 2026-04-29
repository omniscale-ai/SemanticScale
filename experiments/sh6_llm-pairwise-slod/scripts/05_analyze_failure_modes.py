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
    python scripts/05_analyze_failure_modes.py --config config/processbench-gsm8k.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
    resolve_run_paths,
    score_univariate_features,
    trajectory_series_columns,
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
from semanticscale.sh6.failure_analysis_runner import run_models_on_run
from semanticscale.utils import load_config, load_jsonl, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("logreg", "lightgbm", "minirocket")
REPORTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "reports"
PROTOCOL_DELTA = 0.03
N_BOOTSTRAP_DEFAULT = 1000
PROTOCOL_RUN_PATTERNS = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "swe-agent-trajectories/model-all",
    "agenthallu/framework-all",
    "gpqa-diamond/deepseek/deepseek-v3.2_reasoning-auto",
    "processbench/gsm8k",
]


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
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=(
            "Run model comparison artifacts on trajectory_features.csv. "
            f"Defaults to: {' '.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--feature-set",
        default="trajectory_full",
        help="Feature set name used for model comparison artifacts.",
    )
    parser.add_argument(
        "--model-cv-folds",
        type=int,
        default=None,
        help="CV folds for model-comparison artifacts (defaults to failure_analysis.cv_folds).",
    )
    parser.add_argument(
        "--skip-failure-analysis",
        action="store_true",
        dest="skip_failure_analysis",
        help="Skip legacy Stage-05 report generation; useful with --models.",
    )
    parser.add_argument(
        "--aggregate-models",
        action="store_true",
        help="Run cross-dataset aggregation over OOF artifacts under reports/.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR_DEFAULT,
        help="Root reports directory for --aggregate-models mode.",
    )
    parser.add_argument("--incumbent", default="logreg")
    parser.add_argument("--challenger", default="lightgbm")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_slice_dir(value: str) -> str:
    return value.replace("/", "_")


def _attach_file_log(log_path: Path) -> logging.Handler:
    """Stream INFO logs into a deterministic run-local log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _minirocket_trajectory_cols(features_csv: Path) -> list[str]:
    """Return sorted reasoning_traj_t* columns from a features CSV, or []."""
    header = pd.read_csv(features_csv, nrows=0)
    return sorted(c for c in header.columns if c.startswith("reasoning_traj_t"))


def _run_model_comparison_for_run(
    *,
    reports_dir: Path,
    models: list[str],
    feature_set: str,
    target_label: str,
    cv_folds: int,
    random_state: int,
) -> None:
    """Run additional model artifacts for one run using trajectory_features.csv.

    MiniRocket is routed separately: it operates on the ``reasoning_traj_t*``
    time-series columns rather than the tabular ``feature_set``.
    """
    features_csv = reports_dir / "trajectory_features.csv"
    if not features_csv.exists():
        logger.error("Features CSV not found: %s", features_csv)
        raise SystemExit(1)

    artifact_dir = reports_dir / "artifacts"
    log_path = reports_dir / "logs" / f"05_model_comparison_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    file_handler = _attach_file_log(log_path)
    try:
        logger.info("Running model comparison on %s", features_csv)
        logger.info("Artifacts dir: %s", artifact_dir)
        logger.info("Models: %s", models)

        tabular_models = [m for m in models if m != "minirocket"]
        summary: dict = {}

        if tabular_models:
            result = run_models_on_run(
                features_csv=features_csv,
                artifact_dir=artifact_dir,
                models=tabular_models,
                feature_set=feature_set,
                target_label=target_label,
                cv_folds=cv_folds,
                random_state=random_state,
                repo_root=Path.cwd(),
            )
            if "_skipped" in result:
                logger.warning("Model comparison skipped: %s", result["_skipped"]["reason"])
                return
            summary.update(result)

        if "minirocket" in models:
            traj_cols = _minirocket_trajectory_cols(features_csv)
            if not traj_cols:
                logger.warning(
                    "MiniRocket requested but no reasoning_traj_t* columns found in %s "
                    "— re-run without --skip-failure-analysis to regenerate the CSV",
                    features_csv,
                )
            else:
                mr_result = run_models_on_run(
                    features_csv=features_csv,
                    artifact_dir=artifact_dir,
                    models=["minirocket"],
                    feature_set=feature_set,
                    target_label=target_label,
                    cv_folds=cv_folds,
                    random_state=random_state,
                    repo_root=Path.cwd(),
                    feature_cols_override=traj_cols,
                )
                if "_skipped" in mr_result:
                    logger.warning("MiniRocket skipped: %s", mr_result["_skipped"]["reason"])
                else:
                    summary.update(mr_result)

        for model_name, info in summary.items():
            logger.info(
                "%s — AUC %.4f +/- %.4f (%.2fs)",
                model_name,
                info["auc_mean"],
                info["auc_std"],
                info["duration_s"],
            )
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _find_runs_with_oof(reports_dir: Path, incumbent: str, challenger: str) -> list[Path]:
    """Locate every run directory that has incumbent and challenger OOF files."""
    runs: list[Path] = []
    for incumbent_oof in reports_dir.rglob(f"oof_predictions_{incumbent}.parquet"):
        artifact_dir = incumbent_oof.parent
        if not (artifact_dir / f"oof_predictions_{challenger}.parquet").exists():
            continue
        runs.append(artifact_dir.parent)
    return sorted(runs)


def _load_oof(run_dir: Path, model_name: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "artifacts" / f"oof_predictions_{model_name}.parquet")


def _paired_bootstrap_delta(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return mean delta and 95% percentile-bootstrap interval for paired AUCs."""
    n = len(y_true)
    deltas = np.empty(n_bootstrap, dtype=float)
    valid = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() in (0, n):
            continue
        deltas[valid] = roc_auc_score(y_b, p_b[idx]) - roc_auc_score(y_b, p_a[idx])
        valid += 1
    if valid == 0:
        return float("nan"), float("nan"), float("nan")
    deltas = deltas[:valid]
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _model_comparison_verdict(delta_point: float, ci_low: float) -> str:
    if np.isnan(ci_low):
        return "insufficient_data"
    if delta_point >= PROTOCOL_DELTA and ci_low > 0:
        return "win"
    if delta_point <= -PROTOCOL_DELTA and ci_low < 0:
        return "regress"
    return "inconclusive"


def _aggregate_model_outputs(
    reports_dir: Path,
    incumbent: str,
    challenger: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    runs = _find_runs_with_oof(reports_dir, incumbent, challenger)
    if not runs:
        raise SystemExit(
            f"No runs found with both {incumbent} and {challenger} OOF parquets under {reports_dir}"
        )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for run_dir in runs:
        rel = run_dir.relative_to(reports_dir).as_posix()
        oof_a = _load_oof(run_dir, incumbent).sort_values("row_index").reset_index(drop=True)
        oof_b = _load_oof(run_dir, challenger).sort_values("row_index").reset_index(drop=True)
        if not np.array_equal(oof_a["target"].values, oof_b["target"].values):
            logger.warning("Target mismatch in %s — skipping", rel)
            continue
        if not np.array_equal(oof_a["row_index"].values, oof_b["row_index"].values):
            logger.warning("Row order mismatch in %s — skipping", rel)
            continue

        y = oof_a["target"].to_numpy()
        p_a = oof_a["prob"].to_numpy()
        p_b = oof_b["prob"].to_numpy()
        auc_a = float(roc_auc_score(y, p_a))
        auc_b = float(roc_auc_score(y, p_b))
        delta_point = auc_b - auc_a
        delta_mean, ci_low, ci_high = _paired_bootstrap_delta(y, p_a, p_b, n_bootstrap, rng)
        in_protocol = any(rel == pat or rel.startswith(pat + "/") for pat in PROTOCOL_RUN_PATTERNS)
        rows.append(
            {
                "rel_path": rel,
                "n_items": len(y),
                "n_pos": int(y.sum()),
                f"auc_{incumbent}": auc_a,
                f"auc_{challenger}": auc_b,
                "delta_point": delta_point,
                "delta_bootstrap_mean": delta_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "verdict": _model_comparison_verdict(delta_point, ci_low),
                "in_protocol_set": in_protocol,
            }
        )
    return pd.DataFrame(rows).sort_values("rel_path", ignore_index=True)


def _render_model_comparison_markdown(df: pd.DataFrame, incumbent: str, challenger: str) -> str:
    lines = [
        "# SH6 Stage-5 - Cross-Dataset Model Comparison\n",
        f"**Incumbent:** `{incumbent}`   **Challenger:** `{challenger}`\n",
        (
            f"**Decision rule:** Delta-AUC >= +{PROTOCOL_DELTA:.2f} and CI-lower > 0 (per run); "
            ">=3 of 5 protocol runs to carry overall.\n"
        ),
    ]
    cols = [
        "rel_path",
        "n_items",
        "n_pos",
        f"auc_{incumbent}",
        f"auc_{challenger}",
        "delta_point",
        "ci_low",
        "ci_high",
        "verdict",
        "in_protocol_set",
    ]
    table = df[cols].copy()
    table.columns = [
        "Run",
        "N",
        "Pos",
        f"AUC {incumbent}",
        f"AUC {challenger}",
        "Delta-AUC",
        "CI low",
        "CI high",
        "Verdict",
        "In protocol set",
    ]

    def _format_metric(value: object, signed: bool = False) -> str:
        if pd.isna(value):
            return "-"
        if signed:
            return f"{value:+.3f}"
        return f"{value:.3f}"

    for c in [f"AUC {incumbent}", f"AUC {challenger}", "CI low", "CI high"]:
        table[c] = table[c].map(_format_metric)
    table["Delta-AUC"] = table["Delta-AUC"].map(lambda v: _format_metric(v, signed=True))
    lines.append(table.to_markdown(index=False))

    protocol = df[df["in_protocol_set"]]
    n_wins = int((protocol["verdict"] == "win").sum())
    n_protocol = len(protocol)
    n_regressions = int((protocol["verdict"] == "regress").sum())
    carry = n_wins >= 3
    lines.extend(
        [
            "\n## Protocol verdict\n",
            f"- Protocol runs evaluated: **{n_protocol} / 5**",
            f"- Challenger wins: **{n_wins}**",
            f"- Significant regressions: **{n_regressions}**",
            (
                "- >=3-of-5 carry rule: "
                f"**{'PASS - challenger carries the comparison' if carry else 'FAIL - challenger does NOT carry the comparison'}**"
            ),
        ]
    )
    if not carry:
        lines.append(
            "\nPer the stop conditions in `DESIGN-stage5-models.md`, this implies "
            "no broadly applicable interaction signal in the current Stage-5 "
            "feature set. The right next step is feature engineering (TA-pack, "
            "multi-scale, cross-trajectory), not a stronger classifier."
        )
    return "\n".join(lines) + "\n"


def _render_forest_plot(df: pd.DataFrame, incumbent: str, challenger: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df) + 1.5)))
    y_pos = np.arange(len(df))
    deltas = df["delta_point"].to_numpy()
    ci_low = df["ci_low"].to_numpy()
    ci_high = df["ci_high"].to_numpy()
    err_low = np.where(np.isnan(ci_low), 0, deltas - ci_low)
    err_high = np.where(np.isnan(ci_high), 0, ci_high - deltas)
    colors = [
        "#1b7837" if v == "win" else "#762a83" if v == "regress" else "#999999"
        for v in df["verdict"]
    ]
    ax.errorbar(deltas, y_pos, xerr=[err_low, err_high], fmt="none", ecolor="#777777", capsize=3, lw=1)
    ax.scatter(deltas, y_pos, c=colors, s=40, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(PROTOCOL_DELTA, color="#1b7837", lw=0.8, ls="--", label=f"+{PROTOCOL_DELTA:.2f} threshold")
    ax.axvline(-PROTOCOL_DELTA, color="#762a83", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["rel_path"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Delta-AUC ({challenger} - {incumbent}), 95% paired bootstrap CI")
    ax.set_title("SH6 Stage-5 - Delta-AUC across runs")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_cross_dataset_aggregation(
    *,
    reports_dir: Path,
    incumbent: str,
    challenger: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    out_dir = reports_dir / "_cross_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _aggregate_model_outputs(
        reports_dir=reports_dir,
        incumbent=incumbent,
        challenger=challenger,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    csv_path = out_dir / f"model_comparison_{incumbent}_vs_{challenger}.csv"
    md_path = out_dir / f"model_comparison_{incumbent}_vs_{challenger}.md"
    plot_path = out_dir / f"delta_auc_{incumbent}_vs_{challenger}.png"
    json_path = out_dir / f"model_comparison_{incumbent}_vs_{challenger}.json"

    df.to_csv(csv_path, index=False)
    md_path.write_text(_render_model_comparison_markdown(df, incumbent, challenger))
    _render_forest_plot(df, incumbent, challenger, plot_path)

    summary = {
        "n_runs": int(len(df)),
        "n_in_protocol": int(df["in_protocol_set"].sum()),
        "n_wins_protocol": int(((df["verdict"] == "win") & df["in_protocol_set"]).sum()),
        "n_regress_protocol": int(((df["verdict"] == "regress") & df["in_protocol_set"]).sum()),
        "carry": bool(((df["verdict"] == "win") & df["in_protocol_set"]).sum() >= 3),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s, %s, %s", csv_path, md_path, plot_path)
    logger.info("Summary: %s", summary)


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
    baseline_feature_set = "length_only"
    length_features = list(feature_sets.get(baseline_feature_set) or [])
    baseline_note = None
    truncation_feature = "truncation_abort_score"
    truncation_series = df_with_scores.get(truncation_feature)
    if truncation_series is not None:
        truncation_nonnull = truncation_series.dropna()
        if not truncation_nonnull.empty and truncation_nonnull.nunique() >= 2:
            if truncation_feature not in length_features:
                length_features.append(truncation_feature)
            feature_sets.pop("length_only", None)
            baseline_feature_set = "lenght_abort"
            feature_sets[baseline_feature_set] = length_features
            baseline_note = (
                "The `lenght_abort` baseline includes both chunk-count features "
                "and `truncation_abort_score` on this run."
            )
    else:
        feature_sets[baseline_feature_set] = length_features
    mode_stack_scores = [
        col
        for col in score_cols
        if col not in length_features
        and df_with_scores[col].notna().any()
        and df_with_scores[col].dropna().nunique() >= 2
    ]
    if mode_stack_scores:
        # Bundle length features in so the model has the structural baseline to
        # build on; this keeps mode_stack comparable to the baseline and lets us
        # read residual gain from the mode detectors directly.
        feature_sets["mode_stack"] = length_features + mode_stack_scores
    can_predict, effective_cv_folds, feasibility_note = prediction_feasibility(df, cv_folds)
    analysis_note = " ".join(note for note in (feasibility_note, baseline_note) if note) or None
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
        if analysis_note:
            logger.info(analysis_note)
        extra_models = {"lightgbm_trajectory_full": ("lightgbm", full_features)}
        reasoning_traj_cols = trajectory_series_columns(df_with_scores, prefix="reasoning")
        if reasoning_traj_cols:
            extra_models["minirocket_reasoning_traj"] = (
                "minirocket",
                reasoning_traj_cols,
            )
        else:
            logger.info("Skipping minirocket_reasoning_traj: no usable reasoning trajectory columns")
        mode_stack_features = feature_sets.get("mode_stack") or []
        if mode_stack_features:
            extra_models["lightgbm_mode_stack"] = ("lightgbm", mode_stack_features)
        model_results = evaluate_prediction_models(
            df_with_scores,
            feature_sets,
            random_state=random_state,
            cv_folds=effective_cv_folds,
            extra_models=extra_models,
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
        analysis_note=analysis_note,
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
        analysis_note=analysis_note,
        mode_rows=mode_rows,
        coverage=coverage,
        capture=capture,
    )
    write_summary_json(summary, reports_dir / "failure_prediction_summary.json")

    logger.info("Failure analysis complete. Outputs written to %s", reports_dir)


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.aggregate_models:
        reports_dir = args.reports_dir.resolve()
        _run_cross_dataset_aggregation(
            reports_dir=reports_dir,
            incumbent=args.incumbent,
            challenger=args.challenger,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        return

    config = load_config(args.config)

    analysis_cfg = config.get("failure_analysis", {})
    target_label = args.target or analysis_cfg.get("target_label", "auto")
    interp_points = int(analysis_cfg.get("interp_points", 20))
    cv_folds = int(analysis_cfg.get("cv_folds", 5))
    random_state = int(analysis_cfg.get("random_state", 42))
    top_k = int(analysis_cfg.get("top_k_features", 12))

    paths = resolve_run_paths(
        config,
        run_slug=args.run_slug,
        required_files=("traces.jsonl", "chunk_rankings.jsonl"),
    )
    dataset_name = paths.dataset_name
    run_slug = paths.run_slug
    reports_dir = paths.run_reports_dir
    if not args.skip_failure_analysis:
        run_dir = paths.run_dir

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
        if sl_name:
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
                    sl_name,
                    label,
                    len(items),
                    slice_reports_dir,
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
    else:
        logger.info("Skipping Stage-05 failure analysis report generation (--skip-failure-analysis)")

    if args.models:
        comparison_cv_folds = args.model_cv_folds or cv_folds
        _run_model_comparison_for_run(
            reports_dir=reports_dir,
            models=list(args.models),
            feature_set=args.feature_set,
            target_label=target_label,
            cv_folds=comparison_cv_folds,
            random_state=random_state,
        )
    elif args.skip_failure_analysis:
        logger.warning("--skip-failure-analysis was provided without --models; no run-level outputs requested")


if __name__ == "__main__":
    main()
