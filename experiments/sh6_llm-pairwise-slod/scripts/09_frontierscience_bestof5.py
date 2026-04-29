#!/usr/bin/env python
"""SH6 — Stage 9: FrontierScience DeepSeek Olympiad best-of-5 reranking.

This analysis treats five same-config DeepSeek runs on FrontierScience as a
best-of-5 attempt pool, restricted to the Olympiad subset only.

For each run we:

1. Load the Olympiad attempt outcomes from ``traces.jsonl``.
2. Fit strict out-of-fold SLoD-based success scorers on
   ``by-origin/olympiad/trajectory_features.csv`` using the existing Stage-5
   model runner.
3. Use those OOF scores to pick one attempt per problem.

Headline metrics:

- average-per-attempt ``Pass@1`` across the five runs, counting errors as failed
  attempts
- oracle ``Pass@5`` across the same five attempts
- SLoD-selected ``Pass@1`` from the highest-confidence attempt per problem

Outputs:
    reports/_cross_dataset/frontierscience_deepseek_olympiad_bestof5.md
    reports/_cross_dataset/frontierscience_deepseek_olympiad_bestof5.json
    reports/_cross_dataset/frontierscience_deepseek_olympiad_bestof5_attempts.csv
    reports/_cross_dataset/frontierscience_deepseek_olympiad_bestof5_selected.csv

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/09_frontierscience_bestof5.py
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from semanticscale.sh6.failure_analysis_runner import run_models_on_run
from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_RUNS: list[str] = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s2",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s3",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s4",
]
DEFAULT_MODELS = ["logreg", "lightgbm"]
PRIMARY_MODEL_DEFAULT = "lightgbm"
FEATURE_SET_DEFAULT = "trajectory_full"
TARGET_LABEL_DEFAULT = "is_correct"
ORIGIN = "olympiad"

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "experiments/sh6_llm-pairwise-slod/reports"
DATA_ROOT = REPO_ROOT / "data/sh6"
CROSS_DATASET_DIR = REPORTS_ROOT / "_cross_dataset"


@dataclass(frozen=True)
class RunPaths:
    run: str
    traces_jsonl: Path
    features_csv: Path
    artifact_dir: Path
    report_dir: Path

    @classmethod
    def for_run(cls, run: str) -> "RunPaths":
        report_dir = REPORTS_ROOT / run / f"by-origin/{ORIGIN}"
        return cls(
            run=run,
            traces_jsonl=DATA_ROOT / run / "traces.jsonl",
            features_csv=report_dir / "trajectory_features.csv",
            artifact_dir=report_dir / "artifacts",
            report_dir=report_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Full run path (e.g. frontierscience/deepseek/deepseek-v3.2_reasoning-auto). May be repeated.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"OOF scorer models to fit/load. Defaults to: {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--primary-model",
        default=PRIMARY_MODEL_DEFAULT,
        help=f"Model used as the headline SLoD-selected Pass@1. Default: {PRIMARY_MODEL_DEFAULT}",
    )
    parser.add_argument(
        "--feature-set",
        default=FEATURE_SET_DEFAULT,
        help=f"Feature set passed to run_models_on_run. Default: {FEATURE_SET_DEFAULT}",
    )
    parser.add_argument(
        "--target-label",
        default=TARGET_LABEL_DEFAULT,
        help=f"Classification target label for the scorer. Default: {TARGET_LABEL_DEFAULT}",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Reuse existing per-slice OOF artifacts instead of re-fitting the scorers.",
    )
    return parser.parse_args()


def _run_short_slug(run: str) -> str:
    prefix = "frontierscience/"
    return run[len(prefix):] if run.startswith(prefix) else run


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _load_olympiad_attempts(paths: RunPaths, run_order: int) -> pd.DataFrame:
    if not paths.traces_jsonl.exists():
        raise FileNotFoundError(f"traces.jsonl not found at {paths.traces_jsonl}")

    rows: list[dict] = []
    with paths.traces_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("has_final_answer"):
                continue
            error = rec.get("error")
            rows.append(
                {
                    "id": str(rec["id"]),
                    "subject": str(rec.get("subject", "unknown")),
                    "run": paths.run,
                    "run_slug_short": _run_short_slug(paths.run),
                    "run_order": run_order,
                    "answered": not bool(error),
                    "error": error if error is not None else "",
                    "is_correct": bool(rec.get("is_correct")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No Olympiad rows found in {paths.traces_jsonl}")
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(f"Duplicate Olympiad ids in {paths.traces_jsonl}: {dupes}")
    return df.sort_values("id", ignore_index=True)


def _fit_oof_scores(
    paths: RunPaths,
    models: list[str],
    feature_set: str,
    target_label: str,
    cv_folds: int,
    random_state: int,
) -> None:
    if not paths.features_csv.exists():
        raise FileNotFoundError(
            f"trajectory_features.csv not found at {paths.features_csv}; run Stage 05 first"
        )
    logger.info("[%s] fitting OOF scorers on %s", paths.run, paths.features_csv)
    run_models_on_run(
        features_csv=paths.features_csv,
        artifact_dir=paths.artifact_dir,
        models=models,
        feature_set=feature_set,
        target_label=target_label,
        task_type="classification",
        cv_folds=cv_folds,
        random_state=random_state,
        repo_root=REPO_ROOT,
    )


def _load_oof_scores(paths: RunPaths, models: list[str]) -> tuple[pd.DataFrame, dict[str, dict]]:
    merged: pd.DataFrame | None = None
    metrics: dict[str, dict] = {}

    for model in models:
        oof_path = paths.artifact_dir / f"oof_predictions_{model}.parquet"
        if not oof_path.exists():
            raise FileNotFoundError(
                f"Missing {oof_path}; fit scorers first or omit --skip-fit"
            )
        oof = pd.read_parquet(oof_path)
        required = {"id", "target", "prob"}
        missing = required.difference(oof.columns)
        if missing:
            raise ValueError(f"{oof_path} missing columns: {sorted(missing)}")
        oof["id"] = oof["id"].astype(str)
        if oof["id"].duplicated().any():
            dupes = oof.loc[oof["id"].duplicated(), "id"].tolist()[:5]
            raise ValueError(f"Duplicate ids in {oof_path}: {dupes}")

        score_col = f"score_{model}"
        score_df = oof[["id", "prob"]].rename(columns={"prob": score_col})
        merged = score_df if merged is None else merged.merge(score_df, on="id", how="outer")

        metrics[model] = {
            "oof_auc": float(roc_auc_score(oof["target"].astype(int), oof["prob"].astype(float))),
            "n_scored": int(len(oof)),
            "oof_path": str(oof_path),
        }

    assert merged is not None
    return merged, metrics


def _validate_problem_alignment(run_frames: list[pd.DataFrame]) -> None:
    base = run_frames[0][["id", "subject"]].rename(columns={"subject": "subject_base"})
    base_ids = set(base["id"])

    for frame in run_frames[1:]:
        ids = set(frame["id"])
        if ids != base_ids:
            only_here = sorted(ids - base_ids)[:5]
            only_base = sorted(base_ids - ids)[:5]
            raise ValueError(
                "Olympiad id mismatch across runs: "
                f"only_in_current={only_here}, only_in_base={only_base}"
            )
        merged = frame[["id", "subject"]].merge(base, on="id", how="inner")
        mismatch = merged.loc[merged["subject"] != merged["subject_base"]]
        if not mismatch.empty:
            row = mismatch.iloc[0]
            raise ValueError(
                f"Subject mismatch for id={row['id']}: {row['subject']} vs {row['subject_base']}"
            )


def _select_attempts(attempts: pd.DataFrame, model: str) -> pd.DataFrame:
    score_col = f"score_{model}"
    ranked = attempts.copy()
    ranked["_has_score"] = ranked[score_col].notna()
    ranked["_score_sort"] = ranked[score_col].fillna(-np.inf)
    ranked = ranked.sort_values(
        ["id", "_has_score", "_score_sort", "answered", "run_order"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    selected = ranked.groupby("id", as_index=False).head(1).copy()
    selected["model"] = model
    selected["selected_score"] = selected[score_col]
    return selected.drop(columns=["_has_score", "_score_sort"])


def _selection_summary(
    attempts: pd.DataFrame,
    selected: pd.DataFrame,
    model: str,
    n_runs: int,
    baseline_pass1: float,
    pass5: float,
) -> dict:
    score_col = f"score_{model}"
    scored_per_problem = attempts.groupby("id")[score_col].apply(lambda s: int(s.notna().sum()))
    selected_pass1 = float(selected["is_correct"].mean())
    oracle_gap = pass5 - baseline_pass1
    gap_recovered = (
        (selected_pass1 - baseline_pass1) / oracle_gap if oracle_gap > 0 else float("nan")
    )
    return {
        "model": model,
        "selected_pass1": selected_pass1,
        "delta_vs_avg_pass1": selected_pass1 - baseline_pass1,
        "oracle_gap_recovered": gap_recovered,
        "n_scored_attempts": int(attempts[score_col].notna().sum()),
        "n_all_scored_problems": int((scored_per_problem == n_runs).sum()),
        "n_partially_scored_problems": int(
            ((scored_per_problem > 0) & (scored_per_problem < n_runs)).sum()
        ),
        "n_zero_scored_problems": int((scored_per_problem == 0).sum()),
    }


def _write_outputs(
    *,
    runs: list[str],
    primary_model: str,
    feature_set: str,
    target_label: str,
    attempts: pd.DataFrame,
    selected_rows: pd.DataFrame,
    run_rows: list[dict],
    selection_rows: list[dict],
    baseline_pass1: float,
    pass5: float,
) -> None:
    CROSS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    stem = "frontierscience_deepseek_olympiad_bestof5"

    attempts_out = CROSS_DATASET_DIR / f"{stem}_attempts.csv"
    selected_out = CROSS_DATASET_DIR / f"{stem}_selected.csv"
    summary_json = CROSS_DATASET_DIR / f"{stem}.json"
    summary_md = CROSS_DATASET_DIR / f"{stem}.md"

    attempts.to_csv(attempts_out, index=False)
    selected_rows.to_csv(selected_out, index=False)

    primary = next(row for row in selection_rows if row["model"] == primary_model)
    payload = {
        "subset": ORIGIN,
        "runs": runs,
        "feature_set": feature_set,
        "target_label": target_label,
        "baseline_pass1_avg_per_attempt": baseline_pass1,
        "pass5": pass5,
        "primary_model": primary_model,
        "primary_selected_pass1": primary["selected_pass1"],
        "run_rows": run_rows,
        "selection_rows": selection_rows,
        "attempts_csv": str(attempts_out),
        "selected_csv": str(selected_out),
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# FrontierScience DeepSeek Olympiad Best-of-5",
        "",
        "Strict same-dataset reranking analysis over the five DeepSeek FrontierScience runs,",
        "using only the Olympiad subset and strict Olympiad-only out-of-fold SLoD scores.",
        "",
        "## Setup",
        "",
        f"- Runs: {len(runs)}",
        f"- Subset: `{ORIGIN}` only",
        f"- Scorer feature set: `{feature_set}`",
        f"- Scorer target label: `{target_label}`",
        f"- Headline selection model: `{primary_model}`",
        "- Pass metrics count inference/grading errors as failed attempts.",
        "- Unscored attempts rank last during selection; if a problem has no scored attempts at all, the deterministic fallback still counts as a failure.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Average-per-attempt Pass@1 | {_pct(baseline_pass1)} |",
        f"| Pass@5 (oracle across the five attempts) | {_pct(pass5)} |",
        f"| SLoD-selected Pass@1 ({primary_model}) | {_pct(primary['selected_pass1'])} |",
        f"| Gain over average Pass@1 ({primary_model}) | {primary['delta_vs_avg_pass1']:+.3f} ({100.0 * primary['delta_vs_avg_pass1']:+.1f} pts) |",
        f"| Oracle gap recovered ({primary_model}) | {_pct(primary['oracle_gap_recovered']) if np.isfinite(primary['oracle_gap_recovered']) else 'n/a'} |",
        "",
        "## Per-run attempt baselines",
        "",
        "| Run | Answered | Errors | Pass@1 | logreg OOF AUC | lightgbm OOF AUC |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in run_rows:
        lines.append(
            f"| `{row['run_slug_short']}` | {row['answered']} | {row['errors']} | "
            f"{_pct(row['pass1'])} | {row.get('oof_auc_logreg', float('nan')):.3f} | "
            f"{row.get('oof_auc_lightgbm', float('nan')):.3f} |"
        )

    lines += [
        "",
        "## Reranking results",
        "",
        "| Scorer | Scored attempts | Problems with 5 scores | Problems with partial scores | Problems with 0 scores | Selected Pass@1 | Delta vs avg Pass@1 | Oracle gap recovered |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in selection_rows:
        gap = (
            _pct(row["oracle_gap_recovered"])
            if np.isfinite(row["oracle_gap_recovered"])
            else "n/a"
        )
        lines.append(
            f"| `{row['model']}` | {row['n_scored_attempts']} | {row['n_all_scored_problems']} | "
            f"{row['n_partially_scored_problems']} | {row['n_zero_scored_problems']} | "
            f"{_pct(row['selected_pass1'])} | {row['delta_vs_avg_pass1']:+.3f} | {gap} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is a **useful utility evaluation**: it tests whether SLoD-derived confidence helps choose among five attempts from the same model/config family.",
        "- It is **only partially fair as core evidence for SLoD**. The FrontierScience answer-side detector family was selected post-hoc on FrontierScience, so gains here do not by themselves prove broad SLoD validity.",
        "- The important safeguard in this analysis is that selection uses **out-of-fold** scores on the Olympiad slice, which avoids item-level in-sample leakage.",
        "- Stronger evidence would require transfer: e.g. freeze the scorer, then rerank a different FrontierScience generator or a different dataset entirely.",
        "",
        f"Artifacts: `{attempts_out.name}`, `{selected_out.name}`, `{summary_json.name}`.",
    ]
    summary_md.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", summary_md)


def main() -> int:
    setup_logging()
    args = parse_args()
    runs = args.run or list(DEFAULT_RUNS)
    models = list(args.models)
    if args.primary_model not in models:
        raise ValueError(
            f"--primary-model {args.primary_model!r} must be included in --models {models}"
        )

    path_rows = [RunPaths.for_run(run) for run in runs]
    attempt_frames: list[pd.DataFrame] = []
    run_rows: list[dict] = []

    for run_order, paths in enumerate(path_rows):
        attempt_df = _load_olympiad_attempts(paths, run_order=run_order)
        if not args.skip_fit:
            _fit_oof_scores(
                paths=paths,
                models=models,
                feature_set=args.feature_set,
                target_label=args.target_label,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
            )
        score_df, oof_metrics = _load_oof_scores(paths, models)
        attempt_df = attempt_df.merge(score_df, on="id", how="left")
        attempt_frames.append(attempt_df)
        run_rows.append(
            {
                "run": paths.run,
                "run_slug_short": _run_short_slug(paths.run),
                "answered": int(attempt_df["answered"].sum()),
                "errors": int((~attempt_df["answered"]).sum()),
                "pass1": float(attempt_df["is_correct"].mean()),
                **{f"oof_auc_{model}": info["oof_auc"] for model, info in oof_metrics.items()},
            }
        )

    _validate_problem_alignment(attempt_frames)
    attempts = pd.concat(attempt_frames, ignore_index=True)
    attempts = attempts.sort_values(["id", "run_order"], ignore_index=True)

    baseline_pass1 = float(attempts["is_correct"].mean())
    pass5 = float(attempts.groupby("id")["is_correct"].max().mean())

    selected_frames: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    for model in models:
        selected = _select_attempts(attempts, model)
        per_problem = (
            attempts.groupby("id", as_index=False)
            .agg(
                available_correct_attempts=("is_correct", "sum"),
                scored_attempts_for_problem=(
                    f"score_{model}",
                    lambda s: int(s.notna().sum()),
                ),
            )
        )
        selected = selected.merge(per_problem, on="id", how="left")
        selected["available_correct_attempts"] = selected["available_correct_attempts"].astype(int)
        selected["scored_attempts_for_problem"] = selected["scored_attempts_for_problem"].astype(int)
        selected_frames.append(selected)
        selection_rows.append(
            _selection_summary(
                attempts=attempts,
                selected=selected,
                model=model,
                n_runs=len(runs),
                baseline_pass1=baseline_pass1,
                pass5=pass5,
            )
        )

    selected_rows = pd.concat(selected_frames, ignore_index=True)
    _write_outputs(
        runs=runs,
        primary_model=args.primary_model,
        feature_set=args.feature_set,
        target_label=args.target_label,
        attempts=attempts,
        selected_rows=selected_rows,
        run_rows=run_rows,
        selection_rows=selection_rows,
        baseline_pass1=baseline_pass1,
        pass5=pass5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
