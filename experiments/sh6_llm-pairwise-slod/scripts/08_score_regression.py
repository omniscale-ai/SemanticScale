#!/usr/bin/env python
"""SH6 — Stage 8: FrontierScience numeric-score regression from SLoD features.

The classifier path collapses every FrontierScience problem to a binary
``is_correct``. For rubric-graded items it sets
``is_correct = (total_awarded >= 7.0)``, which on most runs throws away the
0.0–0.7 partial-credit gradient by mapping every rubric item to "fail".

This script asks: do the same SLoD trajectory features predict the continuous
``rubric_score = total_awarded / total_max`` on the rubric subset of each
FrontierScience run?

Pipeline:

1. For each eligible run (``trajectory_features.csv`` exists and the matching
   ``traces.jsonl`` has rubric-typed grades), backfill the ``rubric_score``
   column from ``traces.jsonl`` if not already present.
2. Run ridge + LightGBM regressors on the existing ``trajectory_full`` feature
   set under ``KFold(5, shuffle=True, random_state=42)``, plus a length-only
   baseline so the headline result includes a sanity floor.
3. Persist OOF predictions, per-fold metrics, feature importances under
   ``reports/.../artifacts/`` (using
   ``failure_analysis_runner.run_models_on_run`` with
   ``task_type="regression"``).
4. Compute pooled R² and Spearman ρ from the OOF predictions with a
   1000-draw percentile bootstrap CI. Write per-run ``score_regression.md``.
5. Aggregate into ``reports/_cross_dataset/score_regression_summary.{md,csv,json}``
   plus a predicted-vs-actual scatter for the canonical run.

Headline metric: Spearman ρ on rubric items. R² is reported but not the
primary claim because of small n (52–60 per run) and heavy-left-skew score
distributions on most runs.

Usage:
    python scripts/08_score_regression.py --run frontierscience/deepseek/deepseek-v3.2_reasoning-auto
    python scripts/08_score_regression.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from semanticscale.sh6.failure_analysis_runner import run_models_on_run
from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)

# Runs with both trajectory_features.csv on disk and rubric items in
# traces.jsonl. The 1.5B distill is excluded (no rubric items).
ELIGIBLE_RUNS: list[str] = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1",
    "frontierscience/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none",
    "frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto",
    "frontierscience/R1-Distill-32B-cloudjudge_reasoning-auto",
]

CANONICAL_RUN = "frontierscience/deepseek/deepseek-v3.2_reasoning-auto"

REGRESSION_MODELS = ["ridge", "lightgbm_reg"]
LENGTH_ONLY_FEATURES = ["reasoning_n_chunks", "answer_n_chunks", "total_n_chunks"]

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "experiments/sh6_llm-pairwise-slod/reports"
DATA_ROOT = REPO_ROOT / "data/sh6"
CROSS_DATASET_DIR = REPORTS_ROOT / "_cross_dataset"

BOOTSTRAP_DRAWS = 1000
RANDOM_STATE = 42


@dataclass
class RunPaths:
    run: str
    features_csv: Path
    traces_jsonl: Path
    artifact_dir: Path
    report_dir: Path

    @classmethod
    def for_run(cls, run: str) -> "RunPaths":
        report_dir = REPORTS_ROOT / run
        return cls(
            run=run,
            features_csv=report_dir / "trajectory_features.csv",
            traces_jsonl=DATA_ROOT / run / "traces.jsonl",
            artifact_dir=report_dir / "artifacts",
            report_dir=report_dir,
        )


def _backfill_rubric_score(paths: RunPaths) -> int:
    """Add a ``rubric_score`` column to trajectory_features.csv if missing.

    Idempotent: re-runs do nothing once the column is populated. Returns the
    number of rows that received a rubric score.
    """
    df = pd.read_csv(paths.features_csv)
    if "rubric_score" in df.columns and df["rubric_score"].notna().any():
        n = int(df["rubric_score"].notna().sum())
        logger.info("[%s] rubric_score already present (%d rows)", paths.run, n)
        return n

    if not paths.traces_jsonl.exists():
        msg = f"traces.jsonl not found at {paths.traces_jsonl}"
        raise FileNotFoundError(msg)

    score_by_id: dict[str, float] = {}
    with paths.traces_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            grade = rec.get("grade") or {}
            if grade.get("type") != "rubric":
                continue
            total_max = grade.get("total_max") or 0.0
            if total_max <= 0:
                continue
            score_by_id[rec["id"]] = float(grade["total_awarded"]) / float(total_max)

    df["rubric_score"] = df["id"].map(score_by_id)
    df.to_csv(paths.features_csv, index=False)
    n = int(df["rubric_score"].notna().sum())
    logger.info("[%s] backfilled rubric_score for %d items", paths.run, n)
    return n


def _bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    fn,
    n_draws: int = BOOTSTRAP_DRAWS,
    rng_seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on a (y, pred)-valued metric. Returns (point, lo, hi)."""
    point = float(fn(y, pred))
    if len(y) < 4:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n_draws, dtype=float)
    n = len(y)
    for i in range(n_draws):
        idx = rng.integers(0, n, size=n)
        try:
            draws[i] = fn(y[idx], pred[idx])
        except ValueError:
            draws[i] = np.nan
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return point, float(lo), float(hi)


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    rho, _ = spearmanr(y, p)
    return float(rho) if np.isfinite(rho) else 0.0


def _pooled_metrics(oof_path: Path) -> dict:
    """Read an oof_regression parquet and compute pooled R²/ρ with bootstrap CI."""
    df = pd.read_parquet(oof_path)
    y = df["score"].to_numpy(dtype=float)
    p = df["pred"].to_numpy(dtype=float)
    r2 = _bootstrap_ci(y, p, r2_score)
    rho = _bootstrap_ci(y, p, _spearman)
    mae = _bootstrap_ci(y, p, mean_absolute_error)
    return {
        "n": int(len(y)),
        "r2": {"point": r2[0], "lo": r2[1], "hi": r2[2]},
        "spearman": {"point": rho[0], "lo": rho[1], "hi": rho[2]},
        "mae": {"point": mae[0], "lo": mae[1], "hi": mae[2]},
    }


def _run_one(paths: RunPaths) -> dict:
    n_rubric = _backfill_rubric_score(paths)
    if n_rubric < 10:
        logger.warning("[%s] only %d rubric items; skipping", paths.run, n_rubric)
        return {"run": paths.run, "skipped": True, "reason": f"only {n_rubric} rubric items"}

    summaries: dict[str, dict] = {}

    summaries["trajectory_full"] = run_models_on_run(
        paths.features_csv,
        paths.artifact_dir,
        REGRESSION_MODELS,
        feature_set="trajectory_full",
        target_label="rubric_score",
        task_type="regression",
        cv_folds=5,
        random_state=RANDOM_STATE,
        repo_root=REPO_ROOT,
    )

    # Length-only baseline. Same models, but features restricted to chunk
    # counts. We only run the linear baseline here — a regularized linear
    # model on 3 features is the right floor; LightGBM on 3 features would
    # add little.
    summaries["length_only"] = run_models_on_run(
        paths.features_csv,
        paths.artifact_dir / "length_only",
        ["ridge"],
        feature_cols_override=LENGTH_ONLY_FEATURES,
        target_label="rubric_score",
        task_type="regression",
        cv_folds=5,
        random_state=RANDOM_STATE,
        repo_root=REPO_ROOT,
    )

    pooled = {}
    for model in REGRESSION_MODELS:
        oof = paths.artifact_dir / f"oof_regression_{model}.parquet"
        if oof.exists():
            pooled[model] = _pooled_metrics(oof)
    length_oof = paths.artifact_dir / "length_only/oof_regression_ridge.parquet"
    if length_oof.exists():
        pooled["length_only_ridge"] = _pooled_metrics(length_oof)

    _write_run_report(paths, n_rubric, summaries, pooled)
    return {"run": paths.run, "n_rubric": n_rubric, "summaries": summaries, "pooled": pooled}


def _fmt_ci(d: dict) -> str:
    if d["lo"] != d["lo"] or d["hi"] != d["hi"]:  # NaN
        return f"{d['point']:+.3f}"
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def _write_run_report(paths: RunPaths, n_rubric: int, summaries: dict, pooled: dict) -> None:
    md = [
        f"# Score Regression — `{paths.run}`",
        "",
        f"**Target:** `rubric_score` ∈ [0, 1] (rubric items only, n = {n_rubric}).",
        "",
        f"**CV:** KFold(5, shuffle=True, random_state={RANDOM_STATE}). "
        f"Bootstrap: {BOOTSTRAP_DRAWS} draws, percentile CI.",
        "",
        "## Pooled OOF metrics",
        "",
        "| Model | n | R² [95% CI] | Spearman ρ [95% CI] | MAE [95% CI] |",
        "|---|---:|---|---|---|",
    ]
    rows = []
    for model in REGRESSION_MODELS:
        if model in pooled:
            m = pooled[model]
            rows.append(
                f"| `{model}` (trajectory_full) | {m['n']} | "
                f"{_fmt_ci(m['r2'])} | {_fmt_ci(m['spearman'])} | {_fmt_ci(m['mae'])} |"
            )
    if "length_only_ridge" in pooled:
        m = pooled["length_only_ridge"]
        rows.append(
            f"| `ridge` (length_only) | {m['n']} | "
            f"{_fmt_ci(m['r2'])} | {_fmt_ci(m['spearman'])} | {_fmt_ci(m['mae'])} |"
        )
    md += rows + [""]

    md += [
        "## Per-fold metrics (mean ± std)",
        "",
        "| Model | R² | MAE | Spearman ρ |",
        "|---|---|---|---|",
    ]
    for fset, summary in summaries.items():
        for model, info in summary.items():
            if model.startswith("_"):
                continue
            md.append(
                f"| `{model}` ({fset}) | "
                f"{info['r2_mean']:+.3f} ± {info['r2_std']:.3f} | — | "
                f"{info['spearman_mean']:+.3f} ± {info['spearman_std']:.3f} |"
            )
    md += [""]

    md += [
        "## Reading this report",
        "",
        f"- Headline metric is Spearman ρ. With n = {n_rubric}, bootstrap CIs are wide; "
        "treat any model whose ρ-CI overlaps 0 as null on this run.",
        "- The length_only ridge baseline says how much of the signal is just "
        "\"longer answers tend to score higher / lower\". Trajectory shape only "
        "matters if ρ_trajectory > ρ_length_only (with non-overlapping CI).",
        "- R² is shown for completeness but is unstable on small left-skewed "
        "distributions; a near-zero or negative R² with a positive Spearman ρ "
        "means the model recovers ranks but not magnitudes.",
        "",
        "Artifacts in `artifacts/`: "
        "`oof_regression_{ridge,lightgbm_reg}.parquet`, "
        "`regression_metadata_*.json`, "
        "`regression_importance_*.csv`, "
        "`length_only/oof_regression_ridge.parquet`.",
    ]

    out = paths.report_dir / "score_regression.md"
    out.write_text("\n".join(md) + "\n")
    logger.info("[%s] wrote %s", paths.run, out)


def _write_cross_summary(results: list[dict]) -> None:
    CROSS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        if r.get("skipped"):
            continue
        run = r["run"]
        n = r["n_rubric"]
        p = r["pooled"]
        rho_tf = p.get("lightgbm_reg", {}).get("spearman", {})
        rho_rg = p.get("ridge", {}).get("spearman", {})
        rho_lo = p.get("length_only_ridge", {}).get("spearman", {})
        r2_tf = p.get("lightgbm_reg", {}).get("r2", {})
        r2_rg = p.get("ridge", {}).get("r2", {})
        rows.append(
            {
                "run": run,
                "n_rubric": n,
                "ridge_rho": rho_rg.get("point"),
                "ridge_rho_lo": rho_rg.get("lo"),
                "ridge_rho_hi": rho_rg.get("hi"),
                "ridge_r2": r2_rg.get("point"),
                "lightgbm_reg_rho": rho_tf.get("point"),
                "lightgbm_reg_rho_lo": rho_tf.get("lo"),
                "lightgbm_reg_rho_hi": rho_tf.get("hi"),
                "lightgbm_reg_r2": r2_tf.get("point"),
                "length_only_rho": rho_lo.get("point"),
                "length_only_rho_lo": rho_lo.get("lo"),
                "length_only_rho_hi": rho_lo.get("hi"),
            }
        )
    df = pd.DataFrame(rows)
    csv_path = CROSS_DATASET_DIR / "score_regression_summary.csv"
    df.to_csv(csv_path, index=False)

    md = [
        "# FrontierScience Score Regression — Cross-Run Summary",
        "",
        "Headline metric: pooled OOF Spearman ρ between predicted and actual "
        "`rubric_score`, with 95% percentile bootstrap CI on the rubric subset "
        "of each run.",
        "",
        "| Run | n | ridge ρ [CI] | lightgbm_reg ρ [CI] | length_only ρ [CI] | "
        "ridge R² | lightgbm_reg R² |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for r in rows:

        def fmt(point, lo, hi):
            if point is None or point != point:
                return "—"
            if lo is None or lo != lo:
                return f"{point:+.3f}"
            return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"

        md.append(
            "| `{run}` | {n} | {rg} | {tf} | {lo} | {r2rg} | {r2tf} |".format(
                run=r["run"].replace("frontierscience/", ""),
                n=r["n_rubric"],
                rg=fmt(r["ridge_rho"], r["ridge_rho_lo"], r["ridge_rho_hi"]),
                tf=fmt(r["lightgbm_reg_rho"], r["lightgbm_reg_rho_lo"], r["lightgbm_reg_rho_hi"]),
                lo=fmt(r["length_only_rho"], r["length_only_rho_lo"], r["length_only_rho_hi"]),
                r2rg=f"{r['ridge_r2']:+.3f}" if r["ridge_r2"] is not None else "—",
                r2tf=f"{r['lightgbm_reg_r2']:+.3f}" if r["lightgbm_reg_r2"] is not None else "—",
            )
        )
    md += [
        "",
        "## How to read this",
        "",
        "- `ridge` and `lightgbm_reg` use the full `trajectory_full` feature set "
        "(same predictors as the classifier path, just regressing rubric_score "
        "instead of classifying is_correct).",
        "- `length_only` is the ridge baseline on `{reasoning,answer,total}_n_chunks` "
        "only. If the trajectory ρ is not materially above the length-only ρ, the "
        "score signal is mostly answer-length, not SLoD shape.",
        "- ρ-CIs are wide because per-run n ≈ 55. Read the table by direction "
        "of effect across runs, not by any single CI.",
        "- `_s1` / `_types-research` seeds and the s2/s3/s4 reseeds will be added "
        "once `02_slod.py` is run on those configs (currently only s1 has features "
        "extracted).",
        "",
        f"Cross-run CSV: `score_regression_summary.csv`. JSON dump of pooled "
        f"metrics: `score_regression_summary.json`. Predicted-vs-actual scatter "
        f"for the canonical run (`{CANONICAL_RUN.replace('frontierscience/','')}`): "
        f"`score_regression_canonical_scatter.png`.",
    ]
    md_path = CROSS_DATASET_DIR / "score_regression_summary.md"
    md_path.write_text("\n".join(md) + "\n")

    json_path = CROSS_DATASET_DIR / "score_regression_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    logger.info("Cross-run summary: %s", md_path)
    _write_canonical_scatter(results)


def _write_canonical_scatter(results: list[dict]) -> None:
    canonical = next((r for r in results if r["run"] == CANONICAL_RUN), None)
    if not canonical or canonical.get("skipped"):
        return
    paths = RunPaths.for_run(CANONICAL_RUN)
    oof = paths.artifact_dir / "oof_regression_lightgbm_reg.parquet"
    if not oof.exists():
        return
    df = pd.read_parquet(oof)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["score"], df["pred"], alpha=0.6, s=24)
    lo = float(min(df["score"].min(), df["pred"].min()))
    hi = float(max(df["score"].max(), df["pred"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="y = x")
    rho, _ = spearmanr(df["score"], df["pred"])
    r2 = r2_score(df["score"], df["pred"])
    ax.set_xlabel("Actual rubric_score")
    ax.set_ylabel("Predicted (OOF, lightgbm_reg)")
    ax.set_title(
        f"{CANONICAL_RUN.replace('frontierscience/','')}\n"
        f"n={len(df)}  ρ={rho:+.3f}  R²={r2:+.3f}"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = CROSS_DATASET_DIR / "score_regression_canonical_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run path under reports/ (e.g. frontierscience/deepseek/...). May be repeated.",
    )
    p.add_argument("--all", action="store_true", help="Run on every eligible run.")
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.all and args.run:
        logger.error("--all and --run are mutually exclusive")
        return 2
    runs = ELIGIBLE_RUNS if args.all else (args.run or [CANONICAL_RUN])

    results: list[dict] = []
    for run in runs:
        paths = RunPaths.for_run(run)
        if not paths.features_csv.exists():
            logger.warning("[%s] missing %s; skipping", run, paths.features_csv)
            continue
        try:
            results.append(_run_one(paths))
        except Exception:
            logger.exception("[%s] regression failed", run)
            results.append({"run": run, "skipped": True, "reason": "exception"})

    if len(results) > 1:
        _write_cross_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
