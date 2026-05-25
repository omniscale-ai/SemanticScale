#!/usr/bin/env python
"""SH6 — Stage 12: FrontierScience Olympiad LLM self-verification baseline.

For each of the five same-config DeepSeek runs on the FrontierScience Olympiad
subset, ask the same DeepSeek-v3.2 model to review every produced solution
(problem + reasoning + final answer, *without* the reference answer) and emit a
binary verdict plus a 0-100 confidence. The signed score (+confidence when the
verifier says "correct", -confidence otherwise) is used as the best-of-5
selection signal.

This is the LLM-as-judge analogue of the two existing best-of-5 baselines:

* ``09_frontierscience_bestof5.py``      — SLoD-derived confidence
* ``11_frontierscience_self_consistency.py`` — plurality vote across the 5 attempts

All three select among the same five attempts per problem. Comparing them tells
us whether SLoD adds signal beyond simple inter-sample agreement, and whether a
post-hoc LLM self-check is as good as either.

Outputs (under reports/_cross_dataset/):
    frontierscience_deepseek_olympiad_llm_verify.md
    frontierscience_deepseek_olympiad_llm_verify.json
    frontierscience_deepseek_olympiad_llm_verify_attempts.csv
    frontierscience_deepseek_olympiad_llm_verify_selected.csv

Per-run verifier outputs are cached at::

    data/sh6/<run>/llm_verify.jsonl

so re-runs are cheap and resume-safe.

Usage:
    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/12_frontierscience_llm_verify.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from semanticscale.sh6.self_verification import (
    VerifierCache,
    signed_score,
    verify_attempts,
)
from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_RUNS: list[str] = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s2",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s3",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s4",
]
ORIGIN = "olympiad"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "config/frontierscience-deepseek.yaml"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "experiments/sh6_llm-pairwise-slod/reports"
DATA_ROOT = REPO_ROOT / "data/sh6"
CROSS_DATASET_DIR = REPORTS_ROOT / "_cross_dataset"

OUTPUT_STEM = "frontierscience_deepseek_olympiad_llm_verify"


@dataclass(frozen=True)
class RunPaths:
    run: str
    traces_jsonl: Path
    cache_jsonl: Path

    @classmethod
    def for_run(cls, run: str) -> "RunPaths":
        return cls(
            run=run,
            traces_jsonl=DATA_ROOT / run / "traces.jsonl",
            cache_jsonl=DATA_ROOT / run / "llm_verify.jsonl",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Full run path under data/sh6/. May be repeated. Defaults to the 5 deepseek-v3.2 seeded runs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config providing the verifier model (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Concurrent verifier calls (default: 20).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip verifier calls; only re-analyze cached outputs.",
    )
    return parser.parse_args()


def _run_short_slug(run: str) -> str:
    prefix = "frontierscience/"
    return run[len(prefix):] if run.startswith(prefix) else run


def _pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
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
            rows.append(
                {
                    "id": str(rec["id"]),
                    "subject": str(rec.get("subject", "unknown")),
                    "run": paths.run,
                    "run_slug_short": _run_short_slug(paths.run),
                    "run_order": run_order,
                    "answered": bool(rec.get("predicted_answer")) and not bool(rec.get("error")),
                    "is_correct": bool(rec.get("is_correct")),
                    "problem": rec.get("problem") or "",
                    "reasoning_text": rec.get("reasoning_text") or "",
                    "answer_text": rec.get("answer_text") or "",
                    "predicted_answer": rec.get("predicted_answer") or "",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No Olympiad rows found in {paths.traces_jsonl}")
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(f"Duplicate Olympiad ids in {paths.traces_jsonl}: {dupes}")
    return df.sort_values("id", ignore_index=True)


def _validate_problem_alignment(run_frames: list[pd.DataFrame]) -> None:
    base = run_frames[0][["id", "subject"]].rename(columns={"subject": "subject_base"})
    base_ids = set(base["id"])
    for frame in run_frames[1:]:
        ids = set(frame["id"])
        if ids != base_ids:
            only_here = sorted(ids - base_ids)[:5]
            only_base = sorted(base_ids - ids)[:5]
            raise ValueError(
                f"Olympiad id mismatch across runs: only_in_current={only_here}, only_in_base={only_base}"
            )
        merged = frame[["id", "subject"]].merge(base, on="id", how="inner")
        mismatch = merged.loc[merged["subject"] != merged["subject_base"]]
        if not mismatch.empty:
            row = mismatch.iloc[0]
            raise ValueError(
                f"Subject mismatch for id={row['id']}: {row['subject']} vs {row['subject_base']}"
            )


def _run_verifier(
    df: pd.DataFrame,
    paths: RunPaths,
    *,
    model_cfg: dict,
    max_concurrent: int,
    dry_run: bool,
) -> pd.DataFrame:
    cache = VerifierCache(paths.cache_jsonl)
    cache.load()
    attempts: list[dict] = df.to_dict("records")

    if dry_run:
        # Apply cached values only, without calling the model.
        async def _noop() -> list[dict]:
            return await verify_attempts(
                attempts=[],
                model_cfg=model_cfg,
                cache=cache,
                max_concurrent=max_concurrent,
            )

        asyncio.run(_noop())
        enriched: list[dict] = []
        for att in attempts:
            out = dict(att)
            rec = cache.get(att["id"])
            if rec and rec.get("verdict") is not None and rec.get("confidence") is not None:
                out["verify_verdict"] = rec["verdict"]
                out["verify_confidence"] = rec["confidence"]
                out["verify_signed_score"] = signed_score(rec["verdict"], rec["confidence"])
                out["verify_rationale"] = rec.get("rationale", "")
                out["verify_error"] = None
            else:
                out["verify_verdict"] = None
                out["verify_confidence"] = None
                out["verify_signed_score"] = None
                out["verify_rationale"] = ""
                out["verify_error"] = "dry-run: no cache entry"
            enriched.append(out)
    else:
        enriched = asyncio.run(
            verify_attempts(
                attempts=attempts,
                model_cfg=model_cfg,
                cache=cache,
                max_concurrent=max_concurrent,
            )
        )
    return pd.DataFrame(enriched)


def _select_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    ranked = attempts.copy()
    ranked["_has_score"] = ranked["verify_signed_score"].notna()
    ranked["_score_sort"] = ranked["verify_signed_score"].fillna(-np.inf)
    ranked = ranked.sort_values(
        ["id", "_has_score", "_score_sort", "answered", "run_order"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    selected = ranked.groupby("id", as_index=False).head(1).copy()
    return selected.drop(columns=["_has_score", "_score_sort"])


def _per_run_metrics(attempts: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for run, sub in attempts.groupby("run"):
        verdicts = sub.dropna(subset=["verify_verdict", "verify_confidence"])
        n_scored = len(verdicts)
        n_err = int(sub["verify_verdict"].isna().sum())
        # Verifier as classifier of is_correct (label-free AUROC on signed score).
        auc = float("nan")
        if n_scored >= 2 and verdicts["is_correct"].nunique() == 2:
            auc = float(
                roc_auc_score(
                    verdicts["is_correct"].astype(int),
                    verdicts["verify_signed_score"].astype(float),
                )
            )
        verdict_correct = (verdicts["verify_verdict"] == "correct").astype(int)
        labels = verdicts["is_correct"].astype(int)
        agreement = float((verdict_correct == labels).mean()) if n_scored else float("nan")
        pred_correct_rate = float(verdict_correct.mean()) if n_scored else float("nan")
        true_correct_rate = float(labels.mean()) if n_scored else float("nan")
        rows.append(
            {
                "run": run,
                "run_slug_short": _run_short_slug(run),
                "n_attempts": int(len(sub)),
                "n_verified": int(n_scored),
                "n_verify_errors": int(n_err),
                "pass1": float(sub["is_correct"].mean()),
                "verifier_pred_correct_rate": pred_correct_rate,
                "true_correct_rate_on_verified": true_correct_rate,
                "verifier_label_agreement": agreement,
                "verifier_oof_auc": auc,
                "mean_confidence": float(verdicts["verify_confidence"].mean()) if n_scored else float("nan"),
            }
        )
    rows.sort(key=lambda r: r["run_slug_short"])
    return rows


def _per_subject(selected: pd.DataFrame, attempts: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for subject, sub in selected.groupby("subject"):
        sub_ids = set(sub["id"])
        sub_attempts = attempts[attempts["id"].isin(sub_ids)]
        pass5 = float(sub_attempts.groupby("id")["is_correct"].max().mean())
        rows.append(
            {
                "subject": subject,
                "n_problems": int(len(sub)),
                "avg_pass1": float(sub_attempts["is_correct"].mean()),
                "pass5": pass5,
                "verify_pass1": float(sub["is_correct"].mean()),
                "mean_selected_confidence": float(sub["verify_confidence"].mean()),
            }
        )
    rows.sort(key=lambda r: r["subject"])
    return rows


def _selective_curve(selected: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for thr in [0, 25, 50, 75, 90]:
        scored = selected.dropna(subset=["verify_confidence"]).copy()
        mask = scored["verify_signed_score"] >= thr
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "min_signed_score": thr,
                    "coverage": 0.0,
                    "n_problems": 0,
                    "selective_pass1": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "min_signed_score": thr,
                "coverage": float(n) / float(len(selected)),
                "n_problems": n,
                "selective_pass1": float(scored.loc[mask, "is_correct"].mean()),
            }
        )
    return rows


def _write_outputs(
    *,
    runs: list[str],
    model_name: str,
    attempts: pd.DataFrame,
    selected: pd.DataFrame,
    per_run: list[dict],
    per_subject: list[dict],
    selective: list[dict],
    baseline_pass1: float,
    pass5: float,
    verify_pass1: float,
) -> None:
    CROSS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    attempts_out = CROSS_DATASET_DIR / f"{OUTPUT_STEM}_attempts.csv"
    selected_out = CROSS_DATASET_DIR / f"{OUTPUT_STEM}_selected.csv"
    summary_json = CROSS_DATASET_DIR / f"{OUTPUT_STEM}.json"
    summary_md = CROSS_DATASET_DIR / f"{OUTPUT_STEM}.md"

    # Drop heavy text columns before writing CSVs.
    drop_cols = [
        c for c in ("problem", "reasoning_text", "answer_text", "verify_rationale")
        if c in attempts.columns
    ]
    attempts.drop(columns=drop_cols, errors="ignore").to_csv(attempts_out, index=False)
    selected.drop(columns=drop_cols, errors="ignore").to_csv(selected_out, index=False)

    delta = verify_pass1 - baseline_pass1
    oracle_gap = pass5 - baseline_pass1
    gap_recovered = (delta / oracle_gap) if oracle_gap > 0 else float("nan")

    payload = {
        "subset": ORIGIN,
        "runs": runs,
        "verifier_model": model_name,
        "n_problems": int(selected.shape[0]),
        "n_runs": len(runs),
        "baseline_pass1_avg_per_attempt": baseline_pass1,
        "pass5": pass5,
        "verify_pass1": verify_pass1,
        "delta_vs_avg_pass1": delta,
        "oracle_gap_recovered": gap_recovered,
        "per_run": per_run,
        "per_subject": per_subject,
        "selective_curve": selective,
        "attempts_csv": str(attempts_out),
        "selected_csv": str(selected_out),
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# FrontierScience DeepSeek Olympiad — LLM Self-Verification Baseline",
        "",
        "Each of the five same-config DeepSeek runs is re-graded by the same model",
        "(without access to the reference answer). The verifier sees the problem,",
        "the reasoning trace, and the final answer, and returns a `verdict` plus a",
        "0-100 `confidence`. Per problem, the best-of-5 attempt is the one with the",
        "largest **signed** confidence (`+conf` if verdict=correct, `-conf` otherwise).",
        "",
        "Compare against:",
        "",
        "- `frontierscience_deepseek_olympiad_self_consistency.md` (no scorer, plurality vote)",
        "- `frontierscience_deepseek_olympiad_bestof5.md` (SLoD-derived confidence)",
        "",
        "## Setup",
        "",
        f"- Runs: {len(runs)}",
        f"- Subset: `{ORIGIN}` (filter: `has_final_answer == true`)",
        f"- Verifier model: `{model_name}`",
        "- Verifier sees: problem + student reasoning + student final answer (no reference answer)",
        "- Best-of-5 picks highest signed confidence; unscored attempts rank last (deterministic fallback).",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Average-per-attempt Pass@1 | {_pct(baseline_pass1)} |",
        f"| Pass@5 (oracle across the five attempts) | {_pct(pass5)} |",
        f"| **LLM-verify Pass@1** | **{_pct(verify_pass1)}** |",
        f"| Gain over avg Pass@1 | {delta:+.3f} ({100.0 * delta:+.1f} pts) |",
        f"| Oracle gap recovered | {_pct(gap_recovered)} |",
        "",
        "## Per-run verifier diagnostics",
        "",
        "| Run | Attempts | Verified | Errors | Pass@1 | Verifier 'correct' rate | Label agreement | OOF AUC | Mean conf |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    for row in per_run:
        lines.append(
            f"| `{row['run_slug_short']}` | {row['n_attempts']} | {row['n_verified']} | "
            f"{row['n_verify_errors']} | {_pct(row['pass1'])} | "
            f"{_pct(row['verifier_pred_correct_rate'])} | "
            f"{_pct(row['verifier_label_agreement'])} | "
            f"{row['verifier_oof_auc']:.3f} | {row['mean_confidence']:.1f} |"
        )

    lines += [
        "",
        "## Per-subject breakdown",
        "",
        "| Subject | Problems | Avg Pass@1 | Pass@5 | Verify Pass@1 | Mean selected conf |",
        "|---|---:|---|---|---|---:|",
    ]
    for row in per_subject:
        lines.append(
            f"| {row['subject']} | {row['n_problems']} | {_pct(row['avg_pass1'])} | "
            f"{_pct(row['pass5'])} | {_pct(row['verify_pass1'])} | "
            f"{row['mean_selected_confidence']:.1f} |"
        )

    lines += [
        "",
        "## Selective prediction by verifier signed-score threshold",
        "",
        "Restrict the best-of-5 selection to problems whose chosen attempt cleared the",
        "given **signed** confidence threshold (`+conf` if verdict=correct, `-conf` if",
        "verdict=incorrect). A threshold of 0 keeps all problems whose selected attempt",
        "was judged correct.",
        "",
        "| Min signed score | Coverage | Problems | Selective Pass@1 |",
        "|---:|---|---:|---|",
    ]
    for row in selective:
        lines.append(
            f"| ≥{row['min_signed_score']} | {_pct(row['coverage'])} | "
            f"{row['n_problems']} | {_pct(row['selective_pass1'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **LLM self-verification** is a simple, prompt-only baseline: no SLoD signal,",
        "  no inter-sample voting, just ask the same model whether each solution looks",
        "  right. Performance above the average per-attempt Pass@1 means the verifier",
        "  is calibrated enough to discriminate its own correct vs. incorrect attempts.",
        "- Compare directly to `frontierscience_deepseek_olympiad_bestof5.md` and",
        "  `frontierscience_deepseek_olympiad_self_consistency.md`: all three select",
        "  one of the same five attempts per problem, so deltas are apples-to-apples.",
        "",
        f"Artifacts: `{attempts_out.name}`, `{selected_out.name}`, `{summary_json.name}`.",
    ]
    summary_md.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", summary_md)


def main() -> int:
    setup_logging()
    args = parse_args()
    runs = args.run or list(DEFAULT_RUNS)
    config = load_config(args.config)
    grader_cfg = config.get("grader") or config.get("pairwise_slod") or {}
    model_cfg = grader_cfg.get("model")
    if not model_cfg or "name" not in model_cfg:
        raise ValueError(
            f"No grader.model.name in {args.config}; cannot pick a verifier model."
        )
    logger.info(
        "Verifier model: %s (base_url=%s)", model_cfg["name"], model_cfg.get("base_url")
    )

    attempt_frames: list[pd.DataFrame] = []
    for run_order, run in enumerate(runs):
        paths = RunPaths.for_run(run)
        df = _load_olympiad_attempts(paths, run_order=run_order)
        logger.info(
            "[%s] %d Olympiad attempts loaded (%d already cached)",
            _run_short_slug(run),
            len(df),
            sum(paths.cache_jsonl.exists() for _ in [0]),
        )
        enriched = _run_verifier(
            df,
            paths,
            model_cfg=model_cfg,
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run,
        )
        attempt_frames.append(enriched)

    _validate_problem_alignment(attempt_frames)
    attempts = pd.concat(attempt_frames, ignore_index=True)
    attempts = attempts.sort_values(["id", "run_order"], ignore_index=True)

    baseline_pass1 = float(attempts["is_correct"].mean())
    pass5 = float(attempts.groupby("id")["is_correct"].max().mean())

    selected = _select_attempts(attempts)
    verify_pass1 = float(selected["is_correct"].mean())

    per_run = _per_run_metrics(attempts)
    per_subject = _per_subject(selected, attempts)
    selective = _selective_curve(selected)

    logger.info(
        "n_problems=%d  avg_pass1=%.3f  pass5=%.3f  verify_pass1=%.3f",
        len(selected), baseline_pass1, pass5, verify_pass1,
    )

    _write_outputs(
        runs=runs,
        model_name=model_cfg["name"],
        attempts=attempts,
        selected=selected,
        per_run=per_run,
        per_subject=per_subject,
        selective=selective,
        baseline_pass1=baseline_pass1,
        pass5=pass5,
        verify_pass1=verify_pass1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
