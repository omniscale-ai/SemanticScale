#!/usr/bin/env python
"""SH6 — Stage 11: FrontierScience DeepSeek Olympiad self-consistency baseline.

Self-consistency (Wang et al., 2022) over five same-config DeepSeek runs on the
FrontierScience Olympiad subset. For each problem we collect the five
``predicted_answer`` strings, normalize them, and take a plurality vote. The
plurality answer is graded by reusing the per-sample ``is_correct`` label, which
the upstream grader already computed against ``correct_answer``.

This is the canonical baseline to compare against the SLoD-based best-of-5
reranking in ``09_frontierscience_bestof5.py``: both pick one of the same five
attempts per problem, but self-consistency uses inter-sample answer agreement
while best-of-5 uses an SLoD-derived confidence score.

Outputs:
    reports/_cross_dataset/frontierscience_deepseek_olympiad_self_consistency.md
    reports/_cross_dataset/frontierscience_deepseek_olympiad_self_consistency.json
    reports/_cross_dataset/frontierscience_deepseek_olympiad_self_consistency_attempts.csv
    reports/_cross_dataset/frontierscience_deepseek_olympiad_self_consistency_votes.csv

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/11_frontierscience_self_consistency.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_RUNS: list[str] = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s2",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s3",
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s4",
]
ORIGIN = "olympiad"

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "experiments/sh6_llm-pairwise-slod/reports"
DATA_ROOT = REPO_ROOT / "data/sh6"
CROSS_DATASET_DIR = REPORTS_ROOT / "_cross_dataset"

OUTPUT_STEM = "frontierscience_deepseek_olympiad_self_consistency"

_WS_RE = re.compile(r"\s+")
_TRAIL_PUNCT_RE = re.compile(r"[.,;:!?\"'`)\]\}]+$")
_LEAD_PUNCT_RE = re.compile(r"^[\"'`(\[\{]+")


@dataclass(frozen=True)
class RunPaths:
    run: str
    traces_jsonl: Path

    @classmethod
    def for_run(cls, run: str) -> "RunPaths":
        return cls(run=run, traces_jsonl=DATA_ROOT / run / "traces.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Full run path under data/sh6/. May be repeated. Defaults to the 5 deepseek-v3.2 seeded runs.",
    )
    return parser.parse_args()


def _run_short_slug(run: str) -> str:
    prefix = "frontierscience/"
    return run[len(prefix):] if run.startswith(prefix) else run


def _pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _normalize_answer(text: str | None) -> str:
    """Light normalization for plurality voting on free-form answers.

    Lower-cases, NFKC-normalizes, collapses whitespace, and strips wrapping
    punctuation. Intentionally conservative — heavy chemistry/physics
    canonicalization would bias the baseline.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    s = _WS_RE.sub(" ", s)
    s = _LEAD_PUNCT_RE.sub("", s)
    s = _TRAIL_PUNCT_RE.sub("", s)
    return s.strip()


def _load_olympiad_attempts(paths: RunPaths, run_order: int) -> pd.DataFrame:
    if not paths.traces_jsonl.exists():
        raise FileNotFoundError(f"traces.jsonl not found at {paths.traces_jsonl}")

    rows: list[dict] = []
    with paths.traces_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("has_final_answer"):
                continue
            pa_raw = rec.get("predicted_answer")
            rows.append(
                {
                    "id": str(rec["id"]),
                    "subject": str(rec.get("subject", "unknown")),
                    "run": paths.run,
                    "run_slug_short": _run_short_slug(paths.run),
                    "run_order": run_order,
                    "answered": bool(pa_raw) and not bool(rec.get("error")),
                    "predicted_answer": pa_raw or "",
                    "predicted_norm": _normalize_answer(pa_raw),
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


def _self_consistency_vote(group: pd.DataFrame, n_runs: int) -> pd.Series:
    """Plurality vote across the per-problem attempts.

    Voting rule:
    - Only ``answered`` attempts with a non-empty normalized answer vote.
    - Ties broken by (vote count desc, earliest run_order of the winning group).
      We do not break ties on ``is_correct`` to keep voting label-free.
    - SC-correct = majority ``is_correct`` among the attempts that share the
      winning normalized answer.
    """
    eligible = group[group["answered"] & (group["predicted_norm"] != "")]
    n_answered = int(eligible.shape[0])
    if n_answered == 0:
        return pd.Series(
            {
                "subject": group["subject"].iloc[0],
                "n_attempts": n_runs,
                "n_answered": 0,
                "winning_norm": "",
                "winning_count": 0,
                "n_distinct_answers": 0,
                "consistency": 0.0,
                "sc_correct": False,
                "any_correct": bool(group["is_correct"].any()),
                "all_correct": bool(group["is_correct"].all()),
                "mean_correct": float(group["is_correct"].mean()),
            }
        )

    counts = Counter(eligible["predicted_norm"].tolist())
    max_count = max(counts.values())
    candidates = [ans for ans, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        winning = candidates[0]
    else:
        # Break ties by the earliest run_order at which a candidate appeared.
        first_seen = (
            eligible[eligible["predicted_norm"].isin(candidates)]
            .groupby("predicted_norm")["run_order"]
            .min()
        )
        winning = first_seen.idxmin()

    winning_rows = eligible[eligible["predicted_norm"] == winning]
    sc_correct = bool(winning_rows["is_correct"].mean() >= 0.5)
    return pd.Series(
        {
            "subject": group["subject"].iloc[0],
            "n_attempts": n_runs,
            "n_answered": n_answered,
            "winning_norm": winning,
            "winning_count": int(max_count),
            "n_distinct_answers": int(len(counts)),
            "consistency": float(max_count) / float(n_runs),
            "sc_correct": sc_correct,
            "any_correct": bool(group["is_correct"].any()),
            "all_correct": bool(group["is_correct"].all()),
            "mean_correct": float(group["is_correct"].mean()),
        }
    )


def _selective_curve(votes: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for thr in sorted({1, 2, 3, 4, 5}):
        mask = votes["winning_count"] >= thr
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "min_winning_count": thr,
                    "coverage": 0.0,
                    "n_problems": 0,
                    "sc_pass1": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "min_winning_count": thr,
                "coverage": float(n) / float(len(votes)),
                "n_problems": n,
                "sc_pass1": float(votes.loc[mask, "sc_correct"].mean()),
            }
        )
    return rows


def _per_subject(votes: pd.DataFrame, attempts: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for subject, sub in votes.groupby("subject"):
        sub_ids = set(sub["id"]) if "id" in sub.columns else set(sub.index)
        sub_attempts = attempts[attempts["id"].isin(sub_ids)]
        pass5 = float(sub_attempts.groupby("id")["is_correct"].max().mean())
        rows.append(
            {
                "subject": subject,
                "n_problems": int(len(sub)),
                "avg_pass1": float(sub_attempts["is_correct"].mean()),
                "pass5": pass5,
                "sc_pass1": float(sub["sc_correct"].mean()),
                "mean_consistency": float(sub["consistency"].mean()),
            }
        )
    rows.sort(key=lambda r: r["subject"])
    return rows


def _write_outputs(
    *,
    runs: list[str],
    attempts: pd.DataFrame,
    votes: pd.DataFrame,
    per_run: list[dict],
    per_subject: list[dict],
    selective: list[dict],
    baseline_pass1: float,
    pass5: float,
    sc_pass1: float,
) -> None:
    CROSS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    attempts_out = CROSS_DATASET_DIR / f"{OUTPUT_STEM}_attempts.csv"
    votes_out = CROSS_DATASET_DIR / f"{OUTPUT_STEM}_votes.csv"
    summary_json = CROSS_DATASET_DIR / f"{OUTPUT_STEM}.json"
    summary_md = CROSS_DATASET_DIR / f"{OUTPUT_STEM}.md"

    attempts.to_csv(attempts_out, index=False)
    votes.to_csv(votes_out, index=False)

    sc_gain = sc_pass1 - baseline_pass1
    oracle_gap = pass5 - baseline_pass1
    gap_recovered = (sc_gain / oracle_gap) if oracle_gap > 0 else float("nan")

    payload = {
        "subset": ORIGIN,
        "runs": runs,
        "n_problems": int(len(votes)),
        "n_runs": len(runs),
        "baseline_pass1_avg_per_attempt": baseline_pass1,
        "pass5": pass5,
        "sc_pass1": sc_pass1,
        "sc_gain_vs_avg_pass1": sc_gain,
        "oracle_gap_recovered_by_sc": gap_recovered,
        "mean_consistency": float(votes["consistency"].mean()),
        "per_run": per_run,
        "per_subject": per_subject,
        "selective_curve": selective,
        "attempts_csv": str(attempts_out),
        "votes_csv": str(votes_out),
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# FrontierScience DeepSeek Olympiad — Self-Consistency Baseline",
        "",
        "Plurality vote (Wang et al., 2022) across five same-config DeepSeek runs on the",
        "Olympiad subset of FrontierScience. Each per-attempt grade is reused from the",
        "upstream `is_correct` label; the SC verdict is the majority `is_correct` among",
        "attempts that share the winning normalized answer.",
        "",
        "## Setup",
        "",
        f"- Runs: {len(runs)}",
        f"- Subset: `{ORIGIN}` (filter: `has_final_answer == true`)",
        f"- Problems with at least one row in every run: {len(votes)}",
        "- Errors / blank `predicted_answer` abstain (don't vote); SC counts them as wrong if no plurality forms.",
        "- Ties broken by earliest `run_order` of the tied group (label-free).",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Average-per-attempt Pass@1 | {_pct(baseline_pass1)} |",
        f"| Pass@5 (oracle across the five attempts) | {_pct(pass5)} |",
        f"| **Self-consistency Pass@1** | **{_pct(sc_pass1)}** |",
        f"| Gain over avg Pass@1 | {sc_gain:+.3f} ({100.0 * sc_gain:+.1f} pts) |",
        f"| Oracle gap recovered | {_pct(gap_recovered)} |",
        f"| Mean consistency (winning_count / 5) | {votes['consistency'].mean():.3f} |",
        "",
        "## Per-run baselines",
        "",
        "| Run | Answered | Errors | Pass@1 |",
        "|---|---:|---:|---|",
    ]
    for row in per_run:
        lines.append(
            f"| `{row['run_slug_short']}` | {row['answered']} | {row['errors']} | {_pct(row['pass1'])} |"
        )

    lines += [
        "",
        "## Per-subject breakdown",
        "",
        "| Subject | Problems | Avg Pass@1 | Pass@5 | SC Pass@1 | Mean consistency |",
        "|---|---:|---|---|---|---:|",
    ]
    for row in per_subject:
        lines.append(
            f"| {row['subject']} | {row['n_problems']} | {_pct(row['avg_pass1'])} | "
            f"{_pct(row['pass5'])} | {_pct(row['sc_pass1'])} | {row['mean_consistency']:.3f} |"
        )

    lines += [
        "",
        "## Selective prediction by agreement threshold",
        "",
        "Restrict SC predictions to problems whose winning answer received at least N of 5 votes.",
        "",
        "| Min winning votes | Coverage | Problems | SC Pass@1 |",
        "|---:|---|---:|---|",
    ]
    for row in selective:
        lines.append(
            f"| ≥{row['min_winning_count']}/5 | {_pct(row['coverage'])} | "
            f"{row['n_problems']} | {_pct(row['sc_pass1'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is the **textbook self-consistency baseline**: choose the answer most",
        "  models agree on across independent samples. It uses no SLoD signal and no",
        "  external scorer; the only inputs are the five generations and the upstream",
        "  per-sample grading already in `traces.jsonl`.",
        "- Compare directly against `frontierscience_deepseek_olympiad_bestof5.md`,",
        "  which selects among the same five attempts using SLoD-derived confidence.",
        "  Differences in `Pass@1` between the two reflect whether SLoD provides signal",
        "  beyond simple inter-sample agreement.",
        "- Coverage / SC-Pass@1 along the agreement threshold is a selective-prediction",
        "  curve: high-consistency problems should be high-accuracy if self-consistency",
        "  is well-calibrated.",
        "",
        f"Artifacts: `{attempts_out.name}`, `{votes_out.name}`, `{summary_json.name}`.",
    ]
    summary_md.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", summary_md)


def main() -> int:
    setup_logging()
    args = parse_args()
    runs = args.run or list(DEFAULT_RUNS)

    attempt_frames: list[pd.DataFrame] = []
    per_run: list[dict] = []
    for run_order, run in enumerate(runs):
        paths = RunPaths.for_run(run)
        df = _load_olympiad_attempts(paths, run_order=run_order)
        attempt_frames.append(df)
        per_run.append(
            {
                "run": run,
                "run_slug_short": _run_short_slug(run),
                "answered": int(df["answered"].sum()),
                "errors": int((~df["answered"]).sum()),
                "pass1": float(df["is_correct"].mean()),
            }
        )

    _validate_problem_alignment(attempt_frames)
    attempts = pd.concat(attempt_frames, ignore_index=True)
    attempts = attempts.sort_values(["id", "run_order"], ignore_index=True)

    baseline_pass1 = float(attempts["is_correct"].mean())
    pass5 = float(attempts.groupby("id")["is_correct"].max().mean())

    n_runs = len(runs)
    votes = (
        attempts.groupby("id", sort=True, group_keys=False)
        .apply(lambda g: _self_consistency_vote(g, n_runs=n_runs), include_groups=False)
        .reset_index()
    )

    sc_pass1 = float(votes["sc_correct"].mean())
    per_subject = _per_subject(votes, attempts)
    selective = _selective_curve(votes)

    logger.info(
        "n_problems=%d  avg_pass1=%.3f  pass5=%.3f  sc_pass1=%.3f  mean_consistency=%.3f",
        len(votes), baseline_pass1, pass5, sc_pass1, votes["consistency"].mean(),
    )

    _write_outputs(
        runs=runs,
        attempts=attempts,
        votes=votes,
        per_run=per_run,
        per_subject=per_subject,
        selective=selective,
        baseline_pass1=baseline_pass1,
        pass5=pass5,
        sc_pass1=sc_pass1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
