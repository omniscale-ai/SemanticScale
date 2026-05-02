"""Interpretable failure-mode detectors for SH6 trajectories.

The overall-correctness predictor (``failure_analysis.evaluate_prediction_models``)
is intentionally opaque: a multivariate logistic regression over 20+ trajectory
features. It is good at ranking traces by failure risk, but it does not say
*why* a trace looks like a failure.

This module adds a complementary layer of **named detectors**, one per
hypothesised failure mode. Each detector:

- reduces the trace to a single interpretable score (higher = more
  failure-like),
- is calibrated against the success distribution so its threshold has a
  concrete meaning ("this trace sits outside the 90th percentile of
  successful traces"), and
- is reported with precision, recall, and lift over the base failure rate,
  which makes it easy to trust or discard individual detectors.

The modes come from inspecting two complementary trace styles, and were
defined under two **different methodologies** that a reader should keep in
mind when interpreting verdicts:

- **Reasoning-side detectors** (``thrashing``, ``no_commitment``,
  ``derailment_late``, ``rambling_overlong``, ``premature_exit``,
  ``truncation_abort``) were named from a qualitative read of SWE-agent
  failure traces *before* AUCs were computed. Their directional hypothesis
  ("higher score implies more failure-like") was formulated a priori;
  the bootstrap-CI verdict is therefore a real falsification test on every
  run, including the SWE-agent run that motivated them.
- **Answer-side detectors** (``answer_meandering``, ``answer_volatility``,
  ``answer_uncommitted``, ``answer_overrange``, ``answer_drift``) were
  discovered post-hoc by ranking trajectory features by univariate AUC on
  FrontierScience and selecting the top performers, with the score sign
  chosen to make the AUC > 0.5. This means:

  - On **FrontierScience** the ``confirmed`` verdict is circular by
    construction — the detectors were selected to predict on this dataset.
    Treat the FrontierScience verdict as descriptive, not as evidence.
  - The bootstrap CI does **not** correct for selection from ~60 candidate
    features; some "confirmed" verdicts on FS would survive even on noise
    after that much picking.
  - On **other datasets** (SWE-agent has no answer chunks; AgentHallu's
    candidate features all sit around AUC≈0.5) the verdict mechanism is
    unbiased and correctly reports ``insufficient_data`` or
    ``inconclusive``, which is exactly the desired behaviour.

  These detectors should be treated as FrontierScience-suggested hypotheses
  awaiting out-of-sample replication on a different FS run (e.g. a different
  generator model, or a fresh data slice).

The falsifiability machinery below tests every detector on every run, so a
detector that is real on one dataset and inconclusive on another is reported
honestly rather than averaged away. The detectors are deliberately simple
single-feature scores — they label, they do not compete with the multivariate
logistic predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRUNCATION_MARKERS = ("exit_context", "early_exit", "exit_format", "exit_cost")

# Falsifiability configuration. Each detector encodes the hypothesis
# "higher score ==> more failure-like". A detector is only considered
# confirmed on a run if a 95% bootstrap CI on its failure-AUC excludes 0.5
# on the correct side.
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_RANDOM_STATE = 42
MIN_SCORED_FOR_VERDICT = 20

VERDICT_CONFIRMED = "confirmed"
VERDICT_INVERTED = "inverted"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_INSUFFICIENT = "insufficient_data"


@dataclass
class FailureMode:
    """One interpretable detector for a single failure mode.

    Attributes:
        name: short snake_case identifier, used for column names.
        description: human-readable summary for the report.
        score_fn: maps the feature table to a per-row score where higher
            values mean "more failure-like". May return ``NaN`` for rows where
            the mode is undefined (e.g. answer-alignment on traces without
            answer chunks).
        kind: ``continuous`` or ``binary``.
        calibration_quantile: for continuous modes, the quantile of the
            *success* distribution used as the flag threshold.
        fixed_threshold: for binary modes, the fixed threshold (default 0.5).
    """

    name: str
    description: str
    score_fn: Callable[[pd.DataFrame], pd.Series]
    kind: str = "continuous"
    calibration_quantile: float = 0.90
    fixed_threshold: float = 0.5


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column or an all-NaN placeholder if it is missing."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _score_premature_exit(df: pd.DataFrame) -> pd.Series:
    return -_col(df, "reasoning_n_chunks")


def _score_rambling(df: pd.DataFrame) -> pd.Series:
    return _col(df, "reasoning_n_chunks")


def _score_thrashing(df: pd.DataFrame) -> pd.Series:
    return _col(df, "reasoning_direction_changes")


def _score_no_commitment(df: pd.DataFrame) -> pd.Series:
    return -_col(df, "reasoning_monotonicity")


def _score_derailment_late(df: pd.DataFrame) -> pd.Series:
    return _col(df, "reasoning_fall_from_peak")


def _score_answer_drift(df: pd.DataFrame) -> pd.Series:
    return _col(df, "answer_start_minus_reasoning_end").abs()


# ---------------------------------------------------------------------------
# Answer-side detectors. POST-HOC: feature and direction were chosen by ranking
# univariate failure-AUCs on FrontierScience and keeping the top performers.
# The "confirmed" verdict on FrontierScience is therefore not an honest test
# of these detectors — that would require an out-of-sample FS run. Verdicts on
# other datasets remain unbiased.
# ---------------------------------------------------------------------------


def _score_answer_meandering(df: pd.DataFrame) -> pd.Series:
    return _col(df, "answer_direction_changes")


def _score_answer_volatility(df: pd.DataFrame) -> pd.Series:
    return _col(df, "answer_max_rise")


def _score_answer_uncommitted(df: pd.DataFrame) -> pd.Series:
    return -_col(df, "answer_monotonicity")


def _score_answer_overrange(df: pd.DataFrame) -> pd.Series:
    return _col(df, "answer_range_minus_reasoning_range")


def _score_truncation(df: pd.DataFrame) -> pd.Series:
    if "exit_status" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    status = df["exit_status"].fillna("").astype(str).str.lower()
    flags = status.apply(
        lambda s: 1.0 if any(marker in s for marker in TRUNCATION_MARKERS) else 0.0
    )
    # leave NaN on rows where exit_status itself is missing for cross-dataset runs
    flags[df["exit_status"].isna()] = np.nan
    return flags


MODES: list[FailureMode] = [
    FailureMode(
        name="premature_exit",
        description="Very short reasoning trace — the model answered or gave up before exploring.",
        score_fn=_score_premature_exit,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="rambling_overlong",
        description="Reasoning runs much longer than on successful traces, often without ever synthesising.",
        score_fn=_score_rambling,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="thrashing",
        description="Many SLoD direction changes — the model flip-flops between abstraction levels instead of committing.",
        score_fn=_score_thrashing,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="no_commitment",
        description="Low end-to-end monotonicity — the trace never commits to a clear abstraction arc.",
        score_fn=_score_no_commitment,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="derailment_late",
        description="Trace peaks high on the SLoD axis and then falls away, failing to land.",
        score_fn=_score_derailment_late,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="answer_drift",
        description="Answer SLoD is far from where the reasoning ended — the answer does not follow the chain.",
        score_fn=_score_answer_drift,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="answer_meandering",
        description="Answer trajectory has many SLoD direction changes — long, oscillating answer instead of a clean statement (FrontierScience-style hedging).",
        score_fn=_score_answer_meandering,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="answer_volatility",
        description="Large single-step SLoD jump in the answer — the response leaps between abstraction levels, a confabulation pattern.",
        score_fn=_score_answer_volatility,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="answer_uncommitted",
        description="Low monotonicity inside the answer trajectory — the answer never commits to a clear arc.",
        score_fn=_score_answer_uncommitted,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="answer_overrange",
        description="Answer covers a wider SLoD range than the reasoning did — the answer claims abstraction breadth the reasoning never built up.",
        score_fn=_score_answer_overrange,
        kind="continuous",
        calibration_quantile=0.90,
    ),
    FailureMode(
        name="truncation_abort",
        description="Agent exited on context/budget/format, not a clean submit (SWE-agent style).",
        score_fn=_score_truncation,
        kind="binary",
        fixed_threshold=0.5,
    ),
]


def compute_mode_scores(
    df: pd.DataFrame, modes: list[FailureMode] | None = None
) -> pd.DataFrame:
    """Compute per-row scores for every detector.

    Returns a dataframe indexed like ``df`` with two columns per mode:
    ``<mode>_score`` and ``<mode>_flag``. Flags are only set for rows that
    have a valid (non-NaN) score; everywhere else the flag is ``NaN`` so
    downstream metrics can ignore those rows.
    """
    modes = modes or MODES
    out = pd.DataFrame(index=df.index)
    target = df["target"].to_numpy() if "target" in df.columns else None

    for mode in modes:
        scores = mode.score_fn(df).astype(float)
        flags = pd.Series(np.nan, index=df.index, dtype=float)
        valid = scores.notna()
        if valid.any():
            threshold = _compute_threshold(scores[valid], target, valid, mode)
            flags[valid] = (scores[valid] > threshold).astype(float)
        out[f"{mode.name}_score"] = scores
        out[f"{mode.name}_flag"] = flags
    return out


def _compute_threshold(
    valid_scores: pd.Series,
    target: np.ndarray | None,
    valid_mask: pd.Series,
    mode: FailureMode,
) -> float:
    """Resolve the flag threshold for one mode.

    Continuous modes calibrate against the *success* distribution when labels
    are available. Binary modes use ``fixed_threshold``. If the success class
    is absent we fall back to the overall distribution.
    """
    if mode.kind == "binary":
        return float(mode.fixed_threshold)

    if target is not None:
        positive_mask = (target == 1) & valid_mask.to_numpy()
        if positive_mask.sum() >= 3:
            positive_scores = valid_scores[positive_mask[valid_mask.to_numpy()]]
            return float(np.nanquantile(positive_scores, mode.calibration_quantile))

    return float(np.nanquantile(valid_scores, mode.calibration_quantile))


def _bootstrap_auc_ci(
    y_fail: np.ndarray,
    scores: np.ndarray,
    n_boot: int = BOOTSTRAP_SAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> tuple[float, float] | tuple[None, None]:
    """Percentile bootstrap CI on the failure-vs-score AUC.

    Returns ``(lo, hi)`` or ``(None, None)`` if the bootstrap cannot produce a
    usable distribution (e.g. too few scored rows or degenerate labels).
    """
    if len(y_fail) < MIN_SCORED_FOR_VERDICT:
        return (None, None)
    rng = np.random.default_rng(random_state)
    n = len(y_fail)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y_fail[idx]
        s_boot = scores[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_boot, s_boot)))
        except ValueError:
            continue
    if len(aucs) < n_boot // 4:
        return (None, None)
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _verdict_from_ci(ci_lo: float | None, ci_hi: float | None) -> str:
    """Map the bootstrap CI to a hypothesis verdict."""
    if ci_lo is None or ci_hi is None:
        return VERDICT_INSUFFICIENT
    if ci_lo > 0.5:
        return VERDICT_CONFIRMED
    if ci_hi < 0.5:
        return VERDICT_INVERTED
    return VERDICT_INCONCLUSIVE


def evaluate_mode_detectors(
    df: pd.DataFrame,
    mode_scores: pd.DataFrame,
    modes: list[FailureMode] | None = None,
) -> list[dict]:
    """Evaluate every detector against the failure label with a falsifiability test.

    Each detector encodes a directional hypothesis ("higher score means
    more failure-like"). For each run we compute a 95% percentile-bootstrap CI
    on the score-vs-failure AUC and map it to a verdict:

    - ``confirmed``: CI lower bound > 0.5 — the directional claim holds.
    - ``inverted``: CI upper bound < 0.5 — the score actually predicts success.
    - ``inconclusive``: CI spans 0.5 — no evidence either way.
    - ``insufficient_data``: too few rows or only one class.

    The flag's precision / recall / lift are only meaningful when the verdict
    is ``confirmed``; for other verdicts those numbers are set to ``NaN`` so
    they don't appear in reports as spurious confirmations. The raw AUC and CI
    are always reported so a reader can audit the call.
    """
    modes = modes or MODES
    if "target" not in df.columns:
        raise ValueError("df must contain a 'target' column")

    y = df["target"].to_numpy()
    fail = (y == 0).astype(int)

    rows: list[dict] = []
    for mode in modes:
        scores = mode_scores[f"{mode.name}_score"]
        flags = mode_scores[f"{mode.name}_flag"]
        valid = scores.notna().to_numpy()
        n_scored = int(valid.sum())
        if n_scored == 0:
            rows.append(_empty_mode_row(mode, reason="feature unavailable"))
            continue

        y_valid = y[valid]
        fail_valid = fail[valid]
        scores_valid = scores.to_numpy()[valid]
        flags_valid = flags.to_numpy()[valid].astype(bool)

        n_flagged = int(flags_valid.sum())
        base_rate = float(fail_valid.mean()) if n_scored > 0 else float("nan")
        true_positive = int(((flags_valid == 1) & (fail_valid == 1)).sum())
        false_positive = int(((flags_valid == 1) & (fail_valid == 0)).sum())
        false_negative = int(((flags_valid == 0) & (fail_valid == 1)).sum())

        roc_auc = float("nan")
        ci_lo: float | None = None
        ci_hi: float | None = None
        if len(np.unique(y_valid)) == 2 and n_scored >= MIN_SCORED_FOR_VERDICT:
            try:
                roc_auc = float(roc_auc_score(fail_valid, scores_valid))
            except ValueError:
                roc_auc = float("nan")
            ci_lo, ci_hi = _bootstrap_auc_ci(fail_valid, scores_valid)

        verdict = _verdict_from_ci(ci_lo, ci_hi)

        if verdict == VERDICT_CONFIRMED and n_flagged:
            precision = float(true_positive / n_flagged)
            recall = float(true_positive / max(true_positive + false_negative, 1))
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            lift = precision / base_rate if base_rate else float("nan")
        else:
            precision = float("nan")
            recall = float("nan")
            f1 = float("nan")
            lift = float("nan")

        rows.append(
            {
                "mode": mode.name,
                "description": mode.description,
                "kind": mode.kind,
                "n_scored": n_scored,
                "n_flagged": n_flagged,
                "base_rate": base_rate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "lift": lift,
                "roc_auc": roc_auc,
                "roc_auc_ci_lo": ci_lo if ci_lo is not None else float("nan"),
                "roc_auc_ci_hi": ci_hi if ci_hi is not None else float("nan"),
                "verdict": verdict,
                "threshold_quantile": mode.calibration_quantile
                if mode.kind == "continuous"
                else None,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "reason_skipped": None,
            }
        )

    return rows


def _empty_mode_row(mode: FailureMode, reason: str) -> dict:
    return {
        "mode": mode.name,
        "description": mode.description,
        "kind": mode.kind,
        "n_scored": 0,
        "n_flagged": 0,
        "base_rate": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "lift": float("nan"),
        "roc_auc": float("nan"),
        "roc_auc_ci_lo": float("nan"),
        "roc_auc_ci_hi": float("nan"),
        "verdict": VERDICT_INSUFFICIENT,
        "threshold_quantile": mode.calibration_quantile
        if mode.kind == "continuous"
        else None,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "reason_skipped": reason,
    }


def render_mode_table_markdown(mode_rows: list[dict]) -> list[str]:
    """Render the detector-evaluation table as markdown lines.

    Flag-level metrics (precision/recall/F1/lift) are only populated when the
    detector's directional hypothesis was confirmed by the bootstrap CI; for
    other verdicts they render as ``-`` so they cannot be mistaken for
    evidence that the mode is real on this run.
    """
    lines = [
        "| Mode | Verdict | Scored | Score AUC (95% CI) | Base Fail Rate | Flagged | Precision | Recall | F1 | Lift |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in mode_rows:
        if row.get("reason_skipped"):
            lines.append(
                f"| {row['mode']} | {row.get('verdict', VERDICT_INSUFFICIENT)} | 0 | - | - | 0 | - | - | - | - |"
            )
            continue
        auc_txt = _fmt(row["roc_auc"])
        if row.get("roc_auc_ci_lo") == row.get("roc_auc_ci_lo") and row.get(
            "roc_auc_ci_hi"
        ) == row.get("roc_auc_ci_hi"):
            auc_txt = f"{auc_txt} [{_fmt(row['roc_auc_ci_lo'])}, {_fmt(row['roc_auc_ci_hi'])}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mode"],
                    row.get("verdict", VERDICT_INSUFFICIENT),
                    str(row["n_scored"]),
                    auc_txt,
                    _fmt(row["base_rate"]),
                    str(row["n_flagged"]),
                    _fmt(row["precision"]),
                    _fmt(row["recall"]),
                    _fmt(row["f1"]),
                    _fmt(row["lift"]),
                ]
            )
            + " |"
        )
    return lines


def render_mode_descriptions_markdown(mode_rows: list[dict]) -> list[str]:
    """Render a short glossary explaining what each detector catches."""
    lines = ["| Mode | What it catches |", "|---|---|"]
    for row in mode_rows:
        lines.append(f"| {row['mode']} | {row['description']} |")
    return lines


def write_mode_table(
    df: pd.DataFrame,
    mode_scores: pd.DataFrame,
    path,
) -> None:
    """Persist per-item detector scores and flags for follow-up analysis."""
    keep_meta = [
        col
        for col in (
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
            "exit_status",
        )
        if col in df.columns
    ]
    out = pd.concat([df[keep_meta].reset_index(drop=True), mode_scores.reset_index(drop=True)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "-"
    return f"{value:.3f}"
