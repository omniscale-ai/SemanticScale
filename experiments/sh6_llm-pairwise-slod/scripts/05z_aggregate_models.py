#!/usr/bin/env python
"""SH6 Stage-5z: cross-dataset model comparison.

Walks `reports/` looking for runs that have OOF parquets for both the
incumbent (`logreg`) and the challenger (`lightgbm`, default), then:

1. Computes mean OOF ROC-AUC for each model on each run.
2. For every (run, challenger) pair, computes Δ-AUC vs the incumbent on
   the same items, plus a 95% percentile-bootstrap CI on the Δ.
3. Applies the pre-registered decision rule from
   `DESIGN-stage5-models.md`: a model wins on a run iff
       Δ-AUC ≥ +0.03 AND CI-lower > 0
   and carries the comparison iff it wins on ≥3 of the 5 protocol
   datasets.
4. Writes a CSV table, a markdown summary, and a forest plot of Δ-AUC
   per run to `reports/_cross_dataset/`.

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05z_aggregate_models.py
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05z_aggregate_models.py \\
        --reports-dir experiments/sh6_llm-pairwise-slod/reports \\
        --challenger lightgbm
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

from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)


REPORTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "reports"
PROTOCOL_DELTA = 0.03
N_BOOTSTRAP_DEFAULT = 1000

# 5 protocol datasets used to apply the ≥3-of-5 carry rule. Anything outside
# this set is reported but does not count toward the carry decision.
# Patterns are matched against the relative path of the run dir.
PROTOCOL_RUN_PATTERNS = [
    "frontierscience/deepseek/deepseek-v3.2_reasoning-auto",
    "swe-agent-trajectories/model-all",  # excludes ".../model-all_steps-50"
    "agenthallu/framework-all",
    "gpqa-diamond/deepseek/deepseek-v3.2_reasoning-auto",
    # one processbench split is enough for the protocol set; we pick gsm8k
    # because it is the smallest and class-balanced.
    "processbench/gsm8k",
]


@dataclass
class RunRecord:
    rel_path: str
    dataset: str
    run_slug: str
    n_items: int
    pos_count: int


def _find_runs_with_oof(reports_dir: Path, incumbent: str, challenger: str) -> list[Path]:
    """Locate every run dir that has both incumbent and challenger OOF parquets."""
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
    """Δ = AUC(b) - AUC(a) with 95% percentile-bootstrap CI on the difference.

    Resamples row indices with replacement (preserves item-level pairing
    between the two models) and recomputes both AUCs on the same draw.
    Skips draws where one class is absent in the bootstrap sample.
    """
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


def _decide(delta_point: float, ci_low: float) -> str:
    """Pre-registered per-run verdict from DESIGN-stage5-models.md."""
    if np.isnan(ci_low):
        return "insufficient_data"
    if delta_point >= PROTOCOL_DELTA and ci_low > 0:
        return "win"
    if delta_point <= -PROTOCOL_DELTA and ci_low < 0:
        # symmetric: significant regression in the wrong direction
        return "regress"
    return "inconclusive"


def aggregate(
    reports_dir: Path,
    incumbent: str,
    challenger: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    runs = _find_runs_with_oof(reports_dir, incumbent, challenger)
    if not runs:
        msg = f"No runs found with both {incumbent} and {challenger} OOF parquets under {reports_dir}"
        raise SystemExit(msg)

    rng = np.random.default_rng(seed)
    rows = []
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

        # Exact match (or prefix followed by `/`) prevents
        # `swe-agent-trajectories/model-all` from matching the unrelated
        # `swe-agent-trajectories/model-all_steps-50`.
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
                "verdict": _decide(delta_point, ci_low),
                "in_protocol_set": in_protocol,
            }
        )

    return pd.DataFrame(rows).sort_values("rel_path", ignore_index=True)


def render_markdown(df: pd.DataFrame, incumbent: str, challenger: str) -> str:
    """Build a one-table comparison report keyed off the dataframe."""
    lines = [f"# SH6 Stage-5 — Cross-Dataset Model Comparison\n",
             f"**Incumbent:** `{incumbent}` &nbsp;&nbsp; **Challenger:** `{challenger}`\n",
             f"**Decision rule:** Δ-AUC ≥ +{PROTOCOL_DELTA:.2f} AND CI-lower > 0 (per run); "
             f"≥3 of 5 protocol runs to carry overall.\n"]

    cols = [
        "rel_path", "n_items", "n_pos",
        f"auc_{incumbent}", f"auc_{challenger}",
        "delta_point", "ci_low", "ci_high", "verdict", "in_protocol_set",
    ]
    table = df[cols].copy()
    table.columns = [
        "Run", "N", "Pos",
        f"AUC {incumbent}", f"AUC {challenger}",
        "Δ-AUC", "CI low", "CI high", "Verdict", "In protocol set",
    ]
    for c in [f"AUC {incumbent}", f"AUC {challenger}", "Δ-AUC", "CI low", "CI high"]:
        table[c] = table[c].map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}" if c == "Δ-AUC" else f"{v:.3f}")
    lines.append(table.to_markdown(index=False))

    protocol = df[df["in_protocol_set"]]
    n_wins = int((protocol["verdict"] == "win").sum())
    n_protocol = len(protocol)
    n_regressions = int((protocol["verdict"] == "regress").sum())
    carry = n_wins >= 3
    lines.extend([
        "\n## Protocol verdict\n",
        f"- Protocol runs evaluated: **{n_protocol} / 5**",
        f"- Challenger wins: **{n_wins}**",
        f"- Significant regressions: **{n_regressions}**",
        f"- ≥3-of-5 carry rule: **{'PASS — challenger carries the comparison' if carry else 'FAIL — challenger does NOT carry the comparison'}**",
    ])
    if not carry:
        lines.append(
            "\nPer the stop conditions in `DESIGN-stage5-models.md`, this implies "
            "no broadly applicable interaction signal in the current Stage-5 "
            "feature set. The right next step is feature engineering (TA-pack, "
            "multi-scale, cross-trajectory), not a stronger classifier."
        )
    return "\n".join(lines) + "\n"


def render_forest_plot(df: pd.DataFrame, incumbent: str, challenger: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # local import keeps non-plot users light

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df) + 1.5)))
    y_pos = np.arange(len(df))
    deltas = df["delta_point"].to_numpy()
    ci_low = df["ci_low"].to_numpy()
    ci_high = df["ci_high"].to_numpy()
    err_low = np.where(np.isnan(ci_low), 0, deltas - ci_low)
    err_high = np.where(np.isnan(ci_high), 0, ci_high - deltas)
    colors = [
        "#1b7837" if v == "win"
        else "#762a83" if v == "regress"
        else "#999999"
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
    ax.set_xlabel(f"Δ-AUC ({challenger} − {incumbent}), 95% paired bootstrap CI")
    ax.set_title("SH6 Stage-5 — Δ-AUC across runs")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT)
    parser.add_argument("--incumbent", default="logreg")
    parser.add_argument("--challenger", default="lightgbm")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("INFO")
    out_dir = args.reports_dir / "_cross_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate(
        reports_dir=args.reports_dir,
        incumbent=args.incumbent,
        challenger=args.challenger,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    csv_path = out_dir / f"model_comparison_{args.incumbent}_vs_{args.challenger}.csv"
    df.to_csv(csv_path, index=False)
    md_path = out_dir / f"model_comparison_{args.incumbent}_vs_{args.challenger}.md"
    md_path.write_text(render_markdown(df, args.incumbent, args.challenger))
    plot_path = out_dir / f"delta_auc_{args.incumbent}_vs_{args.challenger}.png"
    render_forest_plot(df, args.incumbent, args.challenger, plot_path)

    summary = {
        "n_runs": int(len(df)),
        "n_in_protocol": int(df["in_protocol_set"].sum()),
        "n_wins_protocol": int(((df["verdict"] == "win") & df["in_protocol_set"]).sum()),
        "n_regress_protocol": int(((df["verdict"] == "regress") & df["in_protocol_set"]).sum()),
        "carry": bool(((df["verdict"] == "win") & df["in_protocol_set"]).sum() >= 3),
    }
    (out_dir / f"model_comparison_{args.incumbent}_vs_{args.challenger}.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s, %s, %s", csv_path, md_path, plot_path)
    logger.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
