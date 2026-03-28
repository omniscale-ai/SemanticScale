#!/usr/bin/env python
"""Stage E: Generate auto-report from analysis results."""

import sys
import json
from pathlib import Path
import yaml

WORKING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKING_DIR))

RESULTS_DIR = WORKING_DIR / "data" / "results"
REPORT_PATH = WORKING_DIR / "reports" / "SH5c_auto_report.md"


def load_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    with open(WORKING_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    corr = load_json(RESULTS_DIR / "correlations.json")
    wilcoxon = load_json(RESULTS_DIR / "wilcoxon.json")
    logistic = load_json(RESULTS_DIR / "logistic.json")
    partial = load_json(RESULTS_DIR / "partial_correlations.json")
    subgroups = load_json(RESULTS_DIR / "subgroups.json")

    exit_criteria = cfg.get("exit_criteria", {})

    # Evaluate hypotheses
    h1_confirmed = False
    h1_details = []
    pooled = corr.get("pooled", {})
    for feat, metrics in pooled.items():
        for metric, r in metrics.items():
            if r.get("significant") and abs(r.get("rho", 0)) > exit_criteria.get("h1_min_abs_rho", 0.10):
                h1_confirmed = True
                h1_details.append(f"{feat} vs {metric}: rho={r['rho']:+.4f}, p_corr={r['p_corrected']:.4f}")

    h2_confirmed = False
    h2_details = []
    for pair_key, feats in wilcoxon.items():
        for feat, r in feats.items():
            if r.get("significant") and r.get("routed_lower"):
                h2_confirmed = True
                h2_details.append(f"{pair_key} / {feat}: p={r['p']:.4f}, "
                                  f"routed={r['mean_routed']:.4f}, baseline={r['mean_baseline']:.4f}")

    h3_confirmed = False
    h3_details = []
    alignment_model = logistic.get("alignment_only", {})
    auroc = alignment_model.get("auroc_mean", 0)
    if auroc > exit_criteria.get("h3_min_auroc", 0.60):
        h3_confirmed = True
    h3_details.append(f"Alignment-only AUROC: {auroc:.4f} +/- {alignment_model.get('auroc_std', 0):.4f}")

    # Overall verdict
    if h1_confirmed and (h2_confirmed or h3_confirmed):
        verdict = "CONFIRMED"
    elif h1_confirmed or h2_confirmed or h3_confirmed:
        verdict = "PARTIAL"
    else:
        verdict = "NOT CONFIRMED"

    # Build report
    lines = []
    lines.append("# SH5c Auto-Report: Per-Question SLoD Consistency Predicts Quality")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")

    # H1
    lines.append("## H1: Alignment-Quality Correlation")
    lines.append(f"**Status: {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'}**")
    lines.append("")
    if h1_details:
        for d in h1_details:
            lines.append(f"- {d}")
    else:
        lines.append("No alignment feature reached |rho| > 0.10 with Bonferroni-corrected p < 0.05.")
    lines.append("")

    # Correlation table
    lines.append("### Full Correlation Table (Pooled, N=2000)")
    lines.append("| Feature | vs token-F1 (rho) | p_corr | vs attr-F1 (rho) | p_corr |")
    lines.append("|---------|-------------------|--------|-------------------|--------|")
    for feat in sorted(pooled.keys()):
        tf1 = pooled[feat].get("answer_token_f1", {})
        af1 = pooled[feat].get("attribution_f1", {})
        tf1_sig = " *" if tf1.get("significant") else ""
        af1_sig = " *" if af1.get("significant") else ""
        lines.append(f"| {feat} | {tf1.get('rho', 0):+.4f}{tf1_sig} | {tf1.get('p_corrected', 1):.4f} | "
                     f"{af1.get('rho', 0):+.4f}{af1_sig} | {af1.get('p_corrected', 1):.4f} |")
    lines.append("")

    # H2
    lines.append("## H2: Cross-Condition Alignment")
    lines.append(f"**Status: {'CONFIRMED' if h2_confirmed else 'NOT CONFIRMED'}**")
    lines.append("")
    if h2_details:
        for d in h2_details:
            lines.append(f"- {d}")
    else:
        lines.append("SLoD-routed conditions did not show significantly lower alignment gap.")
    lines.append("")

    # H3
    lines.append("## H3: Predictive Power")
    lines.append(f"**Status: {'CONFIRMED' if h3_confirmed else 'NOT CONFIRMED'}**")
    lines.append("")
    for d in h3_details:
        lines.append(f"- {d}")
    lines.append("")

    # Logistic regression comparison
    lines.append("### Model Comparison")
    lines.append("| Model | AUROC | Std | N | Features |")
    lines.append("|-------|-------|-----|---|----------|")
    for model_name in ["alignment_only", "jump_only", "combined"]:
        r = logistic.get(model_name, {})
        lines.append(f"| {model_name} | {r.get('auroc_mean', 0):.4f} | {r.get('auroc_std', 0):.4f} | "
                     f"{r.get('n_samples', 0)} | {r.get('n_features', 0)} |")
    lines.append("")

    # Subgroup highlights
    lines.append("## Subgroup Analysis")
    by_cond = subgroups.get("by_condition", {})
    if by_cond:
        lines.append("### By Condition")
        lines.append("| Condition | Mean Align Gap | Mean JSD | Mean Token-F1 |")
        lines.append("|-----------|---------------|----------|---------------|")
        for cond in sorted(by_cond.keys()):
            r = by_cond[cond]["means"]
            lines.append(f"| {cond} | {r.get('mean_alignment_gap', 0):.4f} | "
                         f"{r.get('jsd', 0):.4f} | {r.get('answer_token_f1', 0):.4f} |")
    lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("![Alignment Gap by Condition](figures/alignment_gap_by_condition.png)")
    lines.append("![Alignment vs Quality](figures/alignment_vs_quality.png)")
    lines.append("![JSD vs Quality](figures/jsd_vs_quality.png)")
    lines.append("![Feature Importance](figures/feature_importance.png)")
    lines.append("![AUROC Comparison](figures/auroc_comparison.png)")
    lines.append("")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to {REPORT_PATH}")
    print(f"Overall verdict: {verdict}")


if __name__ == "__main__":
    main()
