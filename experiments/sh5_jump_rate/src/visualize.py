"""Stage G: Plots and report generation."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import SLOD_MAP, load_jsonl, resolve_path

logger = logging.getLogger(__name__)


def _setup_style(config: dict) -> None:
    """Configure matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", palette=config["visualization"]["color_palette"])
    plt.rcParams["figure.dpi"] = config["visualization"]["figure_dpi"]


def plot_jump_by_condition(jump_df: pd.DataFrame, config: dict) -> Path:
    """Box plots of normalized_jump_rate per condition."""
    _setup_style(config)
    fig_dir = resolve_path(config, config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "jump_rate_by_condition.png"

    conditions = config["retrieval"]["conditions"]
    colors = config["visualization"]["condition_colors"]

    fig, ax = plt.subplots(figsize=(8, 5))

    data = jump_df[jump_df["condition"].isin(conditions)].copy()
    # Order conditions
    data["condition"] = pd.Categorical(data["condition"], categories=conditions, ordered=True)

    palette = {c: colors[c] for c in conditions}
    sns.boxplot(data=data, x="condition", y="normalized_jump_rate", palette=palette, ax=ax)
    ax.set_xlabel("Retrieval Condition")
    ax.set_ylabel("Normalized Jump Rate")
    ax.set_title("SLoD Jump Rate by Retrieval Condition")
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return out_path


def plot_jump_vs_correctness(merged_df: pd.DataFrame, config: dict) -> Path:
    """Scatter plot: normalized_jump_rate vs answer_token_f1, colored by condition."""
    _setup_style(config)
    fig_dir = resolve_path(config, config["paths"]["figures_dir"])
    out_path = fig_dir / "jump_vs_correctness_scatter.png"

    from scipy import stats as sp_stats

    colors = config["visualization"]["condition_colors"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for cond in config["retrieval"]["conditions"]:
        cond_data = merged_df[merged_df["condition"] == cond]
        ax.scatter(
            cond_data["normalized_jump_rate"],
            cond_data["answer_token_f1"],
            alpha=0.3,
            s=15,
            c=colors[cond],
            label=cond.replace("_", " "),
        )

    # Overall regression line
    x = merged_df["normalized_jump_rate"].values
    y = merged_df["answer_token_f1"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) > 10:
        slope, intercept, _, _, _ = sp_stats.linregress(x_clean, y_clean)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)

        rho, p = sp_stats.spearmanr(x_clean, y_clean)
        ax.annotate(
            f"Spearman rho={rho:.3f}, p={p:.2e}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Normalized Jump Rate")
    ax.set_ylabel("Answer Token-F1")
    ax.set_title("Jump Rate vs Answer Correctness")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return out_path


def plot_slod_sequences(
    jump_df: pd.DataFrame,
    score_df: pd.DataFrame,
    config: dict,
) -> Path:
    """Sample CoT traces showing SLoD level per step as colored bars."""
    _setup_style(config)
    fig_dir = resolve_path(config, config["paths"]["figures_dir"])
    out_path = fig_dir / "slod_sequence_examples.png"

    slod_colors = config["visualization"]["slod_colors"]
    color_list = [slod_colors["macro"], slod_colors["meso"], slod_colors["micro"]]

    merged = jump_df.merge(score_df, on=["question_id", "condition"], how="inner")

    # Pick examples: 2 high + 2 low correctness from routed and unrouted
    examples = []
    for cond_group, conds in [
        ("routed", config["retrieval"]["routed_conditions"]),
        ("unrouted", config["retrieval"]["unrouted_conditions"]),
    ]:
        group_data = merged[merged["condition"].isin(conds)]
        if len(group_data) == 0:
            continue

        # Filter for traces with >= 3 steps
        group_data = group_data[group_data["n_steps"] >= 3]
        if len(group_data) == 0:
            continue

        sorted_data = group_data.sort_values("answer_token_f1")
        # Low correctness (bottom 10%)
        low = sorted_data.head(max(1, len(sorted_data) // 10))
        # High correctness (top 10%)
        high = sorted_data.tail(max(1, len(sorted_data) // 10))

        for label, subset in [("low", low), ("high", high)]:
            if len(subset) > 0:
                row = subset.sample(1, random_state=42).iloc[0]
                examples.append({
                    "label": f"{cond_group} / {label} F1",
                    "condition": row["condition"],
                    "f1": row["answer_token_f1"],
                    "sequence": row["slod_sequence"],
                    "njr": row["normalized_jump_rate"],
                })

    if not examples:
        logger.warning("No examples to plot for SLoD sequences")
        # Create empty figure
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"])
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 1.5 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for ax, ex in zip(axes, examples):
        seq = ex["sequence"]
        for step_i, level in enumerate(seq):
            ax.barh(0, 1, left=step_i, color=color_list[level], edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, len(seq))
        ax.set_yticks([])
        ax.set_xlabel("Step")
        ax.set_title(
            f"{ex['label']} ({ex['condition']}) — F1={ex['f1']:.2f}, NJR={ex['njr']:.2f}",
            fontsize=9,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=slod_colors["macro"], label="Macro"),
        Patch(facecolor=slod_colors["meso"], label="Meso"),
        Patch(facecolor=slod_colors["micro"], label="Micro"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return out_path


def plot_correlation_heatmap(corr_results: dict, config: dict) -> Path:
    """Heatmap of Spearman correlations between jump metrics and correctness measures."""
    _setup_style(config)
    fig_dir = resolve_path(config, config["paths"]["figures_dir"])
    out_path = fig_dir / "correlation_heatmap.png"

    from src.correlate import JUMP_METRICS, SCORE_METRICS

    # Build correlation matrix
    matrix = np.full((len(JUMP_METRICS), len(SCORE_METRICS)), np.nan)
    p_matrix = np.full_like(matrix, np.nan)

    for i, jm in enumerate(JUMP_METRICS):
        for j, sm in enumerate(SCORE_METRICS):
            key = f"{jm}_vs_{sm}"
            if key in corr_results.get("overall", {}):
                matrix[i, j] = corr_results["overall"][key]["spearman_rho"]
                p_matrix[i, j] = corr_results["overall"][key]["spearman_p"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Create annotation text with significance markers
    annot = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            p = p_matrix[i, j]
            if np.isnan(val):
                annot[i, j] = ""
            else:
                sig = ""
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                annot[i, j] = f"{val:.3f}{sig}"

    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        xticklabels=[s.replace("_", "\n") for s in SCORE_METRICS],
        yticklabels=[m.replace("_", "\n") for m in JUMP_METRICS],
        center=0,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
    )
    ax.set_title("Spearman Correlations: Jump Metrics vs Correctness")

    fig.tight_layout()
    fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return out_path


def plot_metric_distributions(jump_df: pd.DataFrame, config: dict) -> Path:
    """Histograms of jump metrics, faceted by condition."""
    _setup_style(config)
    fig_dir = resolve_path(config, config["paths"]["figures_dir"])
    out_path = fig_dir / "metric_distributions.png"

    from src.correlate import JUMP_METRICS

    conditions = config["retrieval"]["conditions"]
    colors = config["visualization"]["condition_colors"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(JUMP_METRICS):
        ax = axes[idx]
        for cond in conditions:
            cond_data = jump_df[jump_df["condition"] == cond][metric].dropna()
            ax.hist(
                cond_data, bins=20, alpha=0.5, color=colors[cond],
                label=cond.replace("_", " "), density=True,
            )
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel(metric)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Jump Metric Distributions by Condition", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return out_path


def plot_answer_type_breakdown(corr_results: dict, config: dict) -> Path:
    """Grouped bar chart: correlation strength by answer type."""
    _setup_style(config)
    fig_dir = resolve_path(config, config["paths"]["figures_dir"])
    out_path = fig_dir / "answer_type_breakdown.png"

    by_type = corr_results.get("by_answer_type", {})
    if not by_type:
        logger.warning("No per-answer_type correlations found")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"])
        plt.close(fig)
        return out_path

    answer_types = sorted(by_type.keys())
    metrics = ["normalized_jump_rate_vs_answer_token_f1", "normalized_jump_rate_vs_attribution_f1"]
    metric_labels = ["vs Token-F1", "vs Attribution-F1"]

    x = np.arange(len(answer_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (metric_key, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for atype in answer_types:
            if metric_key in by_type.get(atype, {}):
                values.append(by_type[atype][metric_key]["spearman_rho"])
            else:
                values.append(0)
        bars = ax.bar(x + i * width, values, width, label=label)

        # Add significance markers
        for j, atype in enumerate(answer_types):
            if metric_key in by_type.get(atype, {}):
                p = by_type[atype][metric_key]["spearman_p"]
                if p < 0.05:
                    ax.annotate("*", (x[j] + i * width, values[j]),
                              ha="center", va="bottom", fontsize=12)

    ax.set_xlabel("Answer Type")
    ax.set_ylabel("Spearman rho (NJR)")
    ax.set_title("Correlation: Normalized Jump Rate vs Correctness by Answer Type")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(answer_types)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config["visualization"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")
    return out_path


def generate_report(config: dict) -> Path:
    """Write the SH5_report.md with all results."""
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    reports_dir = resolve_path(config, config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "SH5_report.md"

    # Load results
    corr_path = data_dir / "results" / "correlation_results.json"
    cond_path = data_dir / "results" / "condition_comparison.json"

    with open(corr_path) as f:
        corr = json.load(f)
    with open(cond_path) as f:
        cond = json.load(f)

    # Load summary data
    jump_df = pd.DataFrame(load_jsonl(data_dir / "jump_metrics.jsonl"))
    score_df = pd.DataFrame(load_jsonl(data_dir / "answer_scores.jsonl"))

    # Determine verdict
    alpha = config["statistics"]["alpha"]
    min_rho = config["statistics"]["min_correlation_rho"]

    h1_key = "normalized_jump_rate_vs_answer_token_f1"
    h1_passed = False
    if h1_key in corr.get("overall", {}):
        r = corr["overall"][h1_key]
        h1_passed = r["spearman_rho"] < min_rho and r["spearman_p"] < alpha

    h2_passed = False
    pair_key = "slod_weighted_vs_chunks_only"
    metric = "normalized_jump_rate"
    if pair_key in cond.get("pairwise_tests", {}):
        pt = cond["pairwise_tests"][pair_key]
        if metric in pt:
            t = pt[metric]
            h2_passed = t["mean_diff"] < 0 and t["wilcoxon_p"] < alpha

    if h1_passed and h2_passed:
        verdict = "CONFIRMED"
        verdict_text = "Both hypotheses are supported by the data."
    elif h1_passed or h2_passed:
        verdict = "PARTIAL"
        verdict_text = "One of the two hypotheses is supported."
    else:
        verdict = "NOT CONFIRMED"
        verdict_text = "Neither hypothesis is supported by the data."

    # Build report
    lines = []
    lines.append("# SH5 Report: Jump Rate as Behavioral Signature\n")
    lines.append(f"**Verdict: {verdict}**\n")
    lines.append(f"**Date:** 2026-03-15\n")
    lines.append("---\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(
        "SH5 tests whether lower abstraction-level jump rates in chain-of-thought "
        "reasoning steps correlate with higher QA correctness, and whether SLoD-routed "
        "retrieval produces more coherent (lower-jump) reasoning traces.\n"
    )
    lines.append(f"**Result:** {verdict_text}\n")

    if h1_key in corr.get("overall", {}):
        r = corr["overall"][h1_key]
        lines.append(
            f"- **H1 (Correlation):** Spearman rho = {r['spearman_rho']:.4f}, "
            f"p = {r['spearman_p']:.2e}, 95% CI = {r.get('bootstrap_ci_95', 'N/A')} "
            f"{'(PASSED)' if h1_passed else '(NOT PASSED)'}\n"
        )

    if pair_key in cond.get("pairwise_tests", {}):
        pt = cond["pairwise_tests"][pair_key]
        if metric in pt:
            t = pt[metric]
            lines.append(
                f"- **H2 (Condition Difference):** slod_weighted NJR mean = {t['routed_mean']:.4f}, "
                f"chunks_only NJR mean = {t['unrouted_mean']:.4f}, "
                f"Wilcoxon p = {t['wilcoxon_p']:.2e} "
                f"{'(PASSED)' if h2_passed else '(NOT PASSED)'}\n"
            )

    lines.append("\n---\n")

    # Data and Methodology
    lines.append("## Data and Methodology\n")
    lines.append(f"- **Questions sampled:** {corr.get('n_total', 'N/A')} (after filtering)\n")
    lines.append(f"- **Conditions:** {', '.join(config['retrieval']['conditions'])}\n")
    lines.append(f"- **Top-k:** {config['retrieval']['top_k']}\n")
    lines.append(f"- **CoT model:** {config['cot']['model']}\n")
    lines.append(f"- **SLoD probe:** SciBERT + LogReg (C={config['probe']['logreg_C']})\n")
    lines.append(f"- **Min steps filter:** {config['cot']['min_steps_filter']}\n")
    lines.append(f"- **Bootstrap resamples:** {config['statistics']['bootstrap_n']}\n")

    if len(jump_df) > 0:
        lines.append(f"\n### Trace Statistics\n")
        lines.append(f"- Total traces: {len(jump_df)}\n")
        lines.append(f"- Mean steps per trace: {jump_df['n_steps'].mean():.1f}\n")
        lines.append(f"- Median steps per trace: {jump_df['n_steps'].median():.0f}\n")

    lines.append("\n---\n")

    # Correlation Results Table
    lines.append("## Correlation Results\n")
    lines.append("### Overall Spearman Correlations\n")
    lines.append("| Jump Metric | vs Token-F1 (rho) | p-value | vs Attribution-F1 (rho) | p-value |\n")
    lines.append("|---|---|---|---|---|\n")

    from src.correlate import JUMP_METRICS
    for jm in JUMP_METRICS:
        row = f"| {jm} "
        for sm in ["answer_token_f1", "attribution_f1"]:
            key = f"{jm}_vs_{sm}"
            if key in corr.get("overall", {}):
                r = corr["overall"][key]
                sig = "*" if r["spearman_p"] < 0.05 else ""
                row += f"| {r['spearman_rho']:.4f}{sig} | {r['spearman_p']:.2e} "
            else:
                row += "| N/A | N/A "
        row += "|\n"
        lines.append(row)

    lines.append("\n*Significance: * p < 0.05*\n")

    # By answer type
    if corr.get("by_answer_type"):
        lines.append("\n### Correlations by Answer Type\n")
        lines.append("| Answer Type | NJR vs Token-F1 (rho) | p-value | n |\n")
        lines.append("|---|---|---|---|\n")
        for atype, results in sorted(corr["by_answer_type"].items()):
            key = "normalized_jump_rate_vs_answer_token_f1"
            if key in results:
                r = results[key]
                sig = "*" if r["spearman_p"] < 0.05 else ""
                lines.append(
                    f"| {atype} | {r['spearman_rho']:.4f}{sig} | {r['spearman_p']:.2e} | {r['n']} |\n"
                )

    lines.append("\n---\n")

    # Condition Comparison
    lines.append("## Condition Comparison\n")
    lines.append("### Mean Jump Metrics by Condition\n")
    lines.append("| Condition | NJR Mean | NJR Median | Mean Steps | n |\n")
    lines.append("|---|---|---|---|---|\n")
    for cond_name, summary in sorted(cond.get("condition_summaries", {}).items()):
        if "normalized_jump_rate" in summary:
            njr = summary["normalized_jump_rate"]
            ns = summary.get("jump_count", {}).get("n", "?")
            n_steps_mean = "?"
            if "jump_count" in summary:
                # n_steps not directly in summary, get from jump_df
                pass
            lines.append(
                f"| {cond_name} | {njr['mean']:.4f} | {njr['median']:.4f} | — | {njr['n']} |\n"
            )

    lines.append("\n### Pairwise Wilcoxon Tests (NJR)\n")
    lines.append("| Comparison | Routed Mean | Unrouted Mean | Diff | p-value | Effect (r) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for pair_name, tests in sorted(cond.get("pairwise_tests", {}).items()):
        if metric in tests:
            t = tests[metric]
            sig = "*" if t["wilcoxon_p"] < 0.05 else ""
            lines.append(
                f"| {pair_name} | {t['routed_mean']:.4f} | {t['unrouted_mean']:.4f} "
                f"| {t['mean_diff']:.4f} | {t['wilcoxon_p']:.2e}{sig} "
                f"| {t['rank_biserial_r']:.4f} |\n"
            )

    lines.append("\n---\n")

    # Regression
    if "regression" in corr and "error" not in corr["regression"]:
        reg = corr["regression"]
        lines.append("## Regression Analysis\n")
        lines.append(
            f"**Model:** answer_token_f1 ~ normalized_jump_rate + n_steps + condition + answer_type\n\n"
        )
        lines.append(f"- R-squared: {reg['r_squared']:.4f}\n")
        lines.append(f"- n samples: {reg['n_samples']}\n\n")
        lines.append("| Feature | Coefficient |\n")
        lines.append("|---|---|\n")
        for feat, coef in sorted(reg["coefficients"].items()):
            lines.append(f"| {feat} | {coef:.6f} |\n")

    lines.append("\n---\n")

    # Figures
    lines.append("## Figures\n")
    figures = [
        ("jump_rate_by_condition.png", "Normalized Jump Rate by Condition"),
        ("jump_vs_correctness_scatter.png", "Jump Rate vs Answer Correctness"),
        ("slod_sequence_examples.png", "Example SLoD Sequences"),
        ("correlation_heatmap.png", "Correlation Heatmap"),
        ("metric_distributions.png", "Jump Metric Distributions"),
        ("answer_type_breakdown.png", "Correlations by Answer Type"),
    ]
    for fname, title in figures:
        lines.append(f"### {title}\n")
        lines.append(f"![{title}](figures/{fname})\n\n")

    lines.append("---\n")

    # Discussion
    lines.append("## Discussion\n")
    lines.append(
        "This experiment tested whether the smoothness of abstraction-level transitions "
        "in chain-of-thought reasoning (as measured by the SH1 SLoD probe) predicts "
        "answer quality, and whether SLoD-routed retrieval (from SH3) naturally induces "
        "smoother reasoning traces.\n\n"
    )

    lines.append("### Limitations\n")
    lines.append("- SLoD probe accuracy is 0.72 macro-F1, introducing classification noise\n")
    lines.append("- CoT steps are a different domain from the SH1 training data (paper sections vs reasoning steps)\n")
    lines.append("- Token-F1 is a noisy measure of answer correctness\n")
    lines.append("- Only 4 retrieval conditions tested at k=5\n")

    lines.append("\n### Implications for SLoD Thesis\n")
    if verdict == "CONFIRMED":
        lines.append(
            "SH5 provides behavioral evidence that SLoD-aware retrieval not only improves "
            "retrieval metrics (SH3) but also produces more coherent reasoning traces. "
            "This closes the argument loop: mechanism (SH1) -> utility (SH3) -> behavior (SH5).\n"
        )
    elif verdict == "PARTIAL":
        lines.append(
            "SH5 provides partial behavioral evidence. The relationship between abstraction "
            "coherence and answer quality exists but the full causal chain is not fully confirmed.\n"
        )
    else:
        lines.append(
            "SH5 did not find strong evidence that abstraction-level jumps predict answer "
            "quality or that SLoD routing reduces jumps. This may reflect domain shift in the "
            "SLoD probe, noise in the correctness metric, or a genuine null result.\n"
        )

    report_text = "".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report written to {report_path}")
    return report_path


def generate_all_visualizations(config: dict) -> list[Path]:
    """Orchestrate all visualization and report generation.

    Returns list of generated file paths.
    """
    data_dir = resolve_path(config, config["paths"]["data_dir"])

    # Load data
    jump_df = pd.DataFrame(load_jsonl(data_dir / "jump_metrics.jsonl"))
    score_df = pd.DataFrame(load_jsonl(data_dir / "answer_scores.jsonl"))

    min_steps = config["cot"]["min_steps_filter"]
    jump_filtered = jump_df[jump_df["n_steps"] >= min_steps].copy()

    merged = jump_filtered.merge(score_df, on=["question_id", "condition"], how="inner")

    # Load correlation results
    corr_path = data_dir / "results" / "correlation_results.json"
    with open(corr_path) as f:
        corr_results = json.load(f)

    generated = []

    # Generate all plots
    try:
        generated.append(plot_jump_by_condition(jump_filtered, config))
    except Exception as e:
        logger.error(f"Failed to plot jump_by_condition: {e}")

    try:
        generated.append(plot_jump_vs_correctness(merged, config))
    except Exception as e:
        logger.error(f"Failed to plot jump_vs_correctness: {e}")

    try:
        generated.append(plot_slod_sequences(jump_df, score_df, config))
    except Exception as e:
        logger.error(f"Failed to plot slod_sequences: {e}")

    try:
        generated.append(plot_correlation_heatmap(corr_results, config))
    except Exception as e:
        logger.error(f"Failed to plot correlation_heatmap: {e}")

    try:
        generated.append(plot_metric_distributions(jump_filtered, config))
    except Exception as e:
        logger.error(f"Failed to plot metric_distributions: {e}")

    try:
        generated.append(plot_answer_type_breakdown(corr_results, config))
    except Exception as e:
        logger.error(f"Failed to plot answer_type_breakdown: {e}")

    # Generate report
    try:
        generated.append(generate_report(config))
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")

    logger.info(f"Generated {len(generated)} visualization artifacts")
    return generated
