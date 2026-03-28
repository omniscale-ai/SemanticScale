"""Stage E: Visualization.

Creates all figures for the SH5c analysis.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(alignment_path: Path) -> pd.DataFrame:
    rows = []
    with open(alignment_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_alignment_gap_by_condition(df: pd.DataFrame, output_path: Path,
                                    figsize=(10, 6), dpi=150):
    """Violin/box plot of mean_alignment_gap per condition."""
    fig, ax = plt.subplots(figsize=figsize)

    conditions = sorted(df["condition"].unique())
    data_by_cond = [df[df["condition"] == c]["mean_alignment_gap"].dropna().values
                    for c in conditions]

    parts = ax.violinplot(data_by_cond, positions=range(len(conditions)),
                          showmeans=True, showmedians=True)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("Mean Alignment Gap")
    ax.set_title("Context-Reasoning Alignment Gap by Retrieval Condition")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_scatter_alignment_vs_quality(df: pd.DataFrame, output_path: Path,
                                      feat: str = "mean_alignment_gap",
                                      metric: str = "answer_token_f1",
                                      figsize=(10, 6), dpi=150):
    """Scatter plot of alignment feature vs quality metric."""
    fig, ax = plt.subplots(figsize=figsize)

    conditions = sorted(df["condition"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))

    for cond, color in zip(conditions, colors):
        sub = df[df["condition"] == cond].dropna(subset=[feat, metric])
        ax.scatter(sub[feat], sub[metric], c=[color], alpha=0.4,
                   s=20, label=cond)

    # Overall regression line
    valid = df[[feat, metric]].dropna()
    if len(valid) > 10:
        z = np.polyfit(valid[feat], valid[metric], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid[feat].min(), valid[feat].max(), 100)
        ax.plot(x_range, p(x_range), "k--", alpha=0.5, label=f"trend (slope={z[0]:.3f})")

    ax.set_xlabel(feat.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{feat} vs {metric}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_context_reasoning_heatmap(df: pd.DataFrame, output_path: Path,
                                    figsize=(8, 6), dpi=150):
    """Heatmap of dominant context SLoD vs dominant reasoning SLoD."""
    fig, ax = plt.subplots(figsize=figsize)

    levels = ["macro", "meso", "micro"]
    level_map = {"macro": 0, "meso": 1, "micro": 2}

    # Get dominant context and reasoning levels
    # Context: from context_slod_mean (round to nearest int)
    # Reasoning: use dominant_level from reasoning distribution
    valid = df.dropna(subset=["context_slod_mean", "mean_alignment_gap"])

    # Build count matrix
    matrix = np.zeros((3, 3))
    for _, row in valid.iterrows():
        ctx_level = int(round(row["context_slod_mean"]))
        ctx_level = max(0, min(2, ctx_level))

        # Reasoning dominant: need to infer from available data
        # We don't have reasoning_slod_distribution directly, but we can use
        # the mean as a proxy
        rsn_mean = row.get("context_slod_mean", 1)  # fallback
        # Better: compute from alignment gap direction
        # Actually, we need to store reasoning_slod_mean in alignment features
        # Use what we have: if mean_alignment_gap is low, they match
        # For now, approximate from context_slod_mean and mean_alignment_gap
        # This is a simplification
        pass

    # Alternative: just do context_slod_mean binned vs answer quality binned
    # Actually, let's do condition x quality heatmap instead — more informative
    # Let me compute dominant levels properly using context_slod_mean

    # Bin context_slod_mean into 3 levels
    bins_ctx = pd.cut(valid["context_slod_mean"], bins=[-0.5, 0.5, 1.5, 2.5],
                      labels=["macro", "meso", "micro"])

    # For reasoning, we need the data. Let's compute from the features we have.
    # dominant_level_match tells us if they agree. Let's plot match rate per context level instead.

    # Better approach: use dominant_level_match as a function of condition
    match_rates = valid.groupby("condition")["dominant_level_match"].mean()

    # Clear and replot as a simpler, more informative visualization
    ax.clear()
    conditions = sorted(match_rates.index)
    values = [match_rates[c] for c in conditions]
    bars = ax.bar(conditions, values, color=plt.cm.Set2(np.linspace(0, 1, len(conditions))))

    ax.set_ylabel("Dominant Level Match Rate")
    ax.set_title("Rate of Context-Reasoning SLoD Level Agreement by Condition")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_feature_importance(results_dir: Path, output_path: Path,
                            figsize=(10, 6), dpi=150):
    """Bar chart of logistic regression coefficients."""
    logistic_path = results_dir / "logistic.json"
    if not logistic_path.exists():
        print(f"Skipping feature importance plot — {logistic_path} not found")
        return

    with open(logistic_path) as f:
        logistic = json.load(f)

    # Use alignment_only model
    model_data = logistic.get("alignment_only", {})
    importance = model_data.get("feature_importance", {})

    if not importance:
        print("No feature importance data available")
        return

    fig, ax = plt.subplots(figsize=figsize)

    features = sorted(importance.keys(), key=lambda x: abs(importance[x]), reverse=True)
    values = [importance[f] for f in features]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace("_", " ") for f in features], fontsize=9)
    ax.set_xlabel("Logistic Regression Coefficient (standardized)")
    ax.set_title("Feature Importance: Predicting Above-Median Token-F1")
    ax.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_subgroup_correlations(results_dir: Path, output_path: Path,
                               figsize=(12, 6), dpi=150):
    """Grouped bar chart of correlations by answer type."""
    subgroups_path = results_dir / "subgroups.json"
    if not subgroups_path.exists():
        print(f"Skipping subgroup plot — {subgroups_path} not found")
        return

    with open(subgroups_path) as f:
        subgroups = json.load(f)

    by_type = subgroups.get("by_answer_type", {})
    if not by_type:
        return

    fig, ax = plt.subplots(figsize=figsize)

    features = ["mean_alignment_gap", "jsd", "weighted_alignment_gap"]
    metric = "answer_token_f1"
    atypes = sorted(by_type.keys())

    x = np.arange(len(features))
    width = 0.25
    colors = plt.cm.Set2(np.linspace(0, 1, len(atypes)))

    for i, atype in enumerate(atypes):
        rhos = []
        for feat in features:
            r = by_type[atype]["correlations"].get(feat, {}).get(metric, {})
            rhos.append(r.get("rho", 0))
        ax.bar(x + i * width, rhos, width, label=atype, color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels([f.replace("_", " ") for f in features], fontsize=9)
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"Alignment-Quality Correlation by Answer Type ({metric})")
    ax.legend()
    ax.axhline(y=0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_roc_curves(results_dir: Path, output_path: Path,
                    figsize=(8, 6), dpi=150):
    """ROC curves placeholder — we only have AUROC numbers, not full curves.
    Plot AUROC comparison bar chart instead."""
    logistic_path = results_dir / "logistic.json"
    if not logistic_path.exists():
        return

    with open(logistic_path) as f:
        logistic = json.load(f)

    fig, ax = plt.subplots(figsize=figsize)

    models = ["alignment_only", "jump_only", "combined"]
    aurocs = [logistic.get(m, {}).get("auroc_mean", 0) for m in models]
    stds = [logistic.get(m, {}).get("auroc_std", 0) for m in models]

    colors = ["#3498db", "#e67e22", "#2ecc71"]
    bars = ax.bar(models, aurocs, yerr=stds, capsize=5, color=colors)

    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_title("Logistic Regression: Predicting Above-Median Token-F1")
    ax.axhline(y=0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    ax.axhline(y=0.6, color="r", linestyle="--", alpha=0.5, label="H3 threshold")
    ax.set_ylim(0.4, 0.8)
    ax.legend()

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def generate_all_plots(alignment_path: Path, results_dir: Path,
                       figures_dir: Path, config: dict):
    """Generate all visualization figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    figsize = tuple(config.get("visualization", {}).get("figsize", [10, 6]))
    dpi = config.get("visualization", {}).get("dpi", 150)

    df = load_data(alignment_path)
    print(f"Loaded {len(df)} traces for visualization")

    # 1. Alignment gap by condition
    plot_alignment_gap_by_condition(df, figures_dir / "alignment_gap_by_condition.png",
                                    figsize=figsize, dpi=dpi)

    # 2. Scatter: alignment gap vs token-F1
    plot_scatter_alignment_vs_quality(df, figures_dir / "alignment_vs_quality.png",
                                      feat="mean_alignment_gap",
                                      metric="answer_token_f1",
                                      figsize=figsize, dpi=dpi)

    # 3. Scatter: JSD vs token-F1
    plot_scatter_alignment_vs_quality(df, figures_dir / "jsd_vs_quality.png",
                                      feat="jsd",
                                      metric="answer_token_f1",
                                      figsize=figsize, dpi=dpi)

    # 4. Context-reasoning heatmap / match rate
    plot_context_reasoning_heatmap(df, figures_dir / "context_reasoning_match.png",
                                   figsize=(8, 6), dpi=dpi)

    # 5. Feature importance
    plot_feature_importance(results_dir, figures_dir / "feature_importance.png",
                            figsize=figsize, dpi=dpi)

    # 6. Subgroup correlations
    plot_subgroup_correlations(results_dir, figures_dir / "subgroup_correlations.png",
                               figsize=(12, 6), dpi=dpi)

    # 7. ROC / AUROC comparison
    plot_roc_curves(results_dir, figures_dir / "auroc_comparison.png",
                    figsize=(8, 6), dpi=dpi)

    print(f"\nAll figures saved to {figures_dir}")
