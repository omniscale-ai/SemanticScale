"""Stage E: Visualization — transition heatmaps, cluster profiles, correlation charts."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


LEVEL_NAMES = ["macro", "meso", "micro"]


def plot_condition_heatmaps(condition_data: dict, matrix_type: str,
                            output_path: str) -> None:
    """Plot 2x2 grid of transition heatmaps, one per condition."""
    conditions = ["chunks_only", "naive_hybrid", "slod_weighted", "slod_weighted_parent"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Transition Matrices by Condition ({matrix_type})", fontsize=14, y=0.98)

    for idx, (cond, ax) in enumerate(zip(conditions, axes.flatten())):
        flat = condition_data[cond]["matrix_flat"]
        mat = np.array(flat).reshape(3, 3)

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(0.4, mat.max()))
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LEVEL_NAMES, fontsize=9)
        ax.set_yticklabels(LEVEL_NAMES, fontsize=9)
        ax.set_xlabel("To")
        ax.set_ylabel("From")
        ax.set_title(cond.replace("_", " ").title(), fontsize=10)

        # Annotate cells
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if mat[i, j] > mat.max() * 0.6 else "black")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_correlation_bars(corr_results: list[dict], output_path: str, top_n: int = 20) -> None:
    """Bar chart of top correlations by |rho|."""
    top = corr_results[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = [f"{r['feature']}\nvs {r['target']}" for r in top]
    rhos = [r["rho"] for r in top]
    colors = ["#2ecc71" if r["significant_bonferroni"] else "#e74c3c" for r in top]

    y_pos = range(len(top))
    ax.barh(y_pos, rhos, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Spearman rho")
    ax.set_title(f"Top {top_n} Correlations (green = Bonferroni significant)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_cluster_profiles(clustering_result: dict, output_path: str) -> None:
    """Heatmap of mean transition matrix per cluster."""
    best_k = clustering_result["best_k"]
    res = clustering_result["results_by_k"][str(best_k)]

    fig, axes = plt.subplots(1, best_k, figsize=(4 * best_k, 4))
    if best_k == 1:
        axes = [axes]
    fig.suptitle(f"Cluster Profiles (k={best_k}, {clustering_result['matrix_type']})", fontsize=13)

    for ci in range(best_k):
        ax = axes[ci]
        center = np.array(res["cluster_quality"][str(ci)]["center"]).reshape(3, 3)
        im = ax.imshow(center, cmap="YlOrRd", vmin=center.min(), vmax=center.max())
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LEVEL_NAMES, fontsize=9)
        ax.set_yticklabels(LEVEL_NAMES, fontsize=9)
        cq = res["cluster_quality"][str(ci)]
        ax.set_title(
            f"Cluster {ci} (n={cq['size']})\n"
            f"F1={cq['mean_token_f1']:.3f}, attr={cq['mean_attribution_f1']:.3f}",
            fontsize=9
        )
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{center[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if center[i, j] > center.max() * 0.6 else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_cluster_quality_boxplot(features: list[dict], clustering_result: dict,
                                 output_path: str) -> None:
    """Boxplot of quality metrics per cluster."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    best_k = clustering_result["best_k"]
    matrix_key = clustering_result["matrix_type"]

    cell_names = [f"{matrix_key}_{cn}" for cn in [
        "macro->macro", "macro->meso", "macro->micro",
        "meso->macro", "meso->meso", "meso->micro",
        "micro->macro", "micro->meso", "micro->micro",
    ]]

    X = np.array([[f[cn] for cn in cell_names] for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    token_f1_key = "answer_token_f1" if "answer_token_f1" in features[0] else "mean_token_f1"
    attr_f1_key = "attribution_f1" if "attribution_f1" in features[0] else "mean_attribution_f1"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Token F1
    data_token = [
        [f[token_f1_key] for f, l in zip(features, labels) if l == ci]
        for ci in range(best_k)
    ]
    ax1.boxplot(data_token, labels=[f"C{i}" for i in range(best_k)])
    ax1.set_title("Token F1 by Cluster")
    ax1.set_ylabel("Token F1")

    # Attribution F1
    data_attr = [
        [f[attr_f1_key] for f, l in zip(features, labels) if l == ci]
        for ci in range(best_k)
    ]
    ax2.boxplot(data_attr, labels=[f"C{i}" for i in range(best_k)])
    ax2.set_title("Attribution F1 by Cluster")
    ax2.set_ylabel("Attribution F1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_sh5_comparison(summary: dict, output_path: str) -> None:
    """Bar chart comparing SH5 jump_rate rho vs SH5a best transition feature rho."""
    comp = summary["comparison_with_sh5"]

    labels = ["Token F1", "Attribution F1"]
    sh5_vals = [comp["sh5_jump_rate_token_f1_rho"], comp["sh5_jump_rate_attribution_f1_rho"]]
    sh5a_vals = [
        comp.get("sh5a_best_token_f1_rho", 0) or 0,
        comp.get("sh5a_best_attribution_f1_rho", 0) or 0,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, [abs(v) for v in sh5_vals], width, label="SH5 (jump rate)", color="#3498db")
    bars2 = ax.bar(x + width / 2, [abs(v) for v in sh5a_vals], width, label="SH5a (best transition)", color="#e74c3c")

    ax.set_ylabel("|Spearman rho|")
    ax.set_title("SH5 vs SH5a: Best Correlation with Quality")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Annotate
    for bar, val in zip(bars1, sh5_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:+.3f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, sh5a_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:+.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def generate_all_plots(results_dir: str = "../../data/sh5a/results",
                       figures_dir: str = "reports/figures") -> None:
    """Generate all visualization plots."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    with open(f"{results_dir}/condition_comparison.json") as f:
        cond_data = json.load(f)

    with open(f"{results_dir}/correlation_results.json") as f:
        corr_data = json.load(f)

    with open(f"{results_dir}/clustering_results.json") as f:
        clust_data = json.load(f)

    with open(f"{results_dir}/summary.json") as f:
        summary = json.load(f)

    # 1. Condition heatmaps (soft)
    plot_condition_heatmaps(
        cond_data["soft"]["per_condition"], "soft",
        f"{figures_dir}/condition_heatmaps_soft.png"
    )

    # 2. Condition heatmaps (hard)
    plot_condition_heatmaps(
        cond_data["hard"]["per_condition"], "hard",
        f"{figures_dir}/condition_heatmaps_hard.png"
    )

    # 3. Correlation bars (per-trace)
    plot_correlation_bars(
        corr_data["per_trace"],
        f"{figures_dir}/correlation_bars.png"
    )

    # 4. Cluster profiles (soft, per-trace)
    plot_cluster_profiles(
        clust_data["per_trace_soft"],
        f"{figures_dir}/cluster_profiles_soft.png"
    )

    # 5. Cluster quality boxplot
    features = []
    with open("../../data/sh5a/features.jsonl") as f:
        for line in f:
            if line.strip():
                features.append(json.loads(line))

    plot_cluster_quality_boxplot(
        features, clust_data["per_trace_soft"],
        f"{figures_dir}/cluster_quality_boxplot.png"
    )

    # 6. SH5 comparison
    plot_sh5_comparison(summary, f"{figures_dir}/sh5_comparison.png")

    print(f"\nAll plots saved to {figures_dir}/")
