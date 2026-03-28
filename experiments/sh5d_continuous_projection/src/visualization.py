"""Visualization functions for SH5d."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_slod_axis_validation(projections: np.ndarray, labels: np.ndarray,
                               save_path: str, title: str = "SLoD Axis Validation (SH1 Test Set)"):
    """Histogram of SH1 test set projections by class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, name, color in [(0, 'Macro', 'C0'), (1, 'Meso', 'C1'), (2, 'Micro', 'C2')]:
        mask = labels == label
        ax.hist(projections[mask], bins=50, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('SLoD Axis Projection')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cot_projections(cot_projections: np.ndarray,
                          sh1_projections: np.ndarray, sh1_labels: np.ndarray,
                          save_path: str):
    """Overlay CoT step projections on SH1 distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, name, color in [(0, 'Macro', 'C0'), (1, 'Meso', 'C1'), (2, 'Micro', 'C2')]:
        mask = sh1_labels == label
        ax.hist(sh1_projections[mask], bins=50, alpha=0.2, color=color, density=True)
    ax.hist(cot_projections, bins=50, alpha=0.6, color='black', density=True,
            label='CoT Steps', histtype='step', linewidth=2)
    ax.set_xlabel('SLoD Axis Projection')
    ax.set_ylabel('Density')
    ax.set_title('CoT Step Projections vs SH1 Reference Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_comparison(corr_df: pd.DataFrame, baselines: dict,
                                 save_path: str):
    """Bar chart comparing correlations across SH5 family."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, metric in enumerate(['answer_token_f1', 'attribution_f1']):
        ax = axes[idx]
        metric_df = corr_df[corr_df['metric'] == metric].copy()

        # Get best SH5d features
        best_5 = metric_df.nlargest(5, 'rho', keep='first') if metric_df['rho'].abs().max() > 0 else metric_df.head(5)
        best_5 = metric_df.reindex(metric_df['rho'].abs().nlargest(5).index)

        # Add baselines
        baseline_data = []
        if metric == 'answer_token_f1':
            baseline_data.append({'feature': 'SH5: jump_rate', 'rho': baselines.get('sh5_jump_rate_token_f1_rho', 0)})
        else:
            baseline_data.append({'feature': 'SH5a: trans_matrix', 'rho': baselines.get('sh5a_best_attr_f1_rho', 0)})
            baseline_data.append({'feature': 'SH5c: alignment', 'rho': baselines.get('sh5c_best_attr_f1_rho', 0)})

        plot_data = pd.concat([
            pd.DataFrame(baseline_data),
            best_5[['feature', 'rho']].rename(columns={'feature': 'feature'})
        ], ignore_index=True)

        colors = ['gray'] * len(baseline_data) + ['steelblue'] * len(best_5)
        ax.barh(range(len(plot_data)), plot_data['rho'], color=colors)
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data['feature'], fontsize=9)
        ax.set_xlabel('Spearman rho')
        ax.set_title(f'Correlations with {metric}')
        ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scatter(df: pd.DataFrame, feature: str, metric: str, save_path: str):
    """Scatter plot of feature vs quality metric."""
    valid = df[[feature, metric]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid[feature], valid[metric], alpha=0.3, s=20)
    ax.set_xlabel(feature)
    ax.set_ylabel(metric)
    ax.set_title(f'{feature} vs {metric}')

    # Add correlation
    from scipy.stats import spearmanr
    rho, p = spearmanr(valid[feature], valid[metric])
    ax.text(0.05, 0.95, f'rho={rho:.3f}, p={p:.1e}', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ablation(ablation_results: dict, save_path: str):
    """Bar chart of AUROC by feature group."""
    groups = list(ablation_results.keys())
    metrics = list(next(iter(ablation_results.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.35

    for i, metric in enumerate(metrics):
        values = [ablation_results[g][metric] for g in groups]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_ylabel('AUROC')
    ax.set_title('Ablation: AUROC by Feature Group')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(groups, rotation=15)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylim(0.4, max(0.7, max(max(ablation_results[g].values()) for g in groups) + 0.05))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
