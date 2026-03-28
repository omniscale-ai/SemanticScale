#!/usr/bin/env python3
"""Stage F: Visualization and report generation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.slod_axis import project_onto_axis
from src.visualization import (
    plot_slod_axis_validation, plot_cot_projections,
    plot_correlation_comparison, plot_scatter, plot_ablation
)

def main():
    parser = argparse.ArgumentParser(description="Stage F: Report")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = Path('data')
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(data_dir / 'trace_features.parquet')
    corr_df = pd.read_csv(data_dir / 'correlation_table.csv')
    with open(data_dir / 'analysis_results.json') as f:
        results = json.load(f)

    # Load SLoD axis and embeddings for projection plots
    slod_data = np.load(data_dir / 'slod_axis.npz', allow_pickle=True)
    centroid_axis = slod_data['centroid_axis']
    step_embs = np.load(data_dir / 'step_embeddings.npz')
    step_embeddings = step_embs['embeddings']

    sh1 = np.load('data_sh1/embeddings/scibert_length_matched.npz')
    sh1_emb = sh1['embeddings']
    sh1_labels = sh1['labels']
    with open('data_sh1/splits.json') as f:
        splits = json.load(f)
    test_idx = np.array(splits['test'])
    test_emb = sh1_emb[test_idx]
    test_labels = sh1_labels[test_idx]

    # 1. SLoD axis validation (already done in stage C, but regenerate)
    test_proj = project_onto_axis(test_emb, centroid_axis)
    plot_slod_axis_validation(test_proj, test_labels,
                               str(figures_dir / 'slod_axis_validation.png'))

    # 2. CoT projections overlay
    cot_proj = project_onto_axis(step_embeddings, centroid_axis)
    plot_cot_projections(cot_proj, test_proj, test_labels,
                          str(figures_dir / 'cot_projections_overlay.png'))

    # 3. Best feature scatter plots
    baselines = results.get('baselines', {})
    for metric in ['answer_token_f1', 'attribution_f1']:
        metric_corr = corr_df[corr_df['metric'] == metric].copy()
        if len(metric_corr) > 0:
            best_idx = metric_corr['rho'].abs().idxmax()
            best_feat = metric_corr.loc[best_idx, 'feature']
            plot_scatter(df, best_feat, metric,
                        str(figures_dir / f'scatter_{metric}.png'))

    # 4. Correlation comparison bar chart
    plot_correlation_comparison(corr_df, baselines,
                                str(figures_dir / 'correlation_comparison.png'))

    # 5. Ablation bar chart
    ablation = results.get('ablation_auroc', {})
    if ablation:
        plot_ablation(ablation, str(figures_dir / 'ablation_auroc.png'))

    # 6. Generate markdown report
    generate_report(df, corr_df, results, figures_dir)
    print("Stage F: DONE")


def generate_report(df, corr_df, results, figures_dir):
    """Auto-generate markdown report."""
    report_path = Path('reports/SH5d_results.md')

    # Extract key numbers
    n_traces = len(df)
    n_sig = results['correlations_summary']['n_significant']
    total_tests = results['correlations_summary']['total_tests']

    lines = [
        "# SH5d Results: Probe-Free Embedding Distance",
        "",
        f"**Traces analyzed:** {n_traces}",
        f"**Significant correlations (Bonferroni):** {n_sig} / {total_tests}",
        "",
        "## Correlation Table",
        "",
        "| Feature | Metric | Spearman rho | p (corrected) | Sig? |",
        "|---------|--------|-------------|---------------|------|",
    ]

    for _, row in corr_df.sort_values('p_corrected').iterrows():
        sig = "YES" if row['significant'] else ""
        lines.append(
            f"| {row['feature']} | {row['metric']} | {row['rho']:+.4f} | {row['p_corrected']:.4e} | {sig} |"
        )

    lines.extend([
        "",
        "## Ablation: AUROC by Feature Group",
        "",
    ])
    ablation = results.get('ablation_auroc', {})
    if ablation:
        lines.append("| Group | token-F1 AUROC | attr-F1 AUROC |")
        lines.append("|-------|---------------|---------------|")
        for group, metrics in ablation.items():
            tf1 = metrics.get('answer_token_f1', float('nan'))
            af1 = metrics.get('attribution_f1', float('nan'))
            lines.append(f"| {group} | {tf1:.4f} | {af1:.4f} |")

    lines.extend([
        "",
        "## Baseline Comparison",
        "",
        "| Method | Best rho (attr-F1) | Best rho (token-F1) |",
        "|--------|-------------------|---------------------|",
        f"| SH5 (jump rate) | — | {results['baselines']['sh5_jump_rate_token_f1_rho']} |",
        f"| SH5a (transition matrix) | {results['baselines']['sh5a_best_attr_f1_rho']} | — |",
        f"| SH5c (alignment) | {results['baselines']['sh5c_best_attr_f1_rho']} | — |",
    ])

    # Add SH5d best
    best_token = results['correlations_summary'].get('best_token_f1', {})
    best_attr = results['correlations_summary'].get('best_attr_f1', {})
    if best_token and best_attr:
        lines.append(
            f"| SH5d (this work) | {best_attr.get('rho', 'N/A')} | {best_token.get('rho', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Figures",
        "",
        "![SLoD Axis Validation](figures/slod_axis_validation.png)",
        "![CoT Projections](figures/cot_projections_overlay.png)",
        "![Correlation Comparison](figures/correlation_comparison.png)",
        "![Ablation AUROC](figures/ablation_auroc.png)",
    ])

    for metric in ['answer_token_f1', 'attribution_f1']:
        fig_path = figures_dir / f'scatter_{metric}.png'
        if fig_path.exists():
            lines.append(f"![Scatter {metric}](figures/scatter_{metric}.png)")

    report_path.write_text('\n'.join(lines))
    print(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()
