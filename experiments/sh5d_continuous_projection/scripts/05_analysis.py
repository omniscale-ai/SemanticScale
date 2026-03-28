#!/usr/bin/env python3
"""Stage E: Statistical analysis — correlations, ablation, logistic regression."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.analysis import compute_correlations, run_ablation, logistic_regression_auroc, FEATURE_GROUPS, ALL_FEATURES

def main():
    parser = argparse.ArgumentParser(description="Stage E: Analysis")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = Path('../../data/sh5d')
    output_corr = data_dir / 'correlation_table.csv'
    output_results = data_dir / 'analysis_results.json'

    if output_corr.exists() and output_results.exists() and not args.force:
        print("Stage E outputs exist, skipping.")
        return

    df = pd.read_parquet(data_dir / 'trace_features.parquet')
    print(f"Loaded {len(df)} traces")

    # 1. Spearman correlations
    print("\n=== Spearman Correlations (Bonferroni-corrected) ===")
    corr_df = compute_correlations(df, n_bonferroni=30)
    corr_df = corr_df.sort_values('p_corrected')

    for _, row in corr_df.iterrows():
        sig = "***" if row['significant'] else ""
        print(f"  {row['feature']:30s} vs {row['metric']:20s}: rho={row['rho']:+.4f}  p_corr={row['p_corrected']:.4e} {sig}")

    corr_df.to_csv(output_corr, index=False)

    # 2. Best correlations by group
    print("\n=== Best Correlations by Feature Group ===")
    group_bests = {}
    for group_name, feats in FEATURE_GROUPS.items():
        group_corr = corr_df[corr_df['feature'].isin(feats)]
        for metric in ['answer_token_f1', 'attribution_f1']:
            metric_corr = group_corr[group_corr['metric'] == metric]
            if len(metric_corr) > 0:
                best = metric_corr.loc[metric_corr['rho'].abs().idxmax()]
                key = f"{group_name}_{metric}"
                group_bests[key] = {
                    'feature': best['feature'],
                    'rho': float(best['rho']),
                    'p_corrected': float(best['p_corrected']),
                    'significant': bool(best['significant'])
                }
                print(f"  {group_name:20s} | {metric:20s}: {best['feature']} rho={best['rho']:+.4f}")

    # 3. Ablation: AUROC by feature group
    print("\n=== Ablation: Logistic Regression AUROC ===")
    ablation = run_ablation(df, cv_folds=5, random_seed=42)

    # 4. Combined SH5d + jump metrics (proxy for SH5a/SH5c integration)
    # Load jump metrics if available
    jump_path = data_dir / 'jump_metrics.parquet'
    combined_auroc = {}
    if jump_path.exists():
        jumps = pd.read_parquet(jump_path)
        # Merge jump features
        jump_feats = ['jump_count', 'mean_abs_delta', 'max_jump',
                       'direction_changes', 'normalized_jump_rate']
        available_jump_feats = [f for f in jump_feats if f in jumps.columns]
        merged = df.merge(jumps[['question_id', 'condition'] + available_jump_feats],
                          on=['question_id', 'condition'], how='left')
        combined_feats = ALL_FEATURES + available_jump_feats
        for metric in ['answer_token_f1', 'attribution_f1']:
            auroc = logistic_regression_auroc(merged, combined_feats, metric, cv_folds=5)
            combined_auroc[metric] = auroc
            print(f"  Combined (SH5d + jump): {metric}: AUROC = {auroc:.4f}")
        ablation['combined_with_jumps'] = combined_auroc

    # 5. Compile results
    results = {
        'correlations_summary': {
            'best_token_f1': corr_df[corr_df['metric'] == 'answer_token_f1'].loc[
                corr_df[corr_df['metric'] == 'answer_token_f1']['rho'].abs().idxmax()
            ].to_dict() if len(corr_df[corr_df['metric'] == 'answer_token_f1']) > 0 else {},
            'best_attr_f1': corr_df[corr_df['metric'] == 'attribution_f1'].loc[
                corr_df[corr_df['metric'] == 'attribution_f1']['rho'].abs().idxmax()
            ].to_dict() if len(corr_df[corr_df['metric'] == 'attribution_f1']) > 0 else {},
            'n_significant': int(corr_df['significant'].sum()),
            'total_tests': len(corr_df),
        },
        'group_bests': group_bests,
        'ablation_auroc': ablation,
        'baselines': {
            'sh5_jump_rate_token_f1_rho': 0.003,
            'sh5a_best_attr_f1_rho': -0.197,
            'sh5c_best_attr_f1_rho': -0.135,
        }
    }

    with open(output_results, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {output_corr}, {output_results}")
    print("Stage E: DONE")

if __name__ == '__main__':
    main()
