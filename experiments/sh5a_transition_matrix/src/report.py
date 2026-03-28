"""Stage E: Auto-generate markdown report from results."""

import json
from pathlib import Path


def generate_report(results_dir: str = "data/results",
                    figures_dir: str = "reports/figures",
                    output_path: str = "reports/SH5a_auto_report.md") -> None:
    """Generate a markdown report summarizing all results."""

    # Load all results
    with open(f"{results_dir}/summary.json") as f:
        summary = json.load(f)

    with open(f"{results_dir}/correlation_results.json") as f:
        corr_data = json.load(f)

    with open(f"{results_dir}/clustering_results.json") as f:
        clust_data = json.load(f)

    with open(f"{results_dir}/condition_comparison.json") as f:
        cond_data = json.load(f)

    with open(f"{results_dir}/logistic_results.json") as f:
        log_data = json.load(f)

    lines = []
    lines.append("# SH5a Auto-Generated Report: Transition Matrix Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Overall Verdict: {summary['overall_verdict']}**")
    lines.append("")
    lines.append(f"- H1 (Transition Signature): {'PASS' if summary['H1_transition_signature']['pass'] else 'FAIL'}")
    lines.append(f"- H2 (Pattern Clusters): {'PASS' if summary['H2_pattern_clusters']['pass'] else 'FAIL'}")
    lines.append(f"- H3 (Condition Effect): {'PASS' if summary['H3_condition_effect']['pass'] else 'FAIL'}")
    lines.append("")

    # H1 details
    lines.append("## H1: Transition Feature Correlations")
    lines.append("")
    h1 = summary["H1_transition_signature"]
    lines.append(f"Bonferroni-corrected significance threshold: p < {corr_data['bonferroni_alpha']:.6f}")
    lines.append(f"Number of comparisons: {corr_data['n_comparisons']}")
    lines.append(f"**Significant features: {h1['n_significant']}**")
    lines.append("")

    # Table of top correlations
    lines.append("### Top 20 Correlations (per-trace, N=2000)")
    lines.append("")
    lines.append("| Feature | Target | rho | p-value | Bonf. p | Sig? |")
    lines.append("|---------|--------|-----|---------|---------|------|")
    for r in corr_data["per_trace"][:20]:
        sig = "Yes" if r["significant_bonferroni"] else ""
        lines.append(
            f"| {r['feature']} | {r['target']} | {r['rho']:+.4f} | {r['p_value']:.2e} | {r['p_bonferroni']:.4f} | {sig} |"
        )
    lines.append("")

    # Significant ones only
    sig_results = [r for r in corr_data["per_trace"] if r["significant_bonferroni"]]
    if sig_results:
        lines.append("### Bonferroni-Significant Correlations")
        lines.append("")
        lines.append("| Feature | Target | rho | p-value |")
        lines.append("|---------|--------|-----|---------|")
        for r in sig_results:
            lines.append(f"| {r['feature']} | {r['target']} | {r['rho']:+.4f} | {r['p_value']:.2e} |")
        lines.append("")

    # H2 details
    lines.append("## H2: Clustering Analysis")
    lines.append("")
    h2 = summary["H2_pattern_clusters"]
    clust_res = clust_data["per_trace_soft"]
    lines.append(f"Best k: {h2['best_k']} (silhouette: {h2['silhouette']:.4f})")
    lines.append(f"ANOVA on token-F1: p = {h2['anova_token_f1_p']:.4e}")
    lines.append(f"ANOVA on attribution-F1: p = {h2['anova_attribution_f1_p']:.4e}")
    lines.append("")

    best_k_str = str(h2["best_k"])
    if best_k_str in clust_res["results_by_k"]:
        res_k = clust_res["results_by_k"][best_k_str]
        lines.append("### Cluster Profiles")
        lines.append("")
        lines.append("| Cluster | Size | Mean Token-F1 | Mean Attribution-F1 |")
        lines.append("|---------|------|---------------|---------------------|")
        for ci in range(h2["best_k"]):
            cq = res_k["cluster_quality"][str(ci)]
            lines.append(f"| {ci} | {cq['size']} | {cq['mean_token_f1']:.4f} | {cq['mean_attribution_f1']:.4f} |")
        lines.append("")

    lines.append("![Cluster Profiles](figures/cluster_profiles_soft.png)")
    lines.append("")
    lines.append("![Cluster Quality](figures/cluster_quality_boxplot.png)")
    lines.append("")

    # H3 details
    lines.append("## H3: Condition Comparison")
    lines.append("")
    h3 = summary["H3_condition_effect"]
    lines.append(f"Chi-squared (4 conditions): chi2 = {h3['chi2']:.2f}, p = {h3['p']:.4e}")
    lines.append(f"Chi-squared (routed vs unrouted): p = {h3['routed_vs_unrouted_p']:.4e}")
    lines.append("")
    lines.append("![Condition Heatmaps (soft)](figures/condition_heatmaps_soft.png)")
    lines.append("")
    lines.append("![Condition Heatmaps (hard)](figures/condition_heatmaps_hard.png)")
    lines.append("")

    # Logistic regression
    lines.append("## Logistic Regression")
    lines.append("")
    for target, res in [("Token-F1", log_data["token_f1"]), ("Attribution-F1", log_data["attribution_f1"])]:
        if "error" in res:
            lines.append(f"**{target}**: {res['error']}")
        else:
            lines.append(f"**{target}**: CV accuracy = {res['cv_accuracy_mean']:.4f} (+/- {res['cv_accuracy_std']:.4f}), AUC = {res['cv_auc_mean']:.4f} (+/- {res['cv_auc_std']:.4f})")
            lines.append("")
            lines.append(f"Top features for {target}:")
            lines.append("")
            lines.append("| Feature | Coefficient |")
            lines.append("|---------|-------------|")
            for ft in res["top_features"][:5]:
                lines.append(f"| {ft['feature']} | {ft['coefficient']:+.4f} |")
        lines.append("")

    # SH5 comparison
    lines.append("## Comparison with SH5")
    lines.append("")
    comp = summary["comparison_with_sh5"]
    lines.append("| Metric | SH5 (jump rate) | SH5a (best transition) |")
    lines.append("|--------|-----------------|------------------------|")
    lines.append(f"| Token-F1 rho | {comp['sh5_jump_rate_token_f1_rho']:+.3f} | {comp.get('sh5a_best_token_f1_rho', 'N/A')} |")
    lines.append(f"| Attribution-F1 rho | {comp['sh5_jump_rate_attribution_f1_rho']:+.3f} | {comp.get('sh5a_best_attribution_f1_rho', 'N/A')} |")
    lines.append("")
    lines.append("![SH5 Comparison](figures/sh5_comparison.png)")
    lines.append("")
    lines.append("![Correlation Bars](figures/correlation_bars.png)")
    lines.append("")

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")
