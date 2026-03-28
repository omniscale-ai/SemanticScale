#!/usr/bin/env python3
"""Stage 7c: Generate figures and report for summarization experiment.

Figures (saved to reports/figures/ with _summ suffix):
1. slod_shift_distribution_summ.png
2. alpha_sensitivity_summ.png
3. layer_comparison_summ.png
4. quality_preservation_summ.png
5. surface_metrics_summ.png
6. example_outputs_summ.png

Report: reports/SH2_summ_results.md

Output: reports/SH2_summ_results.md
        reports/figures/*_summ.png
"""
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_json, load_jsonl
from src.visualization import (
    setup_style,
    plot_slod_shift_distribution,
    plot_alpha_sensitivity,
    plot_layer_comparison,
    plot_quality_preservation,
    plot_surface_metrics,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_example_outputs(examples: list, output_path: str):
    """Side-by-side text display of baseline/micro/macro for 3 example papers.

    Args:
        examples: list of dicts with paper_id, baseline, micro, macro texts
        output_path: path to save figure
    """
    setup_style()
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Baseline", "Micro-steered", "Macro-steered"]
    col_keys = ["baseline", "micro", "macro"]

    for row, ex in enumerate(examples):
        for col, (title, key) in enumerate(zip(col_titles, col_keys)):
            ax = axes[row][col]
            text = ex.get(key, "")
            # Truncate for display
            display_text = text[:400] + ("..." if len(text) > 400 else "")
            ax.text(
                0.05, 0.95, display_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                wrap=True,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
            )
            paper_id = ex.get("paper_id", "")
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Paper: {paper_id}", fontsize=8, rotation=0, labelpad=80)
            ax.axis("off")

    plt.suptitle("Example Outputs: Baseline vs Micro vs Macro", fontsize=13, y=1.01)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_summ_report(results: dict, config: dict, output_path: str):
    """Generate markdown report for summarization experiment.

    Args:
        results: evaluation_results dict from 06d
        config: config dict
        output_path: path to save .md report
    """
    verdict = results.get("verdict", "NOT CONFIRMED")
    today = date.today().isoformat()

    thresholds = config.get("thresholds", {})
    summ_cfg = config.get("summarization", {})

    h1 = results.get("h1_slod_shift", {}).get("micro", {})
    h1_macro = results.get("h1_slod_shift", {}).get("macro", {})
    h2 = results.get("h2_surface", {})
    h3 = results.get("h3_quality", {})

    layer_selection = results.get("layer_selection", {})
    selected_layer = layer_selection.get("selected_layer", "N/A")
    selected_alpha = results.get("selected_alpha", "N/A")
    direction_flip = layer_selection.get("direction_flip", False)

    n_papers = results.get("n_papers", "N/A")

    surface_metric_keys = ["entity_density", "citation_density", "numeric_density", "mean_sentence_length"]
    surface_labels = {
        "entity_density": "Entity Density",
        "citation_density": "Citation Density",
        "numeric_density": "Numeric Density",
        "mean_sentence_length": "Mean Sentence Length",
    }

    def fmt(val, spec):
        if isinstance(val, float):
            return format(val, spec)
        return str(val)

    lines = [
        f"# SH2-summ: Activation Steering on Summarization Task — Results",
        f"",
        f"**Date:** {today}",
        f"**Verdict:** **{verdict}**",
        f"**Papers evaluated:** {n_papers}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"This experiment tests whether activation steering along the SLoD axis produces",
        f"measurable abstraction-level shifts when applied to scientific paper summarization.",
        f"The summarization task was chosen to address the QA-task ceiling identified in",
        f"SH2a (d=0.121) — summaries span the full macro-to-micro range and are in-distribution",
        f"for the SciBERT SLoD evaluation axis.",
        f"",
        f"---",
        f"",
        f"## 1. SLoD Shift (H1)",
        f"",
        f"**Criterion:** paired t-test p < {thresholds.get('h1_shift_p', 0.05)}, Cohen's d > {thresholds.get('h1_shift_d', 0.5)}",
        f"",
        f"### Micro Steering Direction",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Δ SLoD | {fmt(h1.get('mean_delta'), '.4f')} |",
        f"| Std Δ SLoD | {fmt(h1.get('std_delta'), '.4f')} |",
        f"| p-value | {fmt(h1.get('p_value'), '.4e')} |",
        f"| Cohen's d | {fmt(h1.get('cohens_d'), '.4f')} |",
        f"| H1 passed | {'Yes' if results.get('h1_passed', False) else 'No'} |",
        f"",
        f"### Macro Steering Direction",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Δ SLoD | {fmt(h1_macro.get('mean_delta'), '.4f')} |",
        f"| Std Δ SLoD | {fmt(h1_macro.get('std_delta'), '.4f')} |",
        f"| p-value | {fmt(h1_macro.get('p_value'), '.4e')} |",
        f"| Cohen's d | {fmt(h1_macro.get('cohens_d'), '.4f')} |",
        f"",
        f"![SLoD shift distribution](figures/slod_shift_distribution_summ.png)",
        f"",
        f"---",
        f"",
        f"## 2. Surface Metrics (H2)",
        f"",
        f"**Criterion:** at least {thresholds.get('h2_min_metrics', 2)} of 4 surface metrics "
        f"shift significantly (p < {thresholds.get('h2_surface_p', 0.05)})",
        f"",
        f"| Metric | Baseline Mean | Steered Mean | Mean Δ | p-value | Significant |",
        f"|--------|--------------|--------------|--------|---------|-------------|",
    ]

    for key in surface_metric_keys:
        m = h2.get(key, {})
        if all(isinstance(m.get(k), float) for k in ["mean_baseline", "mean_steered", "mean_delta", "p_value"]):
            lines.append(
                f"| {surface_labels.get(key, key)} "
                f"| {m['mean_baseline']:.4f} "
                f"| {m['mean_steered']:.4f} "
                f"| {m['mean_delta']:.4f} "
                f"| {m['p_value']:.4e} "
                f"| {'Yes *' if m.get('significant', False) else 'No'} |"
            )
        else:
            lines.append(f"| {surface_labels.get(key, key)} | — | — | — | — | — |")

    n_sig = h2.get("n_significant", 0)
    h2_passed = results.get("h2_passed", False)
    lines += [
        f"",
        f"**Significant metrics: {n_sig}/4** — H2 {'PASSED' if h2_passed else 'FAILED'}",
        f"",
        f"![Surface metrics](figures/surface_metrics_summ.png)",
        f"",
        f"---",
        f"",
        f"## 3. Quality Preservation (H3 — ROUGE-L)",
        f"",
        f"**Criterion:** ROUGE-L drop < {summ_cfg.get('rouge_drop_threshold', 0.05):.2f} vs baseline",
        f"(both evaluated against paper's micro reference spans)",
        f"",
        f"| Condition | Mean ROUGE-L | Drop |",
        f"|-----------|-------------|------|",
        f"| Baseline vs micro_reference | {fmt(h3.get('baseline_rouge_l'), '.4f')} | — |",
        f"| Micro-steered vs micro_reference | {fmt(h3.get('micro_rouge_l'), '.4f')} | {fmt(h3.get('drop'), '.4f')} |",
        f"| H3 passed | {'Yes' if results.get('h3_passed', False) else 'No'} | — |",
        f"",
        f"![Quality preservation](figures/quality_preservation_summ.png)",
        f"",
        f"---",
        f"",
        f"## 4. Layer and Alpha Selection",
        f"",
        f"- **Selected layer:** {selected_layer}",
        f"- **Selected alpha:** {selected_alpha}",
        f"- **Direction flip:** {direction_flip}",
        f"",
        f"![Layer comparison](figures/layer_comparison_summ.png)",
        f"![Alpha sensitivity](figures/alpha_sensitivity_summ.png)",
        f"",
        f"---",
        f"",
        f"## 5. Comparison with QA Experiments",
        f"",
        f"| Experiment | Task | H1 d | H1 p | H2 | H3 | Verdict |",
        f"|-----------|------|------|------|-----|-----|---------|",
        f"| SH2 (doc-spans) | QA | 0.043 | 0.34 | 0/4 | Pass | NOT CONFIRMED |",
        f"| SH2b (QA-context) | QA | 0.546* | 4.2e-30 | 3/4 | Fail | PARTIAL |",
        f"| SH2c (flip+low-alpha) | QA | 0.190 | 2.5e-5 | 0/4 | Pass | NOT CONFIRMED |",
        f"| SH2a (prompt-only) | QA | 0.121 | 7.2e-3 | 4/4 | Fail | NOT CONFIRMED |",
        f"| SH2-summ (this) | Summarization | {fmt(h1.get('cohens_d'), '.3f')} | {fmt(h1.get('p_value'), '.2e')} | {n_sig}/4 | {'Pass' if results.get('h3_passed') else 'Fail'} | {verdict} |",
        f"",
        f"*SH2b H1 d was negative (inverted direction); |d|=0.546.",
        f"",
        f"---",
        f"",
        f"## 6. Example Outputs",
        f"",
        f"![Example outputs](figures/example_outputs_summ.png)",
        f"",
        f"---",
        f"",
        f"## 7. Interpretation",
        f"",
    ]

    if verdict == "CONFIRMED":
        lines += [
            f"The summarization experiment **CONFIRMED** the hypothesis. Activation steering",
            f"along the SLoD axis produced a statistically significant shift (H1 passed,",
            f"Cohen's d={fmt(h1.get('cohens_d'), '.3f')} > {thresholds.get('h1_shift_d', 0.5)}) while",
            f"preserving factual coverage (H3 passed, ROUGE-L drop={fmt(h3.get('drop'), '.4f')} < {summ_cfg.get('rouge_drop_threshold', 0.05)}).",
            f"",
            f"This resolves the QA-task ceiling (d_ceiling=0.121 from SH2a). The summarization",
            f"task is inherently in-distribution for the SciBERT SLoD evaluator and spans the",
            f"full macro-to-micro range, enabling effective activation steering.",
        ]
    elif verdict == "PARTIAL":
        lines += [
            f"The summarization experiment produced a **PARTIAL** result. H1 (SLoD shift)",
            f"was confirmed (d={fmt(h1.get('cohens_d'), '.3f')}), indicating that activation",
            f"steering does shift summaries along the SLoD axis when applied to summarization.",
            f"However, H3 (quality preservation) failed: ROUGE-L drop={fmt(h3.get('drop'), '.4f')} > {summ_cfg.get('rouge_drop_threshold', 0.05)}.",
            f"",
            f"Future work: lower alpha values or constrained decoding to preserve factual coverage",
            f"while maintaining the SLoD shift.",
        ]
    else:
        lines += [
            f"The summarization experiment was **NOT CONFIRMED**. H1 (SLoD shift) did not",
            f"reach the required threshold (d={fmt(h1.get('cohens_d'), '.3f')} < {thresholds.get('h1_shift_d', 0.5)} required,",
            f"p={fmt(h1.get('p_value'), '.4e')}).",
            f"",
            f"The summarization task was expected to address the QA-task ceiling, but the",
            f"cross-architecture mismatch between the generative model's residual stream and",
            f"the SciBERT SLoD axis persists even for in-distribution summarization outputs.",
            f"",
            f"Recommendation: fine-tune a task-specific SLoD classifier on generated summaries",
            f"(domain adaptation), or use SFT-based control with SLoD-supervised training signal.",
        ]

    lines += ["", f"---", f"", f"*Generated automatically by SH2-summ pipeline on {today}.*"]

    report_text = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 7c: Report for summarization experiment")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    summ_cfg = config["summarization"]

    output_path = Path(config["reports_dir"]) / "SH2_summ_results.md"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_dir"])
    summ_dir = data_dir / "summarization"

    # Load evaluation results
    eval_results_path = Path(summ_cfg["evaluation_results_path"])
    print("Loading evaluation results...")
    results = load_json(str(eval_results_path))

    # --- Figure 1: SLoD shift distribution ---
    print("\nGenerating Figure 1: SLoD shift distribution...")
    try:
        scores_path = summ_dir / "slod_scores.npz"
        scores_data = np.load(str(scores_path))
        deltas_micro = scores_data["deltas_micro"]
        deltas_macro = scores_data["deltas_macro"]

        plot_slod_shift_distribution(
            deltas_micro, deltas_macro,
            str(figures_dir / "slod_shift_distribution_summ.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 1: {e}")

    # --- Figure 2: Alpha sensitivity ---
    print("Generating Figure 2: Alpha sensitivity...")
    try:
        alpha_sweep_path = summ_dir / "alpha_sweep.json"
        alpha_sweep = load_json(str(alpha_sweep_path))
        alpha_values = alpha_sweep["alpha_values"]
        shifts = [float(alpha_sweep["results"][str(a)]["shift"]) for a in alpha_values]

        plot_alpha_sensitivity(
            alpha_values, shifts,
            str(figures_dir / "alpha_sensitivity_summ.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 2: {e}")

    # --- Figure 3: Layer comparison ---
    print("Generating Figure 3: Layer comparison...")
    try:
        layer_sel_path = data_dir / "results" / "layer_selection_summ.json"
        layer_selection = load_json(str(layer_sel_path))
        candidate_layers = layer_selection["candidate_layers"]
        layer_shifts_dict = layer_selection["layer_shifts"]
        layer_shifts = [float(layer_shifts_dict[str(l)]) for l in candidate_layers]
        selected_layer = int(layer_selection["selected_layer"])

        plot_layer_comparison(
            candidate_layers, layer_shifts, selected_layer,
            str(figures_dir / "layer_comparison_summ.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 3: {e}")

    # --- Figure 4: Quality preservation (ROUGE-L scatter) ---
    print("Generating Figure 4: Quality preservation...")
    try:
        h3 = results.get("h3_quality", {})
        baseline_scores_rouge = np.array(h3.get("baseline_scores_per_paper", []))
        micro_scores_rouge = np.array(h3.get("micro_scores_per_paper", []))

        if len(baseline_scores_rouge) > 0 and len(micro_scores_rouge) > 0:
            # Reuse plot_quality_preservation with ROUGE-L scores
            plot_quality_preservation(
                baseline_scores_rouge, micro_scores_rouge,
                str(figures_dir / "quality_preservation_summ.png")
            )
        else:
            print("  Warning: no per-paper ROUGE-L scores available.")
    except Exception as e:
        print(f"  Warning: could not generate Figure 4: {e}")

    # --- Figure 5: Surface metrics ---
    print("Generating Figure 5: Surface metrics...")
    try:
        h2 = results.get("h2_surface", {})
        plot_surface_metrics(
            h2,
            str(figures_dir / "surface_metrics_summ.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 5: {e}")

    # --- Figure 6: Example outputs ---
    print("Generating Figure 6: Example outputs...")
    try:
        summaries_path = summ_dir / "summaries.jsonl"
        all_summaries = load_jsonl(str(summaries_path))

        baseline_by_pid = {r["paper_id"]: r["summary"] for r in all_summaries if r["condition"] == "baseline"}
        micro_by_pid = {r["paper_id"]: r["summary"] for r in all_summaries if r["condition"] == "micro"}
        macro_by_pid = {r["paper_id"]: r["summary"] for r in all_summaries if r["condition"] == "macro"}

        common_ids = sorted(set(baseline_by_pid) & set(micro_by_pid) & set(macro_by_pid))

        # Pick 3 examples (first 3 for reproducibility)
        example_ids = common_ids[:3]
        examples = [
            {
                "paper_id": pid,
                "baseline": baseline_by_pid[pid],
                "micro": micro_by_pid[pid],
                "macro": macro_by_pid[pid],
            }
            for pid in example_ids
        ]

        plot_example_outputs(examples, str(figures_dir / "example_outputs_summ.png"))
    except Exception as e:
        print(f"  Warning: could not generate Figure 6: {e}")

    # --- Generate markdown report ---
    print("\nGenerating markdown report...")
    generate_summ_report(results, config, str(output_path))
    print(f"\nReport complete: {output_path}")


if __name__ == "__main__":
    main()
