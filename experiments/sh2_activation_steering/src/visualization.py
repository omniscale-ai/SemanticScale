"""Visualization functions for SH2 report generation.

Generates 6 figures:
1. SLoD axis validation (histogram of SH1 test projections by class)
2. SLoD shift distribution (Δ_SLoD histogram for steered_micro and steered_macro)
3. Alpha sensitivity (SLoD shift vs α value)
4. Layer comparison (SLoD shift by layer)
5. Quality preservation (token-F1 scatter: baseline vs steered)
6. Surface metrics (bar chart of metric changes)

All figures saved to reports/figures/ as PNG at 150 DPI.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import date


def setup_style():
    """Set consistent plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.figsize"] = (8, 5)


def plot_slod_axis_validation(projections: dict, output_path: str):
    """Plot histogram of SLoD axis projections for SH1 test set by class.

    Args:
        projections: dict with keys 'macro', 'meso', 'micro', each -> array of projections
        output_path: path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"macro": "steelblue", "meso": "orange", "micro": "green"}
    for cls in ["macro", "meso", "micro"]:
        if cls in projections and len(projections[cls]) > 0:
            data = np.array(projections[cls])
            ax.hist(data, bins=40, alpha=0.5, label=cls, color=colors[cls], density=True)

    ax.set_xlabel("SLoD axis projection")
    ax.set_ylabel("Density")
    ax.set_title("SLoD Axis Validation (SH1 test set)")
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_slod_shift_distribution(deltas_micro: np.ndarray, deltas_macro: np.ndarray,
                                  output_path: str):
    """Plot histogram of SLoD shift (Δ_SLoD) for both steering directions.

    Args:
        deltas_micro: (N,) Δ_SLoD for steered_micro
        deltas_macro: (N,) Δ_SLoD for steered_macro
        output_path: path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    mean_micro = float(np.mean(deltas_micro))
    mean_macro = float(np.mean(deltas_macro))

    ax.hist(deltas_micro, bins=30, alpha=0.6, color="green",
            label=f"Micro direction (mean={mean_micro:.3f})", density=True)
    ax.hist(deltas_macro, bins=30, alpha=0.6, color="steelblue",
            label=f"Macro direction (mean={mean_macro:.3f})", density=True)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="No shift")

    ax.set_xlabel("Δ SLoD score (steered − baseline)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of SLoD Shifts by Steering Direction")
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_alpha_sensitivity(alphas: list, shifts: list, output_path: str):
    """Plot SLoD shift vs alpha value (line plot).

    Args:
        alphas: list of alpha values tested
        shifts: list of mean SLoD shifts per alpha
        output_path: path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(alphas, shifts, marker="o", linewidth=2, markersize=8, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("α (steering strength)")
    ax.set_ylabel("Mean SLoD shift")
    ax.set_title("Alpha Sensitivity: SLoD Shift vs Steering Strength")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_layer_comparison(layers: list, shifts: list, selected_layer: int,
                          output_path: str):
    """Plot SLoD shift by layer (bar chart with selected layer highlighted).

    Args:
        layers: list of layer indices
        shifts: list of mean SLoD shifts per layer
        selected_layer: index of selected best layer
        output_path: path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["orange" if l == selected_layer else "steelblue" for l in layers]
    bars = ax.bar([str(l) for l in layers], shifts, color=colors)

    # Add value labels
    for bar, shift in zip(bars, shifts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{shift:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean SLoD shift")
    ax.set_title(f"Layer Comparison (selected: layer {selected_layer})")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="orange", label=f"Selected (layer {selected_layer})"),
        Patch(facecolor="steelblue", label="Other candidates"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_quality_preservation(baseline_f1: np.ndarray, steered_f1: np.ndarray,
                               output_path: str):
    """Scatter plot of token-F1: baseline vs steered with identity line.

    Args:
        baseline_f1: (N,) baseline token-F1 scores
        steered_f1: (N,) steered token-F1 scores
        output_path: path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(baseline_f1, steered_f1, alpha=0.4, s=20, color="steelblue", label="Questions")

    # Identity line
    min_val = min(float(baseline_f1.min()), float(steered_f1.min()))
    max_val = max(float(baseline_f1.max()), float(steered_f1.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="y = x")

    mean_diff = float(np.mean(steered_f1 - baseline_f1))
    ax.set_xlabel("Baseline token-F1")
    ax.set_ylabel("Steered token-F1")
    ax.set_title(f"Quality Preservation (mean Δ = {mean_diff:+.3f})")
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_surface_metrics(metrics_comparison: dict, output_path: str):
    """Bar chart of surface metric changes (steered vs baseline).

    Args:
        metrics_comparison: dict with per-metric comparison data
        output_path: path to save figure
    """
    setup_style()

    metric_keys = ["entity_density", "citation_density", "numeric_density", "mean_sentence_length"]
    metric_labels = ["Entity\nDensity", "Citation\nDensity", "Numeric\nDensity", "Mean Sentence\nLength"]

    mean_deltas = []
    significants = []
    for key in metric_keys:
        if key in metrics_comparison:
            mean_deltas.append(metrics_comparison[key].get("mean_delta", 0.0))
            significants.append(metrics_comparison[key].get("significant", False))
        else:
            mean_deltas.append(0.0)
            significants.append(False)

    colors = ["orange" if sig else "steelblue" for sig in significants]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metric_labels, mean_deltas, color=colors)

    # Asterisk for significant
    for bar, sig, delta in zip(bars, significants, mean_deltas):
        y_pos = bar.get_height() + (0.0005 if delta >= 0 else -0.002)
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    "*", ha="center", va="bottom", fontsize=14, color="red")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean change (steered − baseline)")
    ax.set_title("Surface Metric Changes (micro steering direction)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="orange", label="Significant (p < 0.05)"),
        Patch(facecolor="steelblue", label="Not significant"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_report(results: dict, config: dict, output_path: str):
    """Generate auto-report markdown with verdict and embedded figures.

    Args:
        results: evaluation_results dict
        config: config dict
        output_path: path to save report
    """
    verdict = results.get("verdict", "NOT CONFIRMED")
    today = date.today().isoformat()

    h1 = results.get("h1_micro", {})
    h1_macro = results.get("h1_macro", {})
    h2 = results.get("h2_surface", {})
    h3 = results.get("h3_factuality", {})

    baselines = config.get("baselines", {})
    thresholds = config.get("thresholds", {})

    # Verdict emoji/symbol
    verdict_symbols = {
        "CONFIRMED": "CONFIRMED",
        "PARTIAL": "PARTIAL",
        "NOT CONFIRMED": "NOT CONFIRMED",
    }
    verdict_str = verdict_symbols.get(verdict, verdict)

    lines = [
        f"# SH2: Activation Steering Along the SLoD Axis — Results",
        f"",
        f"**Date:** {today}",
        f"**Verdict:** **{verdict_str}**",
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
        f"| Mean Δ SLoD | {h1.get('mean_delta', 'N/A'):.4f} |" if isinstance(h1.get('mean_delta'), float) else f"| Mean Δ SLoD | {h1.get('mean_delta', 'N/A')} |",
        f"| Std Δ SLoD | {h1.get('std_delta', 'N/A'):.4f} |" if isinstance(h1.get('std_delta'), float) else f"| Std Δ SLoD | {h1.get('std_delta', 'N/A')} |",
        f"| p-value | {h1.get('p_value', 'N/A'):.4e} |" if isinstance(h1.get('p_value'), float) else f"| p-value | {h1.get('p_value', 'N/A')} |",
        f"| Cohen's d | {h1.get('cohens_d', 'N/A'):.4f} |" if isinstance(h1.get('cohens_d'), float) else f"| Cohen's d | {h1.get('cohens_d', 'N/A')} |",
        f"| H1 passed | {'Yes' if results.get('h1_passed', False) else 'No'} |",
        f"",
        f"### Macro Steering Direction",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    # Handle h1_macro values robustly
    for metric_key, metric_label, fmt in [
        ("mean_delta", "Mean Δ SLoD", ".4f"),
        ("std_delta", "Std Δ SLoD", ".4f"),
        ("p_value", "p-value", ".4e"),
        ("cohens_d", "Cohen's d", ".4f"),
    ]:
        val = h1_macro.get(metric_key, "N/A")
        if isinstance(val, float):
            lines.append(f"| {metric_label} | {val:{fmt}} |")
        else:
            lines.append(f"| {metric_label} | {val} |")

    lines += [
        f"",
        f"![SLoD shift distribution](figures/slod_shift_distribution.png)",
        f"",
        f"---",
        f"",
        f"## 2. Surface Metrics (H2)",
        f"",
        f"**Criterion:** at least {thresholds.get('h2_min_metrics', 2)} of 4 surface metrics shift significantly (p < {thresholds.get('h2_surface_p', 0.05)})",
        f"",
    ]

    # Surface metrics table
    surface_metric_keys = ["entity_density", "citation_density", "numeric_density", "mean_sentence_length"]
    surface_labels = {
        "entity_density": "Entity Density",
        "citation_density": "Citation Density",
        "numeric_density": "Numeric Density",
        "mean_sentence_length": "Mean Sentence Length",
    }
    lines += [
        f"| Metric | Baseline Mean | Steered Mean | Mean Δ | p-value | Significant |",
        f"|--------|--------------|--------------|--------|---------|-------------|",
    ]
    for key in surface_metric_keys:
        m = h2.get(key, {})
        lines.append(
            f"| {surface_labels.get(key, key)} "
            f"| {m.get('mean_baseline', 'N/A'):.4f} "
            f"| {m.get('mean_steered', 'N/A'):.4f} "
            f"| {m.get('mean_delta', 'N/A'):.4f} "
            f"| {m.get('p_value', 'N/A'):.4e} "
            f"| {'Yes *' if m.get('significant', False) else 'No'} |"
            if all(isinstance(m.get(k), float) for k in ['mean_baseline', 'mean_steered', 'mean_delta', 'p_value'])
            else f"| {surface_labels.get(key, key)} | — | — | — | — | — |"
        )

    n_sig = h2.get("n_significant", 0)
    h2_passed = results.get("h2_passed", False)
    lines += [
        f"",
        f"**Significant metrics: {n_sig}/4** — H2 {'PASSED' if h2_passed else 'FAILED'}",
        f"",
        f"![Surface metrics](figures/surface_metrics.png)",
        f"",
        f"---",
        f"",
        f"## 3. Factuality Preservation (H3)",
        f"",
        f"**Criterion:** token-F1 drop ≤ {thresholds.get('h3_max_quality_drop', 0.05):.2f} vs baseline",
        f"",
        f"| Condition | Mean Token-F1 |",
        f"|-----------|--------------|",
    ]

    baseline_f1 = h3.get("baseline_mean_f1", "N/A")
    steered_f1 = h3.get("steered_micro_mean_f1", "N/A")
    drop = (steered_f1 - baseline_f1) if isinstance(baseline_f1, float) and isinstance(steered_f1, float) else "N/A"

    lines += [
        f"| Baseline | {baseline_f1:.4f} |" if isinstance(baseline_f1, float) else f"| Baseline | {baseline_f1} |",
        f"| Steered (micro) | {steered_f1:.4f} |" if isinstance(steered_f1, float) else f"| Steered (micro) | {steered_f1} |",
        f"| Drop | {drop:.4f} |" if isinstance(drop, float) else f"| Drop | {drop} |",
        f"| H3 passed | {'Yes' if results.get('h3_passed', False) else 'No'} |",
        f"",
        f"![Quality preservation](figures/quality_preservation.png)",
        f"",
        f"---",
        f"",
        f"## 4. Layer and Alpha Selection",
        f"",
    ]

    layer_selection = results.get("layer_selection", {})
    selected_layer = layer_selection.get("selected_layer", "N/A")
    selected_alpha = results.get("selected_alpha", "N/A")
    lines += [
        f"- **Selected layer:** {selected_layer}",
        f"- **Selected alpha:** {selected_alpha}",
        f"",
        f"![Layer comparison](figures/layer_comparison.png)",
        f"![Alpha sensitivity](figures/alpha_sensitivity.png)",
        f"",
        f"---",
        f"",
        f"## 5. SLoD Axis Validation",
        f"",
        f"![SLoD axis validation](figures/slod_axis_validation.png)",
        f"",
        f"---",
        f"",
        f"## 6. Comparison with SH Family Baselines",
        f"",
        f"| Metric | SH5d Baseline | SH2 Result |",
        f"|--------|--------------|------------|",
        f"| SciBERT SLoD Cohen's d | {baselines.get('sh5d_cohens_d', 'N/A')} | {h1.get('cohens_d', 'N/A'):.4f} |" if isinstance(h1.get('cohens_d'), float) else f"| SciBERT SLoD Cohen's d | {baselines.get('sh5d_cohens_d', 'N/A')} | {h1.get('cohens_d', 'N/A')} |",
        f"| SH1 probe macro-F1 | {baselines.get('sh1_probe_macro_f1', 'N/A')} | — |",
        f"| SH5d SLoD axis ρ | {baselines.get('sh5d_slod_axis_mean_rho', 'N/A')} | — |",
        f"",
        f"---",
        f"",
        f"## 7. Interpretation",
        f"",
    ]

    if verdict == "CONFIRMED":
        lines += [
            f"The steering experiment **CONFIRMED** the hypothesis. Activation steering along the SLoD axis",
            f"produced a statistically significant shift in generated text toward the target abstraction level",
            f"(H1 passed, Cohen's d > {thresholds.get('h1_shift_d', 0.5)}), while factuality was preserved",
            f"(H3 passed, token-F1 drop ≤ {thresholds.get('h3_max_quality_drop', 0.05)}).",
            f"",
            f"This demonstrates that the SLoD direction is not only linearly separable in SciBERT space (SH1, SH5d),",
            f"but is also present in the generative model's representation space and can be used for controllable generation.",
        ]
    elif verdict == "PARTIAL":
        lines += [
            f"The steering experiment produced a **PARTIAL** result. H1 (SLoD shift) was confirmed — steering",
            f"successfully moved generated text along the SLoD axis. However, H3 (factuality preservation) failed,",
            f"indicating that the steering strength used caused quality degradation.",
            f"",
            f"Future work should explore lower alpha values or token-level constraints to maintain factuality.",
        ]
    else:
        lines += [
            f"The steering experiment was **NOT CONFIRMED**. The SLoD shift (H1) did not reach the required",
            f"significance threshold (p < {thresholds.get('h1_shift_p', 0.05)}, Cohen's d > {thresholds.get('h1_shift_d', 0.5)}).",
            f"",
            f"Possible explanations include insufficient α values, suboptimal layer selection, or the SLoD axis",
            f"being less distinct in the generative model's hidden states compared to SciBERT's encoder space.",
        ]

    lines += ["", f"---", f"", f"*Generated automatically by SH2 pipeline on {today}.*"]

    report_text = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {output_path}")
