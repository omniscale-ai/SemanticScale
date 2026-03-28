#!/usr/bin/env python3
"""Stage G: Generate visualization and auto-report.

Produces 6 figures and a markdown report with verdict.

Output: reports/SH2_results.md, reports/figures/*.png
"""
import sys
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_json, load_jsonl
from src.visualization import (
    setup_style,
    plot_slod_axis_validation,
    plot_slod_shift_distribution,
    plot_alpha_sensitivity,
    plot_layer_comparison,
    plot_quality_preservation,
    plot_surface_metrics,
    generate_report,
)


def main():
    parser = argparse.ArgumentParser(description="Stage G: Report")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["reports_dir"]) / "SH2_results.md"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_dir"])
    results_dir = data_dir / "results"

    # --- Load evaluation results ---
    print("Loading evaluation results...")
    results = load_json(str(results_dir / "evaluation_results.json"))

    # --- Figure 1: SLoD axis validation ---
    print("\nGenerating Figure 1: SLoD axis validation...")
    try:
        axis_data = np.load(str(data_dir / "eval_slod_axis.npz"), allow_pickle=True)
        slod_axis = axis_data["centroid_axis"]

        sh1_data = np.load(config["sh1_embeddings"])
        sh1_embeddings = sh1_data["embeddings"]
        sh1_labels = sh1_data["labels"]

        with open(config["sh1_splits"]) as f:
            splits = json.load(f)
        test_idx = np.array(splits["test"])
        test_embs = sh1_embeddings[test_idx]
        test_labels = sh1_labels[test_idx]

        projections_by_class = {
            "macro": (test_embs[test_labels == 0] @ slod_axis).tolist(),
            "meso": (test_embs[test_labels == 1] @ slod_axis).tolist(),
            "micro": (test_embs[test_labels == 2] @ slod_axis).tolist(),
        }

        plot_slod_axis_validation(
            projections_by_class,
            str(figures_dir / "slod_axis_validation.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 1: {e}")

    # --- Figure 2: SLoD shift distribution ---
    print("Generating Figure 2: SLoD shift distribution...")
    try:
        scores_data = np.load(str(results_dir / "slod_scores.npz"))
        deltas_micro = scores_data["deltas_micro"]
        deltas_macro = scores_data["deltas_macro"]

        plot_slod_shift_distribution(
            deltas_micro, deltas_macro,
            str(figures_dir / "slod_shift_distribution.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 2: {e}")

    # --- Figure 3: Alpha sensitivity ---
    print("Generating Figure 3: Alpha sensitivity...")
    try:
        alpha_sweep = load_json(str(results_dir / "alpha_sweep.json"))
        alphas = alpha_sweep["alpha_values"]
        shifts = [float(alpha_sweep["shifts"][str(a)]) for a in alphas]

        plot_alpha_sensitivity(
            alphas, shifts,
            str(figures_dir / "alpha_sensitivity.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 3: {e}")

    # --- Figure 4: Layer comparison ---
    print("Generating Figure 4: Layer comparison...")
    try:
        layer_selection = load_json(str(results_dir / "layer_selection.json"))
        candidate_layers = layer_selection["candidate_layers"]
        layer_shifts_dict = layer_selection["layer_shifts"]
        layer_shifts = [float(layer_shifts_dict[str(l)]) for l in candidate_layers]
        selected_layer = int(layer_selection["selected_layer"])

        plot_layer_comparison(
            candidate_layers, layer_shifts, selected_layer,
            str(figures_dir / "layer_comparison.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 4: {e}")

    # --- Figure 5: Quality preservation ---
    print("Generating Figure 5: Quality preservation...")
    try:
        h3 = results.get("h3_factuality", {})
        baseline_scores_f1 = np.array(h3.get("baseline_scores_per_question", []))
        micro_scores_f1 = np.array(h3.get("micro_scores_per_question", []))

        if len(baseline_scores_f1) > 0 and len(micro_scores_f1) > 0:
            plot_quality_preservation(
                baseline_scores_f1, micro_scores_f1,
                str(figures_dir / "quality_preservation.png")
            )
        else:
            print("  Warning: no per-question F1 scores available.")
    except Exception as e:
        print(f"  Warning: could not generate Figure 5: {e}")

    # --- Figure 6: Surface metrics ---
    print("Generating Figure 6: Surface metrics...")
    try:
        h2 = results.get("h2_surface", {})
        plot_surface_metrics(
            h2,
            str(figures_dir / "surface_metrics.png")
        )
    except Exception as e:
        print(f"  Warning: could not generate Figure 6: {e}")

    # --- Generate markdown report ---
    print("\nGenerating markdown report...")
    generate_report(results, config, str(output_path))
    print(f"\nReport complete: {output_path}")


if __name__ == "__main__":
    main()
