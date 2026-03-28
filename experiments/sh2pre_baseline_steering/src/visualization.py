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
    # TODO: Engineer implements this
    raise NotImplementedError("Engineer agent should implement this")


def plot_slod_shift_distribution(deltas_micro: np.ndarray, deltas_macro: np.ndarray,
                                  output_path: str):
    """Plot histogram of SLoD shift (Δ_SLoD) for both steering directions.

    Args:
        deltas_micro: (N,) Δ_SLoD for steered_micro
        deltas_macro: (N,) Δ_SLoD for steered_macro
        output_path: path to save figure
    """
    # TODO: Engineer implements this
    raise NotImplementedError("Engineer agent should implement this")


def plot_alpha_sensitivity(alphas: list, shifts: list, output_path: str):
    """Plot SLoD shift vs alpha value (line plot).

    Args:
        alphas: list of alpha values tested
        shifts: list of mean SLoD shifts per alpha
        output_path: path to save figure
    """
    # TODO: Engineer implements this
    raise NotImplementedError("Engineer agent should implement this")


def plot_layer_comparison(layers: list, shifts: list, selected_layer: int,
                          output_path: str):
    """Plot SLoD shift by layer (bar chart with selected layer highlighted).

    Args:
        layers: list of layer indices
        shifts: list of mean SLoD shifts per layer
        selected_layer: index of selected best layer
        output_path: path to save figure
    """
    # TODO: Engineer implements this
    raise NotImplementedError("Engineer agent should implement this")


def plot_quality_preservation(baseline_f1: np.ndarray, steered_f1: np.ndarray,
                               output_path: str):
    """Scatter plot of token-F1: baseline vs steered with identity line.

    Args:
        baseline_f1: (N,) baseline token-F1 scores
        steered_f1: (N,) steered token-F1 scores
        output_path: path to save figure
    """
    # TODO: Engineer implements this
    raise NotImplementedError("Engineer agent should implement this")


def plot_surface_metrics(metrics_comparison: dict, output_path: str):
    """Bar chart of surface metric changes (steered vs baseline).

    Args:
        metrics_comparison: dict with per-metric comparison data
        output_path: path to save figure
    """
    # TODO: Engineer implements this
    raise NotImplementedError("Engineer agent should implement this")


def generate_report(results: dict, config: dict, output_path: str):
    """Generate auto-report markdown with verdict and embedded figures.

    Args:
        results: evaluation_results dict
        config: config dict
        output_path: path to save report
    """
    # TODO: Engineer implements this
    # Should generate a markdown file with:
    # - Verdict (CONFIRMED / PARTIAL / NOT CONFIRMED)
    # - H1, H2, H3 results in tables
    # - Embedded figure links
    # - Comparison with SH5 family baselines
    raise NotImplementedError("Engineer agent should implement this")
