"""Stage F: Analysis, visualization, and report generation."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from src.utils import get_project_root, load_config, read_json, write_json


def plot_roc_curves(
    predictions: dict,
    ablation_names: list[str],
    output_path: Path,
) -> None:
    """Plot ROC curves for ablation groups."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"drift_only": "#e74c3c", "surface_only": "#3498db", "combined": "#2ecc71",
              "gbt_combined": "#9b59b6"}
    labels = {"drift_only": "Drift-only", "surface_only": "Surface-only",
              "combined": "Combined (LogReg)", "gbt_combined": "Combined (GBT)"}

    for name in ablation_names:
        if name not in predictions:
            continue
        pred = predictions[name]
        y_true = np.array(pred["y_true"])
        y_proba = np.array(pred["y_proba"])

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_proba)

        color = colors.get(name, "#333333")
        label = f"{labels.get(name, name)} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: Extraction Correctness Prediction", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"ROC curves saved to {output_path}")


def plot_precision_at_k_curves(
    predictions: dict,
    ablation_names: list[str],
    output_path: Path,
) -> None:
    """Plot precision@k curves (correctness vs retention rate)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"drift_only": "#e74c3c", "surface_only": "#3498db", "combined": "#2ecc71",
              "gbt_combined": "#9b59b6"}
    labels = {"drift_only": "Drift-only", "surface_only": "Surface-only",
              "combined": "Combined (LogReg)", "gbt_combined": "Combined (GBT)"}

    k_values = np.linspace(0.05, 1.0, 20)

    for name in ablation_names:
        if name not in predictions:
            continue
        pred = predictions[name]
        y_true = np.array(pred["y_true"])
        y_proba = np.array(pred["y_proba"])

        sorted_idx = np.argsort(-y_proba)
        precisions = []
        for k in k_values:
            n_retain = max(1, int(len(y_true) * k))
            top_idx = sorted_idx[:n_retain]
            prec = np.mean(y_true[top_idx])
            precisions.append(prec)

        color = colors.get(name, "#333333")
        ax.plot(k_values * 100, precisions, color=color, linewidth=2,
                label=labels.get(name, name), marker="o", markersize=3)

    # Baseline: overall correctness rate
    if predictions:
        first_pred = next(iter(predictions.values()))
        baseline = np.mean(first_pred["y_true"])
        ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7,
                    label=f"Baseline ({baseline:.1%})")

    ax.set_xlabel("Retention Rate (%)", fontsize=12)
    ax.set_ylabel("Precision (Correctness)", fontsize=12)
    ax.set_title("Precision@k: Correctness vs Retention Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Precision@k curves saved to {output_path}")


def plot_feature_importance(
    metrics: dict,
    output_path: Path,
) -> None:
    """Plot feature importance (LogReg coefficients) for the combined model."""
    combined = metrics.get("combined", {})
    coefficients = combined.get("coefficients", [])
    feature_names = combined.get("feature_names", [])

    if not coefficients or not feature_names:
        logging.warning("No coefficients available for feature importance plot")
        return

    coefficients = np.array(coefficients)
    # Sort by absolute value
    sort_idx = np.argsort(np.abs(coefficients))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [feature_names[i] for i in sort_idx]
    coefs = coefficients[sort_idx]

    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs]
    ax.barh(range(len(names)), coefs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Coefficient (standardized)", fontsize=12)
    ax.set_title("Feature Importance: Combined LogReg Model", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature importance saved to {output_path}")


def plot_drift_distribution(
    predictions: dict,
    feature_matrix_path: Path,
    output_path: Path,
) -> None:
    """Plot drift distribution for correct vs incorrect extractions."""
    data = np.load(feature_matrix_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])

    # Find drift_mean feature index
    if "drift_mean" not in feature_names:
        logging.warning("drift_mean not in features, skipping distribution plot")
        return

    drift_idx = feature_names.index("drift_mean")
    drift_values = X[:, drift_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    correct_drift = drift_values[y == 1]
    incorrect_drift = drift_values[y == 0]

    ax.hist(correct_drift, bins=20, alpha=0.6, color="#2ecc71", label=f"Correct (n={len(correct_drift)})")
    ax.hist(incorrect_drift, bins=20, alpha=0.6, color="#e74c3c", label=f"Incorrect (n={len(incorrect_drift)})")

    ax.set_xlabel("Mean SLoD Drift", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SLoD Drift Distribution: Correct vs Incorrect Extractions", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Drift distribution saved to {output_path}")


def plot_confusion_matrix(
    predictions: dict,
    model_name: str,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Plot confusion matrix for the specified model."""
    if model_name not in predictions:
        return

    pred = predictions[model_name]
    y_true = np.array(pred["y_true"])
    y_proba = np.array(pred["y_proba"])
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Incorrect", "Correct"],
                yticklabels=["Incorrect", "Correct"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")


def generate_report(
    metrics: dict,
    predictions: dict,
    config: dict,
    figures_dir: Path,
    output_path: Path,
) -> None:
    """Generate markdown report summarizing SH4 results."""
    thresholds = config.get("thresholds", {})

    # Gather key metrics
    splits = metrics.get("splits", {})
    drift = metrics.get("drift_only", {})
    surface = metrics.get("surface_only", {})
    combined = metrics.get("combined", {})
    gbt = metrics.get("gbt_combined", {})

    drift_auroc = drift.get("auroc", 0)
    surface_auroc = surface.get("auroc", 0)
    combined_auroc = combined.get("auroc", 0)
    drift_advantage = drift_auroc - surface_auroc

    # Determine verdict
    auroc_confirmed = thresholds.get("auroc_confirmed", 0.65)
    auroc_partial = thresholds.get("auroc_partial", 0.60)
    drift_adv_threshold = thresholds.get("drift_advantage", 0.03)
    prec_50_threshold = thresholds.get("precision_at_50", 0.70)

    prec_50 = combined.get("precision_at_k", {}).get("k=0.50", 0)

    if combined_auroc >= auroc_confirmed and drift_advantage >= drift_adv_threshold and prec_50 >= prec_50_threshold:
        verdict = "CONFIRMED"
        verdict_detail = "All exit criteria met."
    elif combined_auroc >= auroc_partial:
        verdict = "PARTIAL"
        verdict_detail = "Combined AUROC meets partial threshold."
    else:
        verdict = "NOT CONFIRMED"
        verdict_detail = "Combined AUROC below partial threshold."

    # Build report
    lines = [
        "# SH4 Results: Abstraction Drift Predicts TKH Extraction Quality",
        "",
        f"## Verdict: **{verdict}**",
        "",
        verdict_detail,
        "",
        "---",
        "",
        "## Dataset Statistics",
        "",
        f"| Split | Samples | Correct Rate |",
        f"|-------|---------|-------------|",
        f"| Train | {splits.get('train_size', 'N/A')} | {splits.get('train_correct_rate', 0):.1%} |",
        f"| Val   | {splits.get('val_size', 'N/A')} | {splits.get('val_correct_rate', 0):.1%} |",
        f"| Test  | {splits.get('test_size', 'N/A')} | {splits.get('test_correct_rate', 0):.1%} |",
        "",
        "---",
        "",
        "## Ablation Results",
        "",
        "| Model | AUROC | Brier Score | Precision@50% | Best C |",
        "|-------|-------|-------------|---------------|--------|",
    ]

    for name, label in [("drift_only", "Drift-only"), ("surface_only", "Surface-only"),
                         ("combined", "Combined (LogReg)"), ("gbt_combined", "Combined (GBT)")]:
        m = metrics.get(name, {})
        if not m:
            continue
        auroc = m.get("auroc", "N/A")
        brier = m.get("brier_score", "N/A")
        p50 = m.get("precision_at_k", {}).get("k=0.50", "N/A")
        bc = m.get("best_C", "N/A")
        auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else str(auroc)
        brier_str = f"{brier:.4f}" if isinstance(brier, float) else str(brier)
        p50_str = f"{p50:.4f}" if isinstance(p50, float) else str(p50)
        lines.append(f"| {label} | {auroc_str} | {brier_str} | {p50_str} | {bc} |")

    lines.extend([
        "",
        f"**Drift advantage:** {drift_advantage:+.4f} (threshold: {drift_adv_threshold})",
        "",
        "---",
        "",
        "## Exit Criteria Evaluation",
        "",
        f"| Criterion | Threshold | Actual | Met? |",
        f"|-----------|-----------|--------|------|",
        f"| Combined AUROC | >= {auroc_confirmed} | {combined_auroc:.4f} | {'Yes' if combined_auroc >= auroc_confirmed else 'No'} |",
        f"| Drift advantage | >= {drift_adv_threshold} | {drift_advantage:+.4f} | {'Yes' if drift_advantage >= drift_adv_threshold else 'No'} |",
        f"| Precision@50% | >= {prec_50_threshold} | {prec_50:.4f} | {'Yes' if prec_50 >= prec_50_threshold else 'No'} |",
        "",
        "---",
        "",
        "## Figures",
        "",
        "### ROC Curves",
        f"![[figures/roc_curves.png]]",
        "",
        "### Precision@k Curves",
        f"![[figures/precision_at_k.png]]",
        "",
        "### Feature Importance",
        f"![[figures/feature_importance.png]]",
        "",
        "### Drift Distribution",
        f"![[figures/drift_distribution.png]]",
        "",
        "### Confusion Matrix (Combined)",
        f"![[figures/confusion_matrix.png]]",
        "",
        "---",
        "",
        "## Feature Importance (Top 10)",
        "",
    ])

    # Feature importance table
    coefficients = combined.get("coefficients", [])
    feat_names = combined.get("feature_names", [])
    if coefficients and feat_names:
        coef_arr = np.array(coefficients)
        sort_idx = np.argsort(np.abs(coef_arr))[::-1]

        lines.extend([
            "| Rank | Feature | Coefficient |",
            "|------|---------|-------------|",
        ])
        for rank, idx in enumerate(sort_idx[:10], 1):
            lines.append(f"| {rank} | {feat_names[idx]} | {coef_arr[idx]:+.4f} |")

    lines.extend([
        "",
        "---",
        "",
        f"*Generated by SH4 analysis pipeline*",
    ])

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    logging.info(f"Report saved to {output_path}")


def run_analysis(config: dict = None, project_root: Path = None) -> Path:
    """Run the full analysis and reporting pipeline.

    Returns path to the report.
    """
    if config is None:
        config = load_config()
    if project_root is None:
        project_root = get_project_root()

    data_dir = project_root / config["data_dir"]
    analysis_cfg = config.get("analysis", {})
    figures_dir = project_root / analysis_cfg.get("figures_dir", "reports/figures")
    report_path = project_root / analysis_cfg.get("report", "reports/sh4_results.md")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics_path = data_dir / config["model"]["output"]["metrics"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}. Run Stage E first.")
    metrics = read_json(metrics_path)

    # Load predictions
    predictions_path = data_dir / "results" / "predictions.json"
    predictions = {}
    if predictions_path.exists():
        predictions = read_json(predictions_path)

    # Feature matrix path for distribution plot
    feat_path = data_dir / config["features"]["output"]

    ablation_names = ["drift_only", "surface_only", "combined", "gbt_combined"]

    # Generate all plots
    if predictions:
        plot_roc_curves(predictions, ablation_names, figures_dir / "roc_curves.png")
        plot_precision_at_k_curves(predictions, ablation_names, figures_dir / "precision_at_k.png")
        plot_confusion_matrix(predictions, "combined", figures_dir / "confusion_matrix.png")

    plot_feature_importance(metrics, figures_dir / "feature_importance.png")

    if feat_path.exists():
        plot_drift_distribution(predictions, feat_path, figures_dir / "drift_distribution.png")

    # Generate report
    generate_report(metrics, predictions, config, figures_dir, report_path)

    logging.info(f"=== Analysis complete. Report at {report_path} ===")
    return report_path
