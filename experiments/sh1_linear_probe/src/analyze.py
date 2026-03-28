"""Analysis and visualization for SH1: confusion matrices, t-SNE, PCA, report."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils import LABEL_NAMES, load_config, load_splits


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved to {save_path}")


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | Path,
    max_points: int = 3000,
    seed: int = 42,
) -> None:
    """Plot t-SNE 2D visualization of embeddings, colored by SLoD label."""
    n = len(embeddings)
    if n > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_points, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        logging.info(f"Sampled {max_points} points for t-SNE (from {n})")

    logging.info("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # red, orange, green for macro, meso, micro
    for i, name in enumerate(LABEL_NAMES):
        mask = labels == i
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[i],
            label=name,
            alpha=0.4,
            s=10,
        )
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"t-SNE plot saved to {save_path}")


def plot_pca_variance(
    embeddings: np.ndarray,
    title: str,
    save_path: str | Path,
    n_components: int = 50,
) -> None:
    """Plot PCA explained variance curve."""
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    fig, ax = plt.subplots(figsize=(8, 5))
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, n_components + 1), cumvar, "b-o", markersize=3)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(title)
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="95% variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"PCA variance plot saved to {save_path}")


def generate_report(
    all_results: dict,
    config: dict,
    save_path: str | Path,
) -> None:
    """Generate a Markdown report with tables summarizing all results."""
    lines = []
    lines.append("# SH1 Results: Linear Decodability of SLoD from Frozen Embeddings\n")
    lines.append(f"**Generated:** auto\n")
    lines.append("---\n")

    # Thresholds
    t = config.get("thresholds", {})
    lines.append("## Exit Criteria\n")
    lines.append(f"- Primary (3-way macro-F1): > {t.get('three_way_f1', 0.60)}")
    lines.append(f"- Binary fallback (macro-F1): > {t.get('binary_f1', 0.75)}")
    lines.append(f"- Baseline gap: > {t.get('baseline_gap', 0.15)}\n")

    # Main results table
    lines.append("## Probe Results\n")
    lines.append("| Model | Classifier | C | Val F1 | Test F1 | Macro P | Macro R |")
    lines.append("|-------|-----------|-----|--------|---------|---------|---------|")

    best_test_f1 = 0.0
    best_config_str = ""

    for model_key in config.get("models", {}):
        if model_key not in all_results:
            continue
        model_results = all_results[model_key]
        for clf_key, metrics in model_results.items():
            c_val = metrics.get("C", "?")
            val_f1 = metrics.get("val", {}).get("macro_f1", 0)
            test_f1 = metrics.get("test", {}).get("macro_f1", 0)
            test_p = metrics.get("test", {}).get("macro_precision", 0)
            test_r = metrics.get("test", {}).get("macro_recall", 0)

            lines.append(
                f"| {model_key} | {clf_key} | {c_val} | "
                f"{val_f1:.4f} | {test_f1:.4f} | {test_p:.4f} | {test_r:.4f} |"
            )

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_config_str = f"{model_key} + {clf_key} (C={c_val})"

    lines.append("")

    # Baselines
    if "baselines" in all_results:
        lines.append("## Baselines\n")
        lines.append("| Baseline | Macro F1 | Accuracy |")
        lines.append("|----------|----------|----------|")
        for bl_name, bl_metrics in all_results["baselines"].items():
            lines.append(
                f"| {bl_name} | {bl_metrics.get('macro_f1', 0):.4f} | "
                f"{bl_metrics.get('accuracy', 0):.4f} |"
            )
        lines.append("")

    # Binary fallback
    if "binary_fallback" in all_results:
        bf = all_results["binary_fallback"]
        lines.append("## Binary Fallback (macro vs micro)\n")
        lines.append(f"- Macro-F1: {bf.get('macro_f1', 0):.4f}")
        lines.append(f"- Accuracy: {bf.get('accuracy', 0):.4f}")
        lines.append(f"- N train: {bf.get('n_train', '?')}, N test: {bf.get('n_test', '?')}")
        lines.append("")

    # Confound check
    if "confound_check" in all_results:
        cc = all_results["confound_check"]
        lines.append("## Confound Check (full dataset)\n")
        for k, v in cc.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: macro-F1 = {v.get('macro_f1', 0):.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    # Per-class analysis for best model
    lines.append("## Per-Class Analysis (best model)\n")
    lines.append(f"Best configuration: {best_config_str}\n")

    # Find best model's per-class results
    for model_key in config.get("models", {}):
        if model_key not in all_results:
            continue
        for clf_key, metrics in all_results[model_key].items():
            test_f1 = metrics.get("test", {}).get("macro_f1", 0)
            if abs(test_f1 - best_test_f1) < 1e-6:
                per_class = metrics.get("test", {}).get("per_class_f1", {})
                if per_class:
                    lines.append("| Class | F1 |")
                    lines.append("|-------|-----|")
                    for name, f1_val in per_class.items():
                        lines.append(f"| {name} | {f1_val:.4f} |")
                    lines.append("")
                break
        else:
            continue
        break

    # Verdict
    lines.append("## Verdict\n")
    baseline_f1 = 0.333
    if "baselines" in all_results and "random" in all_results["baselines"]:
        baseline_f1 = all_results["baselines"]["random"]["macro_f1"]
    gap = best_test_f1 - baseline_f1

    if best_test_f1 >= t.get("three_way_f1", 0.60):
        verdict = "SH1 CONFIRMED"
        lines.append(f"**{verdict}**: Best 3-way test macro-F1 = {best_test_f1:.4f} >= {t.get('three_way_f1', 0.60)}")
    elif "binary_fallback" in all_results and all_results["binary_fallback"].get("macro_f1", 0) >= t.get("binary_f1", 0.75):
        verdict = "SH1 PARTIAL (binary confirmed)"
        lines.append(f"**{verdict}**: Binary fallback macro-F1 = {all_results['binary_fallback']['macro_f1']:.4f} >= {t.get('binary_f1', 0.75)}")
    else:
        verdict = "SH1 NOT CONFIRMED"
        lines.append(f"**{verdict}**: Best 3-way test macro-F1 = {best_test_f1:.4f} < {t.get('three_way_f1', 0.60)}")

    lines.append(f"\nBaseline gap: {gap:.4f} (threshold: {t.get('baseline_gap', 0.15)})")
    lines.append("")

    # Limitations
    lines.append("## Known Limitations\n")
    lines.append("- QASPER is NLP-papers only; cross-domain transfer deferred to SH2/SH3")
    lines.append("- CPU-only embeddings (no GPU fine-tuning)")
    lines.append("- Linear probes only (non-linear classifiers deferred)")
    lines.append("")

    # Write report
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Report saved to {save_path}")


def main(
    config: dict,
    project_root: Path | None = None,
) -> None:
    """Load all results, generate figures and report."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["data_dir"]
    reports_dir = project_root / config["reports_dir"]
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = data_dir / "results" / "probe_results.json"
    if not results_path.exists():
        logging.error(f"Results not found at {results_path}")
        return

    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Load splits
    splits = load_splits(data_dir / "splits.json")
    test_idx = np.array(splits["test"])

    # Generate figures for each model
    for model_key in config.get("models", {}):
        npz_path = data_dir / "embeddings" / f"{model_key}_length_matched.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]

        # t-SNE on test set
        plot_tsne(
            embeddings[test_idx],
            labels[test_idx],
            title=f"t-SNE: {model_key} (test set)",
            save_path=figures_dir / f"tsne_{model_key}.png",
        )

        # PCA variance
        plot_pca_variance(
            embeddings,
            title=f"PCA Explained Variance: {model_key}",
            save_path=figures_dir / f"pca_variance_{model_key}.png",
        )

        # Confusion matrices from results
        if model_key in all_results:
            for clf_key, metrics in all_results[model_key].items():
                test_metrics = metrics.get("test", {})
                cm = test_metrics.get("confusion_matrix")
                if cm is not None:
                    # Reconstruct y_true and y_pred from confusion matrix
                    # Instead, we re-predict from saved data
                    # For confusion matrix plot, we just use the stored CM
                    cm_array = np.array(cm)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm_array,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=LABEL_NAMES,
                        yticklabels=LABEL_NAMES,
                        ax=ax,
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(f"Confusion Matrix: {model_key} + {clf_key}")
                    plt.tight_layout()
                    save_p = figures_dir / f"confusion_{model_key}_{clf_key}.png"
                    plt.savefig(save_p, dpi=150)
                    plt.close()
                    logging.info(f"Confusion matrix saved to {save_p}")

    # Generate report
    generate_report(
        all_results,
        config,
        save_path=reports_dir / "sh1_results.md",
    )
