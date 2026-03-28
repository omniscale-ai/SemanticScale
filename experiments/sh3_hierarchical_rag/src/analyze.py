"""Analysis for SH3: breakdowns by SLoD class and answer type, plots, markdown report."""

import logging
from pathlib import Path

import numpy as np

from src.utils import load_config, load_json, load_jsonl, save_json
from src.evaluate import compute_attribution_f1, compute_soft_attribution_f1


def breakdown_by_slod(
    retrieval_results: dict,
    slod_predictions: dict,
    questions: list[dict],
    config: dict,
) -> dict:
    """Compute attribution F1 broken down by predicted SLoD class.

    Returns:
        Dict: condition -> k -> slod_class -> {f1, n_questions}.
    """
    threshold = config["evaluation"]["matching_threshold"]
    eval_split = config["qasper"]["eval_split"]
    qid_to_q = {q["question_id"]: q for q in questions}

    results = {}
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]

    for condition in conditions:
        cond_results = {}
        for k in top_k_values:
            k_str = str(k)
            entries = retrieval_results.get(condition, {}).get(k_str, [])

            by_class = {"macro": [], "meso": [], "micro": []}
            for entry in entries:
                qid = entry["question_id"]
                q = qid_to_q.get(qid)
                if not q or q.get("split") != eval_split:
                    continue

                pred = slod_predictions.get(qid, {}).get("predicted_slod", "unknown")
                if pred not in by_class:
                    continue

                retrieved_texts = [r["text"] for r in entry.get("retrieved", [])]
                gold_texts = q["gold_evidence"]
                attr = compute_attribution_f1(retrieved_texts, gold_texts, threshold)
                by_class[pred].append(attr["f1"])

            class_metrics = {}
            for cls, scores in by_class.items():
                if scores:
                    class_metrics[cls] = {
                        "f1": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "n_questions": len(scores),
                    }
            cond_results[k_str] = class_metrics
        results[condition] = cond_results

    return results


def breakdown_by_answer_type(
    retrieval_results: dict,
    questions: list[dict],
    config: dict,
) -> dict:
    """Compute attribution F1 broken down by answer type.

    Returns:
        Dict: condition -> k -> answer_type -> {f1, n_questions}.
    """
    threshold = config["evaluation"]["matching_threshold"]
    eval_split = config["qasper"]["eval_split"]
    qid_to_q = {q["question_id"]: q for q in questions}

    results = {}
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]

    for condition in conditions:
        cond_results = {}
        for k in top_k_values:
            k_str = str(k)
            entries = retrieval_results.get(condition, {}).get(k_str, [])

            by_type = {}
            for entry in entries:
                qid = entry["question_id"]
                q = qid_to_q.get(qid)
                if not q or q.get("split") != eval_split:
                    continue

                atype = q.get("answer_type", "unknown")
                if atype not in by_type:
                    by_type[atype] = []

                retrieved_texts = [r["text"] for r in entry.get("retrieved", [])]
                gold_texts = q["gold_evidence"]
                attr = compute_attribution_f1(retrieved_texts, gold_texts, threshold)
                by_type[atype].append(attr["f1"])

            type_metrics = {}
            for atype, scores in by_type.items():
                if scores:
                    type_metrics[atype] = {
                        "f1": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "n_questions": len(scores),
                    }
            cond_results[k_str] = type_metrics
        results[condition] = cond_results

    return results


def confusion_analysis(
    retrieval_results: dict,
    slod_predictions: dict,
    questions: list[dict],
    config: dict,
) -> dict:
    """Analyze when SLoD-routing hurts: compare slod_routed vs best baseline per question.

    Returns:
        Dict with hurt/helped/neutral counts and examples.
    """
    threshold = config["evaluation"]["matching_threshold"]
    eval_split = config["qasper"]["eval_split"]
    qid_to_q = {q["question_id"]: q for q in questions}

    # Use k=5 as the representative k for confusion analysis
    k_str = "5"

    baselines = ["chunks_only", "summaries_only", "naive_hybrid"]

    # Get slod_routed results
    slod_entries = {
        e["question_id"]: e
        for e in retrieval_results.get("slod_routed", {}).get(k_str, [])
    }

    # Get best baseline F1 per question
    best_baseline_f1 = {}
    best_baseline_name = {}
    for baseline in baselines:
        for entry in retrieval_results.get(baseline, {}).get(k_str, []):
            qid = entry["question_id"]
            q = qid_to_q.get(qid)
            if not q or q.get("split") != eval_split:
                continue

            retrieved_texts = [r["text"] for r in entry.get("retrieved", [])]
            gold_texts = q["gold_evidence"]
            attr = compute_attribution_f1(retrieved_texts, gold_texts, threshold)

            if qid not in best_baseline_f1 or attr["f1"] > best_baseline_f1[qid]:
                best_baseline_f1[qid] = attr["f1"]
                best_baseline_name[qid] = baseline

    # Compare
    helped = []
    hurt = []
    neutral = []

    for qid, slod_entry in slod_entries.items():
        q = qid_to_q.get(qid)
        if not q or q.get("split") != eval_split:
            continue

        retrieved_texts = [r["text"] for r in slod_entry.get("retrieved", [])]
        gold_texts = q["gold_evidence"]
        slod_attr = compute_attribution_f1(retrieved_texts, gold_texts, threshold)
        slod_f1 = slod_attr["f1"]

        bb_f1 = best_baseline_f1.get(qid, 0.0)
        bb_name = best_baseline_name.get(qid, "unknown")
        pred = slod_predictions.get(qid, {}).get("predicted_slod", "unknown")

        diff = slod_f1 - bb_f1
        info = {
            "question_id": qid,
            "predicted_slod": pred,
            "answer_type": q.get("answer_type", "unknown"),
            "slod_f1": slod_f1,
            "best_baseline_f1": bb_f1,
            "best_baseline": bb_name,
            "diff": diff,
        }

        if diff > 0.01:
            helped.append(info)
        elif diff < -0.01:
            hurt.append(info)
        else:
            neutral.append(info)

    # Analyze hurt cases
    hurt_by_pred_class = {}
    for h in hurt:
        cls = h["predicted_slod"]
        if cls not in hurt_by_pred_class:
            hurt_by_pred_class[cls] = 0
        hurt_by_pred_class[cls] += 1

    return {
        "k": int(k_str),
        "n_helped": len(helped),
        "n_hurt": len(hurt),
        "n_neutral": len(neutral),
        "mean_helped_diff": float(np.mean([h["diff"] for h in helped])) if helped else 0.0,
        "mean_hurt_diff": float(np.mean([h["diff"] for h in hurt])) if hurt else 0.0,
        "hurt_by_predicted_class": hurt_by_pred_class,
        "helped_examples": sorted(helped, key=lambda x: -x["diff"])[:5],
        "hurt_examples": sorted(hurt, key=lambda x: x["diff"])[:5],
    }


def plot_attribution_f1(metrics: dict, output_dir: Path, config: dict) -> None:
    """Bar chart comparing attribution F1 across conditions at each k."""
    import matplotlib.pyplot as plt

    fig_cfg = config["analysis"]["figures"]
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(top_k_values))
    width = 0.18
    offsets = np.arange(len(conditions)) - (len(conditions) - 1) / 2

    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    labels = {
        "chunks_only": "Chunks (meso)",
        "summaries_only": "Summaries (macro)",
        "naive_hybrid": "Naive Hybrid",
        "slod_routed": "SLoD-Routed",
    }

    for i, condition in enumerate(conditions):
        f1_values = []
        for k in top_k_values:
            k_str = str(k)
            f1 = metrics.get(condition, {}).get(k_str, {}).get("attribution_f1", 0.0)
            f1_values.append(f1)

        ax.bar(
            x + offsets[i] * width,
            f1_values,
            width,
            label=labels.get(condition, condition),
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Top-k")
    ax.set_ylabel("Attribution F1")
    ax.set_title("Evidence Attribution F1 by Condition and Top-k")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in top_k_values])
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    out_path = output_dir / "attribution_f1_by_condition.png"
    fig.savefig(out_path, dpi=fig_cfg["dpi"])
    plt.close(fig)
    logging.info(f"Saved plot: {out_path}")


def plot_recall_curves(metrics: dict, output_dir: Path, config: dict) -> None:
    """Recall@k curves for all conditions."""
    import matplotlib.pyplot as plt

    fig_cfg = config["analysis"]["figures"]
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    markers = ["o", "s", "^", "D"]
    labels = {
        "chunks_only": "Chunks (meso)",
        "summaries_only": "Summaries (macro)",
        "naive_hybrid": "Naive Hybrid",
        "slod_routed": "SLoD-Routed",
    }

    for i, condition in enumerate(conditions):
        recall_values = []
        for k in top_k_values:
            k_str = str(k)
            rec = metrics.get(condition, {}).get(k_str, {}).get("recall_at_k", 0.0)
            recall_values.append(rec)

        ax.plot(
            top_k_values,
            recall_values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=labels.get(condition, condition),
            linewidth=2,
        )

    ax.set_xlabel("Top-k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Evidence Recall@k by Condition")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "recall_at_k_curves.png"
    fig.savefig(out_path, dpi=fig_cfg["dpi"])
    plt.close(fig)
    logging.info(f"Saved plot: {out_path}")


def plot_slod_breakdown(slod_breakdown: dict, output_dir: Path, config: dict) -> None:
    """Bar chart showing F1 by predicted SLoD class for SLoD-routed condition."""
    import matplotlib.pyplot as plt

    fig_cfg = config["analysis"]["figures"]

    # Use k=5 for the breakdown
    k_str = "5"
    slod_data = slod_breakdown.get("slod_routed", {}).get(k_str, {})

    if not slod_data:
        logging.warning("No SLoD breakdown data available for k=5")
        return

    classes = ["macro", "meso", "micro"]
    f1_values = [slod_data.get(c, {}).get("f1", 0.0) for c in classes]
    n_values = [slod_data.get(c, {}).get("n_questions", 0) for c in classes]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    bars = ax.bar(classes, f1_values, color=colors)

    # Add count labels on bars
    for bar, n in zip(bars, n_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Predicted SLoD Class")
    ax.set_ylabel("Attribution F1")
    ax.set_title("SLoD-Routed F1 by Predicted Query Class (k=5)")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    out_path = output_dir / "slod_routing_breakdown.png"
    fig.savefig(out_path, dpi=fig_cfg["dpi"])
    plt.close(fig)
    logging.info(f"Saved plot: {out_path}")


def plot_confusion_analysis(confusion: dict, output_dir: Path, config: dict) -> None:
    """Bar chart showing helped/hurt/neutral counts."""
    import matplotlib.pyplot as plt

    fig_cfg = config["analysis"]["figures"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: helped/hurt/neutral counts
    categories = ["Helped", "Hurt", "Neutral"]
    counts = [confusion["n_helped"], confusion["n_hurt"], confusion["n_neutral"]]
    colors = ["#55a868", "#c44e52", "#cccccc"]
    axes[0].bar(categories, counts, color=colors)
    axes[0].set_ylabel("Number of Questions")
    axes[0].set_title(f"SLoD-Routing Impact (k={confusion['k']})")

    # Right: hurt by predicted class
    hurt_by_class = confusion.get("hurt_by_predicted_class", {})
    if hurt_by_class:
        classes = list(hurt_by_class.keys())
        hurt_counts = [hurt_by_class[c] for c in classes]
        axes[1].bar(classes, hurt_counts, color="#c44e52")
        axes[1].set_ylabel("Number of Hurt Questions")
        axes[1].set_title("Hurt Cases by Predicted SLoD Class")
    else:
        axes[1].text(0.5, 0.5, "No hurt cases", ha="center", va="center")
        axes[1].set_title("Hurt Cases by Predicted SLoD Class")

    plt.tight_layout()

    out_path = output_dir / "confusion_analysis.png"
    fig.savefig(out_path, dpi=fig_cfg["dpi"])
    plt.close(fig)
    logging.info(f"Saved plot: {out_path}")


def plot_binary_vs_soft_f1(metrics: dict, output_dir: Path, config: dict) -> None:
    """Grouped bar chart comparing binary F1 vs soft F1 per condition at k=5."""
    import matplotlib.pyplot as plt

    fig_cfg = config["analysis"]["figures"]
    conditions = config["retrieval"]["conditions"]

    # Check if soft metrics exist
    has_soft = any(
        metrics.get(c, {}).get("5", {}).get("soft_attribution_f1") is not None
        for c in conditions
    )
    if not has_soft:
        logging.warning("No soft metrics available, skipping binary vs soft plot")
        return

    labels_map = {
        "chunks_only": "Chunks",
        "summaries_only": "Summaries",
        "naive_hybrid": "Naive Hybrid",
        "naive_hybrid_parent": "Naive Hyb+P",
        "slod_routed": "SLoD-Routed",
        "slod_routed_v2": "SLoD-v2",
        "slod_weighted": "SLoD-Wtd",
        "slod_weighted_parent": "SLoD-Wtd+P",
    }

    cond_labels = []
    binary_vals = []
    soft_vals = []

    for c in conditions:
        data = metrics.get(c, {}).get("5", {})
        bf1 = data.get("attribution_f1", 0.0)
        sf1 = data.get("soft_attribution_f1")
        if sf1 is None:
            continue
        cond_labels.append(labels_map.get(c, c))
        binary_vals.append(bf1)
        soft_vals.append(sf1)

    if not cond_labels:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cond_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, binary_vals, width, label="Binary F1 (threshold=0.5)", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, soft_vals, width, label="Soft F1 (partial credit)", color="#55a868")

    ax.set_xlabel("Condition")
    ax.set_ylabel("F1 Score")
    ax.set_title("Binary vs Soft Attribution F1 at k=5")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(binary_vals), max(soft_vals)) * 1.15)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "binary_vs_soft_f1.png"
    fig.savefig(out_path, dpi=fig_cfg["dpi"])
    plt.close(fig)
    logging.info(f"Saved plot: {out_path}")


def generate_report(
    metrics: dict,
    bootstrap_results: dict,
    slod_breakdown: dict,
    type_breakdown: dict,
    confusion: dict,
    config: dict,
    output_path: Path,
) -> None:
    """Generate markdown report with tables and figure references."""
    conditions = config["retrieval"]["conditions"]
    top_k_values = config["retrieval"]["top_k_values"]

    lines = []
    lines.append("# SH3 Results: SLoD-Routed Hierarchical RAG")
    lines.append("")
    lines.append("## 1. Main Results: Attribution F1 by Condition and Top-k")
    lines.append("")

    # Main results table
    header = "| Condition | " + " | ".join([f"k={k}" for k in top_k_values]) + " |"
    sep = "|" + "|".join(["---"] * (len(top_k_values) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    labels = {
        "chunks_only": "Chunks (meso)",
        "summaries_only": "Summaries (macro)",
        "naive_hybrid": "Naive Hybrid",
        "naive_hybrid_parent": "Naive Hybrid+Parent",
        "slod_routed": "SLoD-Routed",
        "slod_routed_v2": "SLoD-Routed-v2",
        "slod_weighted": "SLoD-Weighted",
        "slod_weighted_parent": "**SLoD-Weighted+Parent**",
    }

    for condition in conditions:
        row = f"| {labels.get(condition, condition)} |"
        for k in top_k_values:
            k_str = str(k)
            f1 = metrics.get(condition, {}).get(k_str, {}).get("attribution_f1", 0.0)
            row += f" {f1:.4f} |"
        lines.append(row)

    lines.append("")
    lines.append("![Attribution F1](figures/attribution_f1_by_condition.png)")
    lines.append("")

    # Recall@k table
    lines.append("## 2. Recall@k")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for condition in conditions:
        row = f"| {labels.get(condition, condition)} |"
        for k in top_k_values:
            k_str = str(k)
            rec = metrics.get(condition, {}).get(k_str, {}).get("recall_at_k", 0.0)
            row += f" {rec:.4f} |"
        lines.append(row)

    lines.append("")
    lines.append("![Recall@k](figures/recall_at_k_curves.png)")
    lines.append("")

    # MRR table
    lines.append("## 3. Mean Reciprocal Rank")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for condition in conditions:
        row = f"| {labels.get(condition, condition)} |"
        for k in top_k_values:
            k_str = str(k)
            mrr = metrics.get(condition, {}).get(k_str, {}).get("mrr", 0.0)
            row += f" {mrr:.4f} |"
        lines.append(row)

    lines.append("")

    # Token cost table
    lines.append("## 4. Mean Token Cost")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for condition in conditions:
        row = f"| {labels.get(condition, condition)} |"
        for k in top_k_values:
            k_str = str(k)
            tc = metrics.get(condition, {}).get(k_str, {}).get("mean_token_cost", 0.0)
            row += f" {tc:.0f} |"
        lines.append(row)

    lines.append("")

    # Soft Attribution F1 (if available)
    has_soft = any(
        metrics.get(c, {}).get(str(top_k_values[0]), {}).get("soft_attribution_f1") is not None
        for c in conditions
    )
    if has_soft:
        lines.append("## 5. Soft Attribution F1 (Partial Credit)")
        lines.append("")
        lines.append("Soft F1 gives proportional credit for partial overlaps instead of binary match/no-match.")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        for condition in conditions:
            row = f"| {labels.get(condition, condition)} |"
            for k in top_k_values:
                k_str = str(k)
                sf1 = metrics.get(condition, {}).get(k_str, {}).get("soft_attribution_f1", 0.0)
                row += f" {sf1:.4f} |"
            lines.append(row)

        lines.append("")

        # Soft P/R table
        lines.append("### Soft Precision and Recall (k=5)")
        lines.append("")
        lines.append("| Condition | Soft Precision | Soft Recall | Soft F1 |")
        lines.append("|-----------|---------------|-------------|---------|")

        for condition in conditions:
            data = metrics.get(condition, {}).get("5", {})
            sp = data.get("soft_precision", 0.0)
            sr = data.get("soft_recall", 0.0)
            sf1 = data.get("soft_attribution_f1", 0.0)
            lines.append(f"| {labels.get(condition, condition)} | {sp:.4f} | {sr:.4f} | {sf1:.4f} |")

        lines.append("")

        # Lift table: soft - binary
        lines.append("### Soft vs Binary F1 Lift (k=5)")
        lines.append("")
        lines.append("| Condition | Binary F1 | Soft F1 | Lift |")
        lines.append("|-----------|-----------|---------|------|")

        for condition in conditions:
            data = metrics.get(condition, {}).get("5", {})
            bf1 = data.get("attribution_f1", 0.0)
            sf1 = data.get("soft_attribution_f1", 0.0)
            lift = sf1 - bf1
            lines.append(f"| {labels.get(condition, condition)} | {bf1:.4f} | {sf1:.4f} | {lift:+.4f} |")

        lines.append("")
        lines.append("![Binary vs Soft F1](figures/binary_vs_soft_f1.png)")
        lines.append("")

    # Bootstrap tests
    section_num = 6 if has_soft else 5
    lines.append(f"## {section_num}. Statistical Significance (Bootstrap Tests)")
    lines.append("")
    lines.append("SLoD-Routed vs each baseline on Attribution F1:")
    lines.append("")
    lines.append("| k | Baseline | Diff | p-value | 95% CI | Significant |")
    lines.append("|---|---------|------|---------|--------|-------------|")

    for k in top_k_values:
        k_str = str(k)
        k_tests = bootstrap_results.get(k_str, {})
        for baseline, test in k_tests.items():
            sig = "Yes" if test.get("significant") else "No"
            lines.append(
                f"| {k} | {labels.get(baseline, baseline)} | "
                f"{test['observed_diff']:.4f} | {test['p_value']:.4f} | "
                f"[{test['ci_lower']:.4f}, {test['ci_upper']:.4f}] | {sig} |"
            )

    lines.append("")

    # Soft F1 bootstrap tests (if available)
    soft_bootstrap = bootstrap_results.get("soft_f1", {})
    soft_primary = bootstrap_results.get("soft_primary_condition", "slod_weighted_parent")
    if soft_bootstrap:
        section_num += 1
        lines.append(f"## {section_num}. Statistical Significance — Soft F1 (Bootstrap Tests)")
        lines.append("")
        lines.append(f"{labels.get(soft_primary, soft_primary)} vs each baseline on Soft Attribution F1:")
        lines.append("")
        lines.append("| k | Baseline | Diff | p-value | 95% CI | Significant |")
        lines.append("|---|---------|------|---------|--------|-------------|")

        for k in top_k_values:
            k_str = str(k)
            k_tests = soft_bootstrap.get(k_str, {})
            for baseline, test in k_tests.items():
                sig = "Yes" if test.get("significant") else "No"
                lines.append(
                    f"| {k} | {labels.get(baseline, baseline)} | "
                    f"{test['observed_diff']:.4f} | {test['p_value']:.4f} | "
                    f"[{test['ci_lower']:.4f}, {test['ci_upper']:.4f}] | {sig} |"
                )

        lines.append("")

    # SLoD breakdown
    section_num += 1
    lines.append(f"## {section_num}. Breakdown by Predicted SLoD Class (k=5)")
    lines.append("")
    lines.append("| Condition | Macro | Meso | Micro |")
    lines.append("|-----------|-------|------|-------|")

    for condition in conditions:
        slod_data = slod_breakdown.get(condition, {}).get("5", {})
        row = f"| {labels.get(condition, condition)} |"
        for cls in ["macro", "meso", "micro"]:
            f1 = slod_data.get(cls, {}).get("f1", 0.0)
            n = slod_data.get(cls, {}).get("n_questions", 0)
            row += f" {f1:.4f} (n={n}) |"
        lines.append(row)

    lines.append("")
    lines.append("![SLoD Breakdown](figures/slod_routing_breakdown.png)")
    lines.append("")

    # Answer type breakdown
    section_num += 1
    lines.append(f"## {section_num}. Breakdown by Answer Type (k=5)")
    lines.append("")

    # Collect all answer types
    all_types = set()
    for condition in conditions:
        type_data = type_breakdown.get(condition, {}).get("5", {})
        all_types.update(type_data.keys())
    all_types = sorted(all_types)

    if all_types:
        header_types = "| Condition | " + " | ".join(all_types) + " |"
        sep_types = "|" + "|".join(["---"] * (len(all_types) + 1)) + "|"
        lines.append(header_types)
        lines.append(sep_types)

        for condition in conditions:
            type_data = type_breakdown.get(condition, {}).get("5", {})
            row = f"| {labels.get(condition, condition)} |"
            for atype in all_types:
                f1 = type_data.get(atype, {}).get("f1", 0.0)
                n = type_data.get(atype, {}).get("n_questions", 0)
                row += f" {f1:.4f} (n={n}) |"
            lines.append(row)

    lines.append("")

    # Confusion analysis
    section_num += 1
    lines.append(f"## {section_num}. Confusion Analysis")
    lines.append("")
    lines.append(f"At k={confusion['k']}:")
    lines.append(f"- **Helped:** {confusion['n_helped']} questions (mean +{confusion['mean_helped_diff']:.4f} F1)")
    lines.append(f"- **Hurt:** {confusion['n_hurt']} questions (mean {confusion['mean_hurt_diff']:.4f} F1)")
    lines.append(f"- **Neutral:** {confusion['n_neutral']} questions")
    lines.append("")

    if confusion.get("hurt_by_predicted_class"):
        lines.append("Hurt cases by predicted SLoD class:")
        for cls, count in sorted(confusion["hurt_by_predicted_class"].items()):
            lines.append(f"- {cls}: {count}")
        lines.append("")

    lines.append("![Confusion Analysis](figures/confusion_analysis.png)")
    lines.append("")

    # Conclusion
    section_num += 1
    lines.append(f"## {section_num}. Conclusion")
    lines.append("")

    # Determine outcome
    best_k = "5"
    slod_f1 = metrics.get("slod_routed", {}).get(best_k, {}).get("attribution_f1", 0.0)
    best_baseline_f1 = 0.0
    best_baseline_name = ""
    for baseline in ["chunks_only", "summaries_only", "naive_hybrid"]:
        bf1 = metrics.get(baseline, {}).get(best_k, {}).get("attribution_f1", 0.0)
        if bf1 > best_baseline_f1:
            best_baseline_f1 = bf1
            best_baseline_name = baseline

    if slod_f1 > best_baseline_f1:
        bootstrap_test = bootstrap_results.get(best_k, {}).get(best_baseline_name, {})
        if bootstrap_test.get("significant"):
            lines.append(f"**SH3 claim SUPPORTED.** SLoD-routed ({slod_f1:.4f}) significantly outperforms "
                        f"the best baseline ({labels.get(best_baseline_name, best_baseline_name)}: {best_baseline_f1:.4f}) "
                        f"at k={best_k} (p={bootstrap_test['p_value']:.4f}).")
        else:
            lines.append(f"**SH3 claim PARTIALLY SUPPORTED.** SLoD-routed ({slod_f1:.4f}) outperforms "
                        f"the best baseline ({labels.get(best_baseline_name, best_baseline_name)}: {best_baseline_f1:.4f}) "
                        f"at k={best_k}, but the difference is not statistically significant "
                        f"(p={bootstrap_test.get('p_value', 'N/A')}).")
    else:
        lines.append(f"**SH3 claim NOT SUPPORTED.** SLoD-routed ({slod_f1:.4f}) does not outperform "
                    f"the best baseline ({labels.get(best_baseline_name, best_baseline_name)}: {best_baseline_f1:.4f}) "
                    f"at k={best_k}.")

    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Report saved to {output_path}")


def run_analysis(config: dict, project_root: Path | None = None) -> None:
    """Orchestrate all analysis: breakdowns, plots, report."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    results_dir = data_dir / "results"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "SH3_report.md"
    if report_path.exists():
        logging.info("Analysis report already exists, skipping.")
        return

    # Load data
    metrics = load_json(results_dir / "evaluation_metrics.json")
    bootstrap_results = load_json(results_dir / "bootstrap_tests.json")
    retrieval_results = load_json(results_dir / "retrieval_results.json")
    slod_predictions = load_json(data_dir / "query_slod_predictions.json")
    questions = load_jsonl(data_dir / "questions.jsonl")

    # Run breakdowns
    logging.info("Computing SLoD breakdown...")
    slod_bd = breakdown_by_slod(retrieval_results, slod_predictions, questions, config)

    logging.info("Computing answer type breakdown...")
    type_bd = breakdown_by_answer_type(retrieval_results, questions, config)

    logging.info("Running confusion analysis...")
    confusion = confusion_analysis(retrieval_results, slod_predictions, questions, config)

    # Save breakdown data
    save_json({
        "by_slod_class": slod_bd,
        "by_answer_type": type_bd,
        "confusion": confusion,
    }, results_dir / "analysis_breakdowns.json")

    # Generate plots
    logging.info("Generating plots...")
    plot_attribution_f1(metrics, figures_dir, config)
    plot_recall_curves(metrics, figures_dir, config)
    plot_slod_breakdown(slod_bd, figures_dir, config)
    plot_confusion_analysis(confusion, figures_dir, config)
    plot_binary_vs_soft_f1(metrics, figures_dir, config)

    # Generate report
    logging.info("Generating report...")
    generate_report(
        metrics, bootstrap_results, slod_bd, type_bd, confusion,
        config, report_path,
    )

    logging.info("Analysis complete.")
