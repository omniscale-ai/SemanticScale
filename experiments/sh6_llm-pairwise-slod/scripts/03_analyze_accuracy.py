#!/usr/bin/env python
"""SH6 — Stage 3a: Accuracy analysis across runs of a dataset.

Scans {data_dir}/{dataset}/*/summary.json and produces:
    reports/{dataset}/accuracy_comparison.png
    reports/{dataset}/summary.md

Usage:
    python scripts/03_analyze_accuracy.py --config config-frontierscience-nano.yaml
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from semanticscale.sh6 import datasets as ds
from semanticscale.utils import load_config, load_jsonl, setup_logging

logger = logging.getLogger(__name__)


def load_all_summaries(dataset_dir: Path) -> list[tuple[Path, dict]]:
    summaries: list[tuple[Path, dict]] = []
    for path in sorted(dataset_dir.glob("**/summary.json")):
        with open(path) as f:
            summaries.append((path, json.load(f)))
    return summaries


def compute_slice_accuracy(
    config: dict, traces_path: Path
) -> dict[str, dict] | None:
    """Group traces by ``slice_label`` and compute per-slice accuracy.

    Returns ``None`` if the dataset has no slice dimension or the file is
    missing. Otherwise returns ``{slice_value: {total, correct, errors,
    accuracy}}`` (same shape as ``score_results``'s ``by_subject``).
    """
    if not traces_path.exists():
        return None
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(traces_path):
        label = ds.slice_label(config, row)
        if label is None:
            continue
        grouped_rows[label].append(row)
    if not grouped_rows:
        return None
    return {
        label: {
            "total": summary["total"],
            "correct": summary["correct"],
            "errors": summary["errors"],
            "accuracy": summary["accuracy"],
        }
        for label, summary in sorted(
            (
                (label, ds.score_results(config, rows))
                for label, rows in grouped_rows.items()
            ),
            key=lambda item: item[0],
        )
    }


def _slug(s: dict) -> str:
    return s.get("run_slug") or s.get("model_slug") or "unknown"


def plot_accuracy_comparison(summaries: list[dict], out_path: Path, dataset: str) -> None:
    import matplotlib.pyplot as plt

    slugs = [_slug(s) for s in summaries]
    accuracies = [100 * s["accuracy"] for s in summaries]

    fig, ax = plt.subplots(figsize=(max(6, len(slugs) * 1.5), 5))
    bars = ax.bar(range(len(slugs)), accuracies, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(slugs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset} — Accuracy by Run")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy comparison to %s", out_path)


def write_summary_md(
    summaries: list[dict],
    out_path: Path,
    dataset: str,
    slice_breakdowns: dict[str, dict[str, dict]] | None = None,
    slice_name: str | None = None,
) -> None:
    lines = [
        f"# SH6 {dataset} — Results Summary",
        "",
        "## Overall Accuracy",
        "",
        "| Run slug | Accuracy | Correct | Answered | Errors |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {_slug(s)} "
            f"| {100 * s['accuracy']:.1f}% "
            f"| {s['correct']} "
            f"| {s['answered']} "
            f"| {s['errors']} |"
        )

    if any("judgment" in s for s in summaries):
        lines += [
            "",
            "## AgentHallu Judgment and Attribution",
            "",
            "| Run slug | Method | Judge model | Macro-F1 | Macro-recall | Accuracy | Step localization | GEVAL | Hallucinated answered |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in summaries:
            judgment = s.get("judgment")
            attribution = s.get("attribution")
            if not judgment or not attribution:
                continue
            lines.append(
                f"| {_slug(s)} "
                f"| {s.get('evaluation_method', '') or 'n/a'} "
                f"| {s.get('model', '') or 'n/a'} "
                f"| {100 * judgment.get('macro_f1', 0.0):.1f}% "
                f"| {100 * judgment.get('macro_recall', 0.0):.1f}% "
                f"| {100 * judgment.get('accuracy', 0.0):.1f}% "
                f"| {100 * attribution.get('step_localization_accuracy', 0.0):.1f}% "
                f"| {attribution.get('geval_score', 0.0):.2f} "
                f"| {attribution.get('hallucinated_answered', 0)} |"
            )

    for s in summaries:
        by_subj = s.get("by_subject", {})
        if not by_subj:
            continue
        lines += [
            "",
            f"## {_slug(s)} — Per-Subject Accuracy",
            "",
            "| Subject | Accuracy | Correct | Total | Errors |",
            "|---|---|---|---|---|",
        ]
        for subj, v in sorted(by_subj.items(), key=lambda x: -x[1]["accuracy"]):
            lines.append(
                f"| {subj} "
                f"| {100 * v['accuracy']:.1f}% "
                f"| {v['correct']} "
                f"| {v['total']} "
                f"| {v['errors']} |"
            )

    if slice_breakdowns and slice_name:
        header = slice_name.capitalize()
        for s in summaries:
            slug = _slug(s)
            by_slice = slice_breakdowns.get(slug)
            if not by_slice:
                continue
            lines += [
                "",
                f"## {slug} — Per-{header} Accuracy",
                "",
                f"| {header} | Accuracy | Correct | Total | Errors |",
                "|---|---|---|---|---|",
            ]
            for label, v in sorted(by_slice.items(), key=lambda x: -x[1]["accuracy"]):
                lines.append(
                f"| {label} "
                f"| {100 * v['accuracy']:.1f}% "
                f"| {v['correct']} "
                f"| {v['total']} "
                f"| {v['errors']} |"
                )

    for s in summaries:
        by_category = s.get("by_hallucination_category", {})
        if not by_category:
            continue
        lines += [
            "",
            f"## {_slug(s)} — Hallucination Category Breakdown",
            "",
            "| Category | Macro-F1 | Macro-recall | Accuracy | Step localization | GEVAL | Answered | Errors |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for category, metrics in sorted(
            by_category.items(),
            key=lambda item: item[0],
        ):
            judgment = metrics.get("judgment", {})
            attribution = metrics.get("attribution", {})
            lines.append(
                f"| {category} "
                f"| {100 * judgment.get('macro_f1', 0.0):.1f}% "
                f"| {100 * judgment.get('macro_recall', 0.0):.1f}% "
                f"| {100 * metrics.get('accuracy', 0.0):.1f}% "
                f"| {100 * attribution.get('step_localization_accuracy', 0.0):.1f}% "
                f"| {attribution.get('geval_score', 0.0):.2f} "
                f"| {metrics.get('answered', 0)} "
                f"| {metrics.get('errors', 0)} |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved summary markdown to %s", out_path)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(here / "config.yaml"))
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])

    dataset_name = ds.dataset_name(config)
    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve() / dataset_name

    dataset_dir = data_dir / dataset_name
    summary_pairs = load_all_summaries(dataset_dir)
    if not summary_pairs:
        logger.warning(
            "No summary.json files under %s — run 01_traces.py first", dataset_dir
        )
        return

    summaries = [s for _, s in summary_pairs]
    logger.info("Found %d run(s): %s", len(summaries), [_slug(s) for s in summaries])

    sl_name = ds.slice_name(config)
    slice_breakdowns: dict[str, dict[str, dict]] = {}
    if sl_name:
        for summary_path, summary in summary_pairs:
            traces_path = summary_path.parent / "traces.jsonl"
            by_slice = compute_slice_accuracy(config, traces_path)
            if by_slice:
                slice_breakdowns[_slug(summary)] = by_slice

    plot_accuracy_comparison(summaries, reports_dir / "accuracy_comparison.png", dataset_name)
    write_summary_md(
        summaries,
        reports_dir / "summary.md",
        dataset_name,
        slice_breakdowns=slice_breakdowns or None,
        slice_name=sl_name,
    )


if __name__ == "__main__":
    main()
