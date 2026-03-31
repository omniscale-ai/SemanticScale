#!/usr/bin/env python
"""SH6 — Stage 2: Aggregate results across model configurations and produce reports.

Scans {data_dir}/*/summary.json for all completed runs and generates:
    reports/accuracy_comparison.png  — bar chart comparing models
    reports/summary.md               — markdown table with accuracy breakdown

Usage:
    python scripts/02_analyze.py [--config PATH]
"""

import argparse
import json
import logging
from pathlib import Path

from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def load_all_summaries(data_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(data_dir.glob("*/summary.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def plot_accuracy_comparison(summaries: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    slugs = [s["model_slug"] for s in summaries]
    accuracies = [100 * s["accuracy"] for s in summaries]

    fig, ax = plt.subplots(figsize=(max(6, len(slugs) * 1.5), 5))
    bars = ax.bar(range(len(slugs)), accuracies, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(slugs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("FrontierScience — Accuracy by Model Configuration")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy comparison to %s", out_path)


def write_summary_md(summaries: list[dict], out_path: Path) -> None:
    lines = [
        "# SH6 FrontierScience — Results Summary",
        "",
        "## Overall Accuracy",
        "",
        "| Model slug | Accuracy | Correct | Answered | Errors |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model_slug']} "
            f"| {100 * s['accuracy']:.1f}% "
            f"| {s['correct']} "
            f"| {s['answered']} "
            f"| {s['errors']} |"
        )

    # Per-subject breakdown for each run
    for s in summaries:
        by_subj = s.get("by_subject", {})
        if not by_subj:
            continue
        lines += [
            "",
            f"## {s['model_slug']} — Per-Subject Accuracy",
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved summary markdown to %s", out_path)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(here / "config.yaml"), help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve()

    summaries = load_all_summaries(data_dir)
    if not summaries:
        logger.warning("No summary.json files found under %s — run 01_run_inference.py first", data_dir)
        return

    logger.info("Found %d run(s): %s", len(summaries), [s["model_slug"] for s in summaries])

    plot_accuracy_comparison(summaries, reports_dir / "accuracy_comparison.png")
    write_summary_md(summaries, reports_dir / "summary.md")


if __name__ == "__main__":
    main()
