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
from pathlib import Path

from semanticscale.sh6 import datasets as ds
from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def load_all_summaries(dataset_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(dataset_dir.glob("*/summary.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


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


def write_summary_md(summaries: list[dict], out_path: Path, dataset: str) -> None:
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
    summaries = load_all_summaries(dataset_dir)
    if not summaries:
        logger.warning(
            "No summary.json files under %s — run 01_traces.py first", dataset_dir
        )
        return

    logger.info("Found %d run(s): %s", len(summaries), [_slug(s) for s in summaries])

    plot_accuracy_comparison(summaries, reports_dir / "accuracy_comparison.png", dataset_name)
    write_summary_md(summaries, reports_dir / "summary.md", dataset_name)


if __name__ == "__main__":
    main()
