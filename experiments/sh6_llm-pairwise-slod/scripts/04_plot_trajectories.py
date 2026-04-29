#!/usr/bin/env python
"""SH6 — Stage 3b: SLoD trajectory analysis.

Loads traces.jsonl + chunk_rankings.jsonl for one run, merges them by id,
and writes two figures per run:

    reports/{dataset}/{run_slug}/trajectory_mean.png
    reports/{dataset}/{run_slug}/trajectory_examples.png

Usage:
    python scripts/04_plot_trajectories.py --config config-frontierscience-nano.yaml
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from semanticscale.sh6 import datasets as ds
from semanticscale.utils import load_config, load_jsonl, setup_logging

logger = logging.getLogger(__name__)

GRID_COLS = 3
MIN_CHUNKS = 3
INTERP_N = 20


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--run-slug", default=None, dest="run_slug")
    parser.add_argument("--n-examples", type=int, default=6, dest="n_examples")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalise(params: list[float]) -> np.ndarray:
    arr = np.array(params, dtype=float)
    return arr - arr.mean()


def _interpolate(params: np.ndarray, n: int = INTERP_N) -> np.ndarray:
    x_orig = np.linspace(0.0, 1.0, len(params))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_orig, params)


def _build_trajectories(
    merged: list[dict], field: str, min_chunks: int = MIN_CHUNKS
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    correct, wrong = [], []
    params_key = f"{field}_params"
    for item in merged:
        params = item.get(params_key) or []
        if len(params) < min_chunks:
            continue
        traj = _interpolate(_normalise(params))
        if item.get("is_correct"):
            correct.append(traj)
        else:
            wrong.append(traj)
    return correct, wrong


def _mean_band(trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.stack(trajs)
    return mat.mean(axis=0), mat.std(axis=0), np.linspace(0.0, 1.0, INTERP_N)


def _has_answer_traces(merged: list[dict]) -> bool:
    return any((m.get("answer_params") or []) for m in merged)


def plot_mean_trajectories(merged: list[dict], reports_dir: Path) -> None:
    fields = [("reasoning", "Reasoning trace")]
    if _has_answer_traces(merged):
        fields.append(("answer", "Answer"))

    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4), sharey=False, squeeze=False)
    axes_flat = axes[0]
    x = np.linspace(0.0, 1.0, INTERP_N)

    for ax, (field, title) in zip(axes_flat, fields):
        correct, wrong = _build_trajectories(merged, field)
        for trajs, color, label in [
            (correct, "#2166ac", f"Correct (n={len(correct)})"),
            (wrong, "#d6604d", f"Wrong (n={len(wrong)})"),
        ]:
            if not trajs:
                continue
            mean, std, _ = _mean_band(trajs)
            ax.plot(x, mean, color=color, linewidth=2, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("Normalised position")
        ax.set_ylabel("SLoD parameter (higher = more abstract)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Mean SLoD trajectory: correct vs wrong answers", fontsize=13)
    fig.tight_layout()
    out = reports_dir / "trajectory_mean.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_example_trajectories(
    merged: list[dict], reports_dir: Path, n_examples: int, seed: int
) -> None:
    rng = random.Random(seed)
    correct_items = [m for m in merged if m.get("is_correct") and
                     len(m.get("reasoning_params") or []) >= MIN_CHUNKS]
    wrong_items = [m for m in merged if not m.get("is_correct") and
                   len(m.get("reasoning_params") or []) >= MIN_CHUNKS]

    correct_sample = rng.sample(correct_items, min(n_examples, len(correct_items)))
    wrong_sample = rng.sample(wrong_items, min(n_examples, len(wrong_items)))
    all_samples = correct_sample + wrong_sample

    n_plots = len(all_samples)
    if n_plots == 0:
        logger.warning("No items with >= %d reasoning chunks; skipping", MIN_CHUNKS)
        return
    n_rows = (n_plots + GRID_COLS - 1) // GRID_COLS
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(GRID_COLS * 4, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    for i, item in enumerate(all_samples):
        ax = axes_flat[i]
        r_params = item.get("reasoning_params") or []
        a_params = item.get("answer_params") or []

        if len(r_params) >= MIN_CHUNKS:
            r_traj = _interpolate(_normalise(r_params))
            x_r = np.linspace(0.0, 1.0, len(r_traj))
            ax.plot(x_r, r_traj, color="#2166ac", linewidth=1.5,
                    label="Reasoning", marker="o", markersize=3)

        if len(a_params) >= MIN_CHUNKS:
            a_traj = _interpolate(_normalise(a_params))
            x_a = np.linspace(0.0, 1.0, len(a_traj))
            ax.plot(x_a, a_traj, color="#f4a582", linewidth=1.5,
                    label="Answer", marker="s", markersize=3)

        # ProcessBench: if there's an error_step_index, mark it
        err_idx = item.get("error_step_index")
        if err_idx is not None and len(r_params) > 0:
            x_err = err_idx / max(len(r_params) - 1, 1)
            ax.axvline(x_err, color="#b2182b", linewidth=1.0, linestyle=":", alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        correct_str = "correct" if item.get("is_correct") else "wrong"
        ax.set_title(f"{item.get('subject', '?')}  {correct_str}", fontsize=9)
        ax.set_xlabel("Position", fontsize=7)
        ax.set_ylabel("SLoD", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("SLoD trajectories — reasoning (blue) and answer (orange)", fontsize=12)
    fig.tight_layout()
    out = reports_dir / "trajectory_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def _safe_slice_dir(value: str) -> str:
    return value.replace("/", "_")


def _qualifying_count(items: list[dict]) -> int:
    return sum(1 for m in items if len(m.get("reasoning_params") or []) >= MIN_CHUNKS)


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])

    dataset_name = ds.dataset_name(config)
    slug = args.run_slug or ds.run_slug(config)

    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    run_dir = data_dir / dataset_name / slug
    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve() / dataset_name / slug
    reports_dir.mkdir(parents=True, exist_ok=True)

    traces_path = run_dir / "traces.jsonl"
    rankings_path = run_dir / "chunk_rankings.jsonl"

    if not traces_path.exists():
        logger.error("traces.jsonl not found at %s", traces_path)
        raise SystemExit(1)
    if not rankings_path.exists():
        logger.error(
            "chunk_rankings.jsonl not found at %s — run 02_slod.py first", rankings_path
        )
        raise SystemExit(1)

    traces = load_jsonl(traces_path)
    rankings = load_jsonl(rankings_path)

    rank_by_id = {r["id"]: r for r in rankings}
    merged = []
    for res in traces:
        rank = rank_by_id.get(res["id"])
        if rank is None:
            continue
        merged.append({**res, **rank})

    logger.info(
        "Merged %d items (%d correct, %d wrong)",
        len(merged),
        sum(1 for m in merged if m.get("is_correct")),
        sum(1 for m in merged if not m.get("is_correct")),
    )

    plot_mean_trajectories(merged, reports_dir)
    plot_example_trajectories(merged, reports_dir, args.n_examples, args.seed)
    logger.info("Global figures written to %s", reports_dir)

    sl_name = ds.slice_name(config)
    if not sl_name:
        return

    slices: dict[str, list[dict]] = defaultdict(list)
    for item in merged:
        label = ds.slice_label(config, item)
        if label is None:
            continue
        slices[label].append(item)

    for label, items in sorted(slices.items()):
        if _qualifying_count(items) < MIN_CHUNKS:
            logger.info(
                "Skipping slice %s=%s: only %d items with >= %d reasoning chunks",
                sl_name, label, _qualifying_count(items), MIN_CHUNKS,
            )
            continue
        slice_dir = reports_dir / f"by-{sl_name}" / _safe_slice_dir(label)
        slice_dir.mkdir(parents=True, exist_ok=True)
        plot_mean_trajectories(items, slice_dir)
        plot_example_trajectories(items, slice_dir, args.n_examples, args.seed)
        logger.info(
            "Slice %s=%s: %d items → %s", sl_name, label, len(items), slice_dir
        )


if __name__ == "__main__":
    main()
