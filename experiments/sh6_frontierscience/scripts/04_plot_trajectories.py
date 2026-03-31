#!/usr/bin/env python
"""SH6 — Stage 4: Plot SLoD trajectory figures.

Loads chunk_rankings.jsonl and results.jsonl, then produces two figures:

  reports/trajectory_mean.png    — mean ± 1 std for correct vs wrong answers
  reports/trajectory_examples.png — grid of individual trajectory examples

Usage:
    python scripts/04_plot_trajectories.py [options]

    --config PATH       Path to config YAML (default: ../config.yaml)
    --model-slug SLUG   Which results subdir to use (default: auto-detect)
    --n-examples N      Number of examples per group in the grid (default: 6)
    --seed N            Random seed for example selection (default: 42)
"""

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from semanticscale.sh6.inference import make_model_slug
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
    parser.add_argument("--model-slug", default=None, dest="model_slug")
    parser.add_argument("--n-examples", type=int, default=6, dest="n_examples")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalise(params: list[float]) -> np.ndarray:
    """Zero-mean normalise params for cross-problem comparison."""
    arr = np.array(params, dtype=float)
    return arr - arr.mean()


def _interpolate(params: np.ndarray, n: int = INTERP_N) -> np.ndarray:
    """Resample a trajectory to n evenly-spaced points."""
    x_orig = np.linspace(0.0, 1.0, len(params))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_orig, params)


def _build_trajectories(
    merged: list[dict], field: str, min_chunks: int = MIN_CHUNKS
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (correct_trajs, wrong_trajs) for the given field."""
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


def plot_mean_trajectories(merged: list[dict], reports_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    x = np.linspace(0.0, 1.0, INTERP_N)

    for ax, field, title in zip(axes, ["reasoning", "answer"], ["Reasoning trace", "Answer"]):
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
            ax.plot(x_r, r_traj, color="#2166ac", linewidth=1.5, label="Reasoning", marker="o",
                    markersize=3)

        if len(a_params) >= MIN_CHUNKS:
            a_traj = _interpolate(_normalise(a_params))
            x_a = np.linspace(0.0, 1.0, len(a_traj))
            ax.plot(x_a, a_traj, color="#f4a582", linewidth=1.5, label="Answer", marker="s",
                    markersize=3)

        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        correct_str = "✓" if item.get("is_correct") else "✗"
        ax.set_title(f"{item.get('subject', '?')}  {correct_str}", fontsize=9)
        ax.set_xlabel("Position", fontsize=7)
        ax.set_ylabel("SLoD", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("SLoD trajectories — reasoning (blue) and answer (orange)", fontsize=12)
    fig.tight_layout()
    out = reports_dir / "trajectory_examples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])

    model = config["model"]["name"]
    reasoning = config["model"].get("reasoning", {})
    model_slug = args.model_slug or make_model_slug(model, reasoning)

    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    run_dir = data_dir / model_slug
    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    rankings_path = run_dir / "chunk_rankings.jsonl"

    if not results_path.exists():
        logger.error("results.jsonl not found at %s", results_path)
        raise SystemExit(1)
    if not rankings_path.exists():
        logger.error("chunk_rankings.jsonl not found at %s — run 03_slod_trajectory.py first",
                     rankings_path)
        raise SystemExit(1)

    results = load_jsonl(results_path)
    rankings = load_jsonl(rankings_path)

    # Merge by id
    rank_by_id = {r["id"]: r for r in rankings}
    merged = []
    for res in results:
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
    logger.info("Done. Figures written to %s", reports_dir)


if __name__ == "__main__":
    main()
