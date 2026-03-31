#!/usr/bin/env python
"""SH6 — Stage 3: Compute per-chunk SLoD rankings via pairwise LLM comparisons.

For each problem, chunks reasoning_text and answer_text by \\n\\n, runs a
pairwise comparison tournament (consecutive pairs first, then choix-guided
active learning), and computes Bradley-Terry parameters as SLoD ranks.

Usage:
    python scripts/03_slod_trajectory.py [options]

    --config PATH       Path to config YAML (default: ../config.yaml)
    --model-slug SLUG   Which results subdir to load (default: auto-detect)
    --force             Re-run even if chunk_rankings.jsonl already exists

Output:
    {data_dir}/{model_slug}/chunk_rankings.jsonl
"""

import argparse
import asyncio
import logging
from pathlib import Path

from semanticscale.sh6.inference import make_model_slug
from semanticscale.sh6.slod_rank import rank_all
from semanticscale.utils import load_config, load_jsonl, save_jsonl, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(here / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--model-slug",
        default=None,
        dest="model_slug",
        help="Model slug subdir (default: derived from config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists",
    )
    return parser.parse_args()


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
    results_path = run_dir / "results.jsonl"
    out_path = run_dir / "chunk_rankings.jsonl"

    if out_path.exists() and not args.force:
        logger.info("Output already exists at %s — use --force to re-run", out_path)
        return

    if not results_path.exists():
        logger.error("results.jsonl not found at %s — run 01_run_inference.py first", results_path)
        raise SystemExit(1)

    results = load_jsonl(results_path)
    logger.info("Loaded %d results from %s", len(results), results_path)

    rankings = asyncio.run(rank_all(results, config, project_root))

    save_jsonl(rankings, out_path)
    logger.info("Saved %d chunk rankings to %s", len(rankings), out_path)


if __name__ == "__main__":
    main()
