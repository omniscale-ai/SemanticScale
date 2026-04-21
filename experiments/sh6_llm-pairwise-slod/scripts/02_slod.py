#!/usr/bin/env python
"""SH6 — Stage 2: Traces → SLoD.

For each trace, compute a per-chunk Semantic Level of Detail (SLoD) ranking
via pairwise LLM comparisons followed by Bradley-Terry aggregation.

Chunking behaviour:
  * If the trace already carries ``reasoning_chunks`` / ``answer_chunks``
    (ProcessBench), those are used as-is.
  * Otherwise ``reasoning_text`` / ``answer_text`` are split on ``\\n\\n``
    (FrontierScience).

Output:
    {data_dir}/{dataset}/{run_slug}/chunk_rankings.jsonl

Usage:
    python scripts/02_slod.py --config config-frontierscience-nano.yaml
"""

import argparse
import asyncio
import logging
from pathlib import Path

from semanticscale.sh6 import datasets as ds
from semanticscale.sh6.slod_rank import rank_all
from semanticscale.utils import load_config, load_jsonl, save_jsonl, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument(
        "--run-slug",
        default=None,
        dest="run_slug",
        help="Override the auto-derived run slug",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])

    dataset_name = ds.dataset_name(config)
    slug = args.run_slug or ds.run_slug(config)

    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    run_dir = data_dir / dataset_name / slug
    traces_path = run_dir / "traces.jsonl"
    out_path = run_dir / "chunk_rankings.jsonl"

    if out_path.exists() and not args.force:
        logger.info("Output exists at %s — use --force to re-run", out_path)
        return

    if not traces_path.exists():
        logger.error("traces.jsonl not found at %s — run 01_traces.py first", traces_path)
        raise SystemExit(1)

    traces = load_jsonl(traces_path)
    logger.info("Loaded %d traces from %s", len(traces), traces_path)

    rankings = asyncio.run(rank_all(traces, config, project_root))

    save_jsonl(rankings, out_path)
    logger.info("Saved %d chunk rankings to %s", len(rankings), out_path)


if __name__ == "__main__":
    main()
