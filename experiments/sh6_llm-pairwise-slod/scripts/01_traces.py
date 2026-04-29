#!/usr/bin/env python
"""SH6 — Stage 1: Problems → Traces + overall correctness labels.

Dispatches to the dataset configured in config.yaml:

  * ``frontierscience`` — downloads problems, runs LLM inference, and grades
    the outputs with a second LLM call. The grade becomes the overall
    correctness label.

  * ``processbench`` — no inference is needed because ProcessBench already
    ships reasoning traces pre-chunked into ``steps`` together with a
    ``label`` marking the first erroneous step. We just load and normalise.

  * ``agenterrorbench`` — loads trajectory JSON/JSONL exported from
    AgentErrorBench / AgentDebug, and optionally merges detector
    ``critical_error`` annotations when they are present alongside the
    trajectories.

Output:
    {data_dir}/{dataset}/{run_slug}/traces.jsonl
    {data_dir}/{dataset}/{run_slug}/summary.json

Usage:
    python scripts/01_traces.py --config config/frontierscience-nano.yaml
    python scripts/01_traces.py --config config/processbench-gsm8k.yaml
    python scripts/01_traces.py --config config/agenterrorbench.yaml
"""

import argparse
import json
import logging
from pathlib import Path

from semanticscale.sh6.agenthallu_eval import evaluate_traces
from semanticscale.sh6 import datasets as ds
from semanticscale.utils import load_config, save_jsonl, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--model", default=None, help="Override model name (frontierscience)")
    parser.add_argument("--service-tier", default=None, dest="service_tier")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        dest="question_types",
        help="frontierscience: filter by origin (e.g. olympiad research)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="processbench: subsets to load (gsm8k math olympiadbench omnimath)",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help="processbench: restrict to traces from these generator models",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        dest="sample_idx",
        help="Best-of-N sample index. Appends _s{N} to run_slug and seeds inference with N.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict:
    return {
        k: v
        for k, v in {
            "model": args.model,
            "service_tier": args.service_tier,
            "max_samples": args.max_samples,
            "question_types": args.question_types,
            "subsets": args.subsets,
            "generators": args.generators,
            "sample_idx": args.sample_idx,
        }.items()
        if v is not None
    }


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    overrides = _overrides(args)

    dataset_name = ds.dataset_name(config)
    slug = ds.run_slug(config, overrides)

    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    out_dir = data_dir / dataset_name / slug
    traces_path = out_dir / "traces.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info("Dataset: %s | run_slug: %s", dataset_name, slug)
    logger.info("Output dir: %s", out_dir)

    if traces_path.exists() and not args.force:
        logger.info("Output exists at %s — use --force to re-run", traces_path)
        return

    traces = ds.produce_traces(config, project_root, overrides)
    traces = evaluate_traces(traces, config)
    for trace in traces:
        trace.pop("_agenthallu_eval_steps", None)

    save_jsonl(traces, traces_path)
    logger.info("Saved %d traces to %s", len(traces), traces_path)

    summary = ds.score_results(config, traces)
    out_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Summary: accuracy=%.1f%% (%d/%d answered, %d errors)",
        100 * summary["accuracy"],
        summary["correct"],
        summary["answered"],
        summary["errors"],
    )


if __name__ == "__main__":
    main()
