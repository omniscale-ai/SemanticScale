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
from collections import defaultdict
from pathlib import Path

from semanticscale.sh6 import datasets as ds
from semanticscale.sh6.slod_rank import rank_all
from semanticscale.utils import load_config, load_jsonl, save_jsonl, setup_logging

logger = logging.getLogger(__name__)
MIN_BALANCED_PER_SLICE_CLASS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-slug",
        default=None,
        dest="run_slug",
        help="Override the auto-derived run slug",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of traces to process",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "With --limit N, sample a balanced success/failure subset "
            "(by is_correct)."
        ),
    )
    parser.add_argument(
        "--balance-per-slice",
        action="store_true",
        help=(
            "With --balanced --limit N, balance within the dataset's "
            "slice_label groups and keep at least 100 successes and 100 "
            "failures per slice when available."
        ),
    )
    return parser.parse_args()


def _balanced_sample(traces: list[dict], limit: int) -> list[dict]:
    successes = [t for t in traces if t.get("is_correct") is True]
    failures = [t for t in traces if t.get("is_correct") is False]
    per_class = limit // 2
    if len(successes) < per_class:
        logger.warning(
            "Only %d successful traces available (wanted %d)",
            len(successes), per_class,
        )
    if len(failures) < per_class:
        logger.warning(
            "Only %d failed traces available (wanted %d)",
            len(failures), per_class,
        )
    sampled = successes[:per_class] + failures[:per_class]
    logger.info(
        "Balanced sampling: %d successful + %d failed = %d traces",
        min(len(successes), per_class),
        min(len(failures), per_class),
        len(sampled),
    )
    return sampled


def _balanced_sample_by_slice(traces: list[dict], config: dict, limit: int) -> list[dict]:
    grouped: dict[str, dict[bool, list[dict]]] = defaultdict(lambda: {True: [], False: []})
    skipped_without_slice = 0
    for trace in traces:
        outcome = trace.get("is_correct")
        if outcome not in (True, False):
            continue
        slice_label = ds.slice_label(config, trace)
        if slice_label is None:
            skipped_without_slice += 1
            continue
        grouped[str(slice_label)][bool(outcome)].append(trace)

    if not grouped:
        slice_name = ds.slice_name(config)
        logger.error(
            "--balance-per-slice requested, but dataset %s has no usable %s labels",
            ds.dataset_name(config),
            slice_name or "slice",
        )
        raise SystemExit(1)

    slice_name = ds.slice_name(config) or "slice"
    slice_count = len(grouped)
    requested_per_slice_class = limit // (2 * slice_count)
    per_slice_class = max(
        requested_per_slice_class,
        MIN_BALANCED_PER_SLICE_CLASS,
    )
    minimum_total = 2 * slice_count * MIN_BALANCED_PER_SLICE_CLASS
    if limit < minimum_total:
        logger.info(
            "Requested --limit=%d, but per-slice balancing needs at least %d "
            "traces to keep %d successes and %d failures per %s across %d slices.",
            limit,
            minimum_total,
            MIN_BALANCED_PER_SLICE_CLASS,
            MIN_BALANCED_PER_SLICE_CLASS,
            slice_name,
            slice_count,
        )

    selected_trace_ids: set[int] = set()
    summary_parts: list[str] = []
    for label, buckets in grouped.items():
        success_bucket = buckets[True]
        failure_bucket = buckets[False]
        if len(success_bucket) < per_slice_class:
            logger.warning(
                "%s %s has only %d successful traces available (wanted %d)",
                slice_name,
                label,
                len(success_bucket),
                per_slice_class,
            )
        if len(failure_bucket) < per_slice_class:
            logger.warning(
                "%s %s has only %d failed traces available (wanted %d)",
                slice_name,
                label,
                len(failure_bucket),
                per_slice_class,
            )

        selected_successes = success_bucket[:per_slice_class]
        selected_failures = failure_bucket[:per_slice_class]
        selected_trace_ids.update(id(trace) for trace in selected_successes)
        selected_trace_ids.update(id(trace) for trace in selected_failures)
        summary_parts.append(
            f"{label}: {len(selected_successes)} success + {len(selected_failures)} failure"
        )

    sampled = [trace for trace in traces if id(trace) in selected_trace_ids]
    logger.info(
        "Per-%s balanced sampling: %s = %d traces",
        slice_name,
        "; ".join(summary_parts),
        len(sampled),
    )
    if skipped_without_slice:
        logger.warning(
            "Skipped %d traces without a %s label during per-slice balancing",
            skipped_without_slice,
            slice_name,
        )
    return sampled


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])

    dataset_name = ds.dataset_name(config)
    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    requested_slug = args.run_slug or ds.run_slug(config)
    slug, run_dir = ds.resolve_run_dir(
        config,
        data_dir,
        requested_slug,
        required_files=("traces.jsonl",),
    )
    traces_path = run_dir / "traces.jsonl"
    out_path = run_dir / "chunk_rankings.jsonl"
    if slug != requested_slug:
        logger.info("Resolved run slug %s -> %s", requested_slug, slug)

    if out_path.exists() and not args.force:
        logger.info("Output exists at %s — use --force to re-run", out_path)
        return

    if not traces_path.exists():
        logger.error("traces.jsonl not found at %s — run 01_traces.py first", traces_path)
        raise SystemExit(1)

    traces = load_jsonl(traces_path)
    logger.info("Loaded %d traces from %s", len(traces), traces_path)

    if args.balanced and args.limit is None:
        logger.error("--balanced requires --limit")
        raise SystemExit(1)
    if args.balance_per_slice and not args.balanced:
        logger.error("--balance-per-slice requires --balanced")
        raise SystemExit(1)

    if args.limit is not None:
        if args.balanced:
            if args.balance_per_slice:
                traces = _balanced_sample_by_slice(traces, config, args.limit)
            else:
                traces = _balanced_sample(traces, args.limit)
        else:
            traces = traces[: args.limit]
            logger.info("Limiting to %d traces", len(traces))

    rankings = asyncio.run(rank_all(traces, config, run_dir))

    save_jsonl(rankings, out_path)
    logger.info("Saved %d chunk rankings to %s", len(rankings), out_path)


if __name__ == "__main__":
    main()
