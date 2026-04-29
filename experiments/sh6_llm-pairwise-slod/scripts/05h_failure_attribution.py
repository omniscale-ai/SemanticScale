#!/usr/bin/env python
"""SH6 — SLoD-only failure-attribution baseline (location + type).

Runs a trajectory-only baseline over the existing SH6 artifacts
(``traces.jsonl`` + ``chunk_rankings.jsonl``) for one dataset:

1. **Failure location** — per-step classifier picks the responsible step.
   Runs whenever any trace in the run has ``error_step_index`` populated
   (ProcessBench, AgentHallu, AgentErrorBench).
2. **Failure type** — multi-class classifier predicts the categorical failure
   type. Skipped on datasets that don't carry such a label (e.g. ProcessBench).

Outputs under ``reports/{dataset}/{run_slug}/``:

    failure_attribution_summary.json
    failure_attribution.md
    failure_location_oof.csv               (when location ran)
    failure_location_step_scores.csv       (when location ran)
    failure_type_oof.csv                   (when type ran)
"""

from __future__ import annotations

import argparse
import logging

from semanticscale.sh6.failure_attribution import evaluate_run, write_outputs
from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the SH6 dataset config "
            "(e.g. experiments/sh6_llm-pairwise-slod/config/agenthallu.yaml)"
        ),
    )
    parser.add_argument("--run-slug", default=None, dest="run_slug")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)

    summary, outputs, reports_dir = evaluate_run(config, run_slug=args.run_slug)
    paths = write_outputs(summary, outputs, reports_dir)
    logger.info("Saved failure-attribution summary to %s", paths["summary_json"])
    logger.info("Saved failure-attribution markdown to %s", paths["summary_md"])


if __name__ == "__main__":
    main()
