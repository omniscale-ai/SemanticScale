#!/usr/bin/env python
"""SH6 — AgentHallu SLoD-only judgment + attribution baseline.

Runs a trajectory-only baseline over existing SH6 artifacts:

1. Judgment from the trajectory feature table derived from `chunk_rankings.jsonl`
2. Attribution from a step-level classifier over per-step SLoD sequence features

Outputs under `reports/{dataset}/{run_slug}/`:
    slod_prediction_summary.json
    slod_prediction.md
    slod_judgment_oof.csv
    slod_attribution_oof.csv
    slod_attribution_step_scores.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from semanticscale.sh6 import datasets as ds
from semanticscale.sh6.agenthallu_slod import evaluate_run, write_outputs
from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(here / "config/agenthallu.yaml"))
    parser.add_argument("--run-slug", default=None, dest="run_slug")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    if ds.dataset_name(config) != "agenthallu":
        raise SystemExit("This script only supports dataset.name=agenthallu")

    summary, outputs, reports_dir = evaluate_run(config, run_slug=args.run_slug)
    paths = write_outputs(summary, outputs, reports_dir)
    logger.info("Saved SLoD baseline summary to %s", paths["summary_json"])
    logger.info("Saved SLoD baseline markdown to %s", paths["summary_md"])


if __name__ == "__main__":
    main()
