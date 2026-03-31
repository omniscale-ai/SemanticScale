#!/usr/bin/env python
"""SH6 — Stage 1: Run LLM inference on FrontierScience and save results.

Usage:
    python scripts/01_run_inference.py [options]

    --model MODEL               Override default model from config.yaml
    --reasoning-effort EFFORT   Override reasoning effort (low/medium/high)
    --service-tier TIER         Override service_tier passed to the OpenAI client
    --max-samples N             Run on first N items only (default: all)
    --config PATH               Path to config.yaml (default: ../config.yaml)
    --force                     Re-run even if output already exists

Output is written to:
    {data_dir}/{model_slug}/results.jsonl   — one JSON record per question
    {data_dir}/{model_slug}/summary.json    — aggregate accuracy stats
"""

import argparse
import json
import logging
from pathlib import Path

from semanticscale.sh6.dataset import load_frontierscience
from semanticscale.sh6.inference import make_model_slug, run_inference
from semanticscale.sh6.scoring import score_results
from semanticscale.utils import load_config, save_jsonl, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, with config.yaml as the source of defaults."""
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        dest="reasoning_effort",
        help="Reasoning effort: low, medium, high (overrides config)",
    )
    parser.add_argument(
        "--service-tier",
        default=None,
        dest="service_tier",
        help="OpenAI service_tier parameter (overrides config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        help="Limit to first N samples (overrides config)",
    )
    parser.add_argument(
        "--config",
        default=str(here / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load dataset, run inference, save results and summary."""
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])

    # Resolve effective parameters (CLI overrides config)
    model = args.model or config["model"]["name"]
    reasoning_effort = args.reasoning_effort or config["model"]["reasoning_effort"]
    service_tier = args.service_tier or config["model"]["service_tier"]
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else config["dataset"].get("max_samples")
    )

    model_slug = make_model_slug(model, reasoning_effort)
    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    out_dir = data_dir / model_slug
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info(
        "Model: %s | reasoning_effort=%s | service_tier=%s",
        model,
        reasoning_effort,
        service_tier,
    )
    logger.info("Output dir: %s", out_dir)

    if results_path.exists() and not args.force:
        logger.info(
            "Output already exists at %s — use --force to re-run", results_path
        )
        return

    # Load dataset
    items = load_frontierscience(
        hf_path=config["dataset"]["hf_path"],
        split=config["dataset"]["split"],
        max_samples=max_samples,
    )

    # Run inference
    results = run_inference(
        items=items,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        config=config,
    )

    # Save results
    save_jsonl(results, results_path)
    logger.info("Saved %d results to %s", len(results), results_path)

    # Score and save summary
    summary = score_results(results)
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
