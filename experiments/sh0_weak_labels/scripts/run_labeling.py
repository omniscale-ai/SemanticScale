#!/usr/bin/env python
"""Orchestrator: runs the full SH0 labeling pipeline end-to-end.

Steps:
  1. Load QASPER dataset and flatten to raw JSONL
  2. Apply heuristic labeling to produce labeled spans
  3. Create length-matched subset
  4. Run automated quality checks

Usage:
    cd playground/SLoD-SH0
    python scripts/run_labeling.py [--config config.yaml] [--skip-download]
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import spacy

from src.utils import load_config, resolve_path, read_jsonl, write_jsonl, setup_logging
from src.load_qasper import load_and_flatten
from src.heuristic_labeler import label_all_spans, print_distribution
from src.length_matcher import create_length_matched_subset, print_summary
from src.quality_checks import run_all_checks, generate_distribution_plot

logger = setup_logging("run_labeling")


def main():
    parser = argparse.ArgumentParser(description="Run full SH0 labeling pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip step 1 if raw data already exists"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    t0 = time.time()

    # ── Step 1: Load QASPER ──────────────────────────────────────────────
    raw_path = resolve_path(config["paths"]["raw_data"])

    if args.skip_download and raw_path.exists():
        logger.info(f"Step 1: Skipping download, loading from {raw_path}")
        raw_records = read_jsonl(raw_path)
    else:
        logger.info("Step 1: Loading and flattening QASPER dataset...")
        raw_records = load_and_flatten(config)
        write_jsonl(raw_records, raw_path)
        logger.info(f"Step 1: Wrote {len(raw_records)} records to {raw_path}")

    t1 = time.time()
    logger.info(f"Step 1 completed in {t1 - t0:.1f}s")

    # ── Step 2: Heuristic labeling ───────────────────────────────────────
    logger.info("Step 2: Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "attribute_ruler"])
    # Increase max_length for long paragraphs
    nlp.max_length = 2_000_000

    logger.info("Step 2: Applying heuristic labels...")
    spans = label_all_spans(raw_records, config, nlp)

    spans_path = resolve_path(config["paths"]["labeled_spans"])
    write_jsonl(spans, spans_path)
    logger.info(f"Step 2: Wrote {len(spans)} labeled spans to {spans_path}")
    print_distribution(spans)

    t2 = time.time()
    logger.info(f"Step 2 completed in {t2 - t1:.1f}s")

    # ── Step 3: Length matching ──────────────────────────────────────────
    logger.info("Step 3: Creating length-matched subset...")
    lm_config = config["length_matching"]
    matched = create_length_matched_subset(
        spans,
        bucket_size=lm_config["bucket_size"],
        min_bucket_count=lm_config["min_bucket_count"],
    )

    matched_path = resolve_path(config["paths"]["length_matched"])
    write_jsonl(matched, matched_path)
    logger.info(f"Step 3: Wrote {len(matched)} matched spans to {matched_path}")
    print_summary(spans, matched)

    t3 = time.time()
    logger.info(f"Step 3 completed in {t3 - t2:.1f}s")

    # ── Step 4: Quality checks ───────────────────────────────────────────
    logger.info("Step 4: Running automated quality checks...")
    checks = run_all_checks(spans, config)

    report_path = resolve_path(config["paths"]["validation_report"])
    plot_path = resolve_path(config["paths"]["label_distribution_plot"])

    import json
    report = {
        "total_spans": len(spans),
        "length_matched_spans": len(matched),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    generate_distribution_plot(spans, plot_path)

    all_passed = report["all_passed"]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        logger.info(f"  [{status}] {check['name']}")

    t4 = time.time()
    logger.info(f"Step 4 completed in {t4 - t3:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = t4 - t0
    logger.info("=" * 60)
    logger.info(f"Pipeline completed in {total_time:.1f}s")
    logger.info(f"  Total spans:          {len(spans)}")
    logger.info(f"  Length-matched spans:  {len(matched)}")
    logger.info(f"  Quality checks:       {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logger.info(f"  Output files:")
    logger.info(f"    {spans_path}")
    logger.info(f"    {matched_path}")
    logger.info(f"    {report_path}")
    logger.info(f"    {plot_path}")
    logger.info("=" * 60)

    if not all_passed:
        logger.warning("Review validation_report.json for failed checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
