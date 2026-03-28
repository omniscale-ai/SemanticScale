#!/usr/bin/env python
"""LLM spot-check validation for SH0 labeled spans.

Samples 100 spans stratified by (label, confidence), sends each to an
OpenAI-compatible LLM for independent classification, and computes agreement.

The LLM validation is optional - if no API key is available or --skip-llm
is passed, the script samples the spans and writes them out for manual review.

Usage:
    cd playground/SLoD-SH0
    python scripts/run_validation.py [--input data/qasper_slod_spans.jsonl] \
                                     [--n-samples 100] \
                                     [--model gpt-4o-mini] \
                                     [--skip-llm]

Environment:
    OPENAI_API_KEY  - API key for OpenAI-compatible endpoint
    OPENAI_BASE_URL - Optional custom base URL (for local/alternative APIs)
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, resolve_path, read_jsonl, setup_logging

logger = setup_logging("run_validation")

LABELS = ["macro", "meso", "micro"]

LLM_PROMPT_TEMPLATE = """Given this text span from a scientific paper, classify its semantic level of detail:
- MACRO: High-level claims, motivations, conclusions, broad statements
- MESO: Method descriptions, approach overviews, dataset descriptions
- MICRO: Specific numbers, experimental results, implementation details, table data

Text: "{span_text}"
Section: "{section_name}"

Respond with exactly one word: MACRO, MESO, or MICRO."""


def stratified_sample(
    spans: list[dict], n_samples: int = 100, seed: int = 42
) -> list[dict]:
    """Sample spans stratified by (label, confidence).

    Strategy from PLAN.md 4.2:
      - 20 high-confidence per label (60 total)
      - Up to 20 low-confidence from each label (up to 40)
    """
    random.seed(seed)

    HIGH_CONF_THRESHOLD = 0.7
    PER_LABEL_HIGH = 20
    PER_LABEL_LOW = min(20, (n_samples - PER_LABEL_HIGH * len(LABELS)) // len(LABELS))

    # Split by label and confidence
    high_by_label = defaultdict(list)
    low_by_label = defaultdict(list)

    for s in spans:
        if s["confidence"] >= HIGH_CONF_THRESHOLD:
            high_by_label[s["label"]].append(s)
        else:
            low_by_label[s["label"]].append(s)

    sampled = []
    for lbl in LABELS:
        # High confidence
        pool = high_by_label[lbl]
        n = min(PER_LABEL_HIGH, len(pool))
        sampled.extend(random.sample(pool, n))
        if n < PER_LABEL_HIGH:
            logger.warning(
                f"Only {n} high-confidence {lbl} spans available "
                f"(wanted {PER_LABEL_HIGH})"
            )

        # Low confidence
        pool = low_by_label[lbl]
        n = min(PER_LABEL_LOW, len(pool))
        sampled.extend(random.sample(pool, n))

    logger.info(
        f"Sampled {len(sampled)} spans: "
        + ", ".join(
            f"{lbl}={sum(1 for s in sampled if s['label']==lbl)}"
            for lbl in LABELS
        )
    )
    return sampled


def classify_with_llm(
    span: dict, model: str, client
) -> str | None:
    """Classify a single span using the LLM. Returns predicted label or None."""
    prompt = LLM_PROMPT_TEMPLATE.format(
        span_text=span["text"][:2000],  # Truncate very long spans
        section_name=span["section_name"],
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()

        # Parse response
        for lbl in LABELS:
            if lbl in raw:
                return lbl

        logger.warning(f"Unparseable LLM response: '{raw}' for span {span['span_id']}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for {span['span_id']}: {e}")
        return None


def compute_agreement(samples: list[dict]) -> dict:
    """Compute agreement metrics between heuristic and LLM labels."""
    valid = [s for s in samples if s.get("llm_label") is not None]

    if not valid:
        return {"agreement": 0.0, "n_valid": 0, "n_total": len(samples)}

    agree = sum(1 for s in valid if s["label"] == s["llm_label"])
    agreement = agree / len(valid)

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for s in valid:
        confusion[s["label"]][s["llm_label"]] += 1

    # Convert to serializable dict
    confusion_dict = {
        k: dict(v) for k, v in confusion.items()
    }

    return {
        "agreement": round(agreement, 4),
        "n_agree": agree,
        "n_valid": len(valid),
        "n_total": len(samples),
        "n_failed": len(samples) - len(valid),
        "confusion_matrix": confusion_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM spot-check validation")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM calls; only produce the sampled spans for manual review"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["paths"]["labeled_spans"])
    report_path = resolve_path(config["paths"]["validation_report"])

    logger.info(f"Loading spans from {input_path}")
    spans = read_jsonl(input_path)
    logger.info(f"Loaded {len(spans)} spans")

    # Stratified sampling
    samples = stratified_sample(spans, n_samples=args.n_samples, seed=args.seed)

    # LLM classification
    llm_results = None
    if args.skip_llm:
        logger.info("Skipping LLM validation (--skip-llm flag set)")
    elif not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY not set. Skipping LLM validation. "
            "Set the environment variable or use --skip-llm."
        )
    else:
        try:
            from openai import OpenAI

            client_kwargs = {}
            if os.environ.get("OPENAI_BASE_URL"):
                client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

            client = OpenAI(**client_kwargs)
            logger.info(f"Running LLM classification with model={args.model}...")

            from tqdm import tqdm
            for sample in tqdm(samples, desc="LLM spot-check"):
                llm_label = classify_with_llm(sample, args.model, client)
                sample["llm_label"] = llm_label
                # Rate limiting: small delay between calls
                time.sleep(0.1)

            llm_results = compute_agreement(samples)
            logger.info(f"LLM agreement: {llm_results['agreement']:.1%} "
                        f"({llm_results['n_agree']}/{llm_results['n_valid']})")

            if llm_results["agreement"] < 0.75:
                logger.warning(
                    "Agreement is below 75% threshold. "
                    "Consider refining heuristic rules."
                )
            else:
                logger.info("Agreement meets the 75% threshold.")

            # Print confusion matrix
            logger.info("Confusion matrix (rows=heuristic, cols=LLM):")
            cm = llm_results["confusion_matrix"]
            header = "          " + "  ".join(f"{l:>6}" for l in LABELS)
            logger.info(header)
            for row_label in LABELS:
                row = cm.get(row_label, {})
                vals = "  ".join(f"{row.get(l, 0):>6}" for l in LABELS)
                logger.info(f"  {row_label:>6}  {vals}")

        except ImportError:
            logger.warning(
                "openai package not installed. Skipping LLM validation. "
                "Install with: pip install openai"
            )
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")

    # Save sampled spans for review
    samples_path = resolve_path("../../data/sh0/validation_samples.jsonl")
    from src.utils import write_jsonl
    # Strip non-serializable fields and write
    clean_samples = []
    for s in samples:
        cs = {
            "span_id": s["span_id"],
            "text": s["text"][:500],
            "label": s["label"],
            "label_source": s["label_source"],
            "confidence": s["confidence"],
            "section_name": s["section_name"],
            "micro_score": s["micro_score"],
            "word_count": s["word_count"],
        }
        if "llm_label" in s:
            cs["llm_label"] = s["llm_label"]
        clean_samples.append(cs)

    write_jsonl(clean_samples, samples_path)
    logger.info(f"Wrote {len(clean_samples)} validation samples to {samples_path}")

    # Update validation report
    validation_update = {
        "llm_validation": {
            "model": args.model if not args.skip_llm else None,
            "n_samples": len(samples),
            "skipped": args.skip_llm or llm_results is None,
        }
    }
    if llm_results:
        validation_update["llm_validation"].update(llm_results)

    # Merge with existing report if present
    if report_path.exists():
        with open(report_path, "r") as f:
            existing_report = json.load(f)
        existing_report.update(validation_update)
        report = existing_report
    else:
        report = validation_update

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Updated validation report at {report_path}")


if __name__ == "__main__":
    main()
