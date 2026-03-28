"""Step 3: Create length-matched subset by binning spans by word count.

For each word-count bucket, sample equal numbers from each label class
(min of the three classes in that bucket). This controls for the length
confound when training SH1.

Usage:
    python src/length_matcher.py [--input data/qasper_slod_spans.jsonl] \
                                 [--output data/qasper_slod_length_matched.jsonl]
"""

import argparse
import random
from collections import defaultdict, Counter

from src.utils import (
    load_config,
    resolve_path,
    read_jsonl,
    write_jsonl,
    setup_logging,
)

logger = setup_logging("length_matcher")

LABELS = ["macro", "meso", "micro"]


def create_length_matched_subset(
    spans: list[dict],
    bucket_size: int = 10,
    min_bucket_count: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Create a length-matched subset by word-count bucketing.

    For each bucket:
      - Group spans by label
      - Take min(count_macro, count_meso, count_micro) per bucket
      - Sample that many from each label class

    Args:
        spans: Full labeled spans list.
        bucket_size: Width of word-count buckets.
        min_bucket_count: Minimum spans per class in a bucket to include it.
        seed: Random seed for reproducibility.

    Returns:
        Length-matched subset of spans.
    """
    random.seed(seed)

    # Group spans by (bucket, label)
    buckets: dict[int, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for span in spans:
        bucket = span["word_count"] // bucket_size
        buckets[bucket][span["label"]].append(span)

    matched = []
    bucket_stats = []

    for bucket_id in sorted(buckets.keys()):
        label_groups = buckets[bucket_id]
        counts = {lbl: len(label_groups.get(lbl, [])) for lbl in LABELS}

        # All three labels must be present with minimum count
        if any(counts[lbl] < min_bucket_count for lbl in LABELS):
            continue

        # Sample min count from each label
        n_sample = min(counts.values())
        for lbl in LABELS:
            sampled = random.sample(label_groups[lbl], n_sample)
            matched.extend(sampled)

        wc_lo = bucket_id * bucket_size
        wc_hi = wc_lo + bucket_size - 1
        bucket_stats.append((wc_lo, wc_hi, n_sample))

    logger.info(f"Length matching: {len(bucket_stats)} buckets used")
    for lo, hi, n in bucket_stats[:10]:
        logger.info(f"  words [{lo}-{hi}]: {n} per class")
    if len(bucket_stats) > 10:
        logger.info(f"  ... and {len(bucket_stats) - 10} more buckets")

    return matched


def print_summary(original: list[dict], matched: list[dict]) -> None:
    """Print before/after summary."""
    orig_counts = Counter(s["label"] for s in original)
    match_counts = Counter(s["label"] for s in matched)

    logger.info("Original distribution:")
    for lbl in LABELS:
        logger.info(f"  {lbl}: {orig_counts.get(lbl, 0)}")

    logger.info("Length-matched distribution:")
    for lbl in LABELS:
        logger.info(f"  {lbl}: {match_counts.get(lbl, 0)}")

    logger.info(f"Total: {len(original)} -> {len(matched)} spans")

    # Word count stats per label in matched set
    for lbl in LABELS:
        wcs = [s["word_count"] for s in matched if s["label"] == lbl]
        if wcs:
            logger.info(
                f"  {lbl} word count: mean={sum(wcs)/len(wcs):.1f}, "
                f"min={min(wcs)}, max={max(wcs)}"
            )


def main():
    parser = argparse.ArgumentParser(description="Create length-matched subset")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["paths"]["labeled_spans"])
    output_path = resolve_path(args.output or config["paths"]["length_matched"])

    lm_config = config["length_matching"]

    logger.info(f"Loading spans from {input_path}")
    spans = read_jsonl(input_path)
    logger.info(f"Loaded {len(spans)} spans")

    matched = create_length_matched_subset(
        spans,
        bucket_size=lm_config["bucket_size"],
        min_bucket_count=lm_config["min_bucket_count"],
    )

    write_jsonl(matched, output_path)
    logger.info(f"Wrote {len(matched)} matched spans to {output_path}")

    print_summary(spans, matched)


if __name__ == "__main__":
    main()
