"""Step 4: Automated sanity checks from PLAN.md section 4.1.

Checks:
  1. Label distribution: no class < 15%
  2. Mean word count per level: macro < meso ~ micro (no 10x gap)
  3. Section coverage: >80% of sections matched by regex
  4. Abstract always macro: 100%
  5. Conclusion always macro: >95%
  6. Methods/Experiments sections have micro spans: >60%

Outputs:
  - JSON report to data/validation_report.json
  - Distribution plot to data/label_distribution.png

Usage:
    python src/quality_checks.py [--input data/qasper_slod_spans.jsonl]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_config, resolve_path, read_jsonl, setup_logging

logger = setup_logging("quality_checks")

LABELS = ["macro", "meso", "micro"]


def check_label_distribution(spans: list[dict], min_fraction: float) -> dict:
    """Check that no class is below min_fraction of total."""
    counts = Counter(s["label"] for s in spans)
    total = len(spans)
    fractions = {lbl: counts.get(lbl, 0) / total for lbl in LABELS}
    passed = all(f >= min_fraction for f in fractions.values())
    return {
        "name": "label_distribution",
        "passed": passed,
        "threshold": min_fraction,
        "fractions": {k: round(v, 4) for k, v in fractions.items()},
        "counts": {k: counts.get(k, 0) for k in LABELS},
        "total": total,
    }


def check_mean_word_count(spans: list[dict]) -> dict:
    """Check that macro word count is not wildly different from micro (no 10x gap)."""
    wc_by_label = defaultdict(list)
    for s in spans:
        wc_by_label[s["label"]].append(s["word_count"])

    means = {}
    for lbl in LABELS:
        wcs = wc_by_label[lbl]
        means[lbl] = sum(wcs) / len(wcs) if wcs else 0

    # Check: max/min ratio should be < 10
    nonzero = [v for v in means.values() if v > 0]
    ratio = max(nonzero) / min(nonzero) if len(nonzero) >= 2 else 1.0
    passed = ratio < 10.0

    return {
        "name": "mean_word_count",
        "passed": passed,
        "means": {k: round(v, 1) for k, v in means.items()},
        "max_min_ratio": round(ratio, 2),
    }


def check_section_coverage(spans: list[dict], threshold: float) -> dict:
    """Check that sufficient spans have a section name matched by regex.

    Uses span-weighted coverage (fraction of spans whose section name is
    classified) rather than unique-name coverage, which gives a more
    accurate picture when ``:::``-delimited names are common.
    """
    from src.heuristic_labeler import classify_section_name

    # Spans eligible for classification (exclude Title, Abstract)
    eligible = [s for s in spans if s.get("section_name") and s["section_name"] not in ("Title", "Abstract")]

    matched_spans = sum(1 for s in eligible if classify_section_name(s["section_name"]) is not None)
    coverage = matched_spans / len(eligible) if eligible else 1.0

    # Also report unique unmatched names for diagnostics
    section_names = {s["section_name"] for s in eligible}
    unmatched = sorted(sn for sn in section_names if classify_section_name(sn) is None)

    return {
        "name": "section_coverage",
        "passed": coverage >= threshold,
        "coverage": round(coverage, 4),
        "threshold": threshold,
        "total_spans": len(eligible),
        "matched_spans": matched_spans,
        "unique_sections": len(section_names),
        "unmatched_examples": unmatched[:20],
    }


def check_abstract_macro(spans: list[dict], threshold: float) -> dict:
    """Check that all abstract spans are labeled macro."""
    abstract_spans = [s for s in spans if s["section_name"] == "Abstract"]
    if not abstract_spans:
        return {"name": "abstract_macro", "passed": False, "detail": "No abstract spans found"}

    macro_count = sum(1 for s in abstract_spans if s["label"] == "macro")
    fraction = macro_count / len(abstract_spans)

    return {
        "name": "abstract_macro",
        "passed": fraction >= threshold,
        "fraction_macro": round(fraction, 4),
        "total_abstracts": len(abstract_spans),
        "threshold": threshold,
    }


def _any_segment_matches(section_name: str, pattern: re.Pattern) -> bool:
    """Return True if any ``:::``-delimited segment of *section_name* matches *pattern*."""
    if not section_name:
        return False
    return any(pattern.match(seg.strip()) for seg in section_name.split(" ::: "))


_CONCLUSION_RE = re.compile(r"(?i)^(conclusion|summary)$")


def check_conclusion_macro(spans: list[dict], threshold: float) -> dict:
    """Check that >95% of conclusion paragraphs are labeled macro."""
    conclusion_spans = [
        s for s in spans
        if s["section_name"] and _any_segment_matches(s["section_name"], _CONCLUSION_RE)
    ]
    if not conclusion_spans:
        return {
            "name": "conclusion_macro",
            "passed": True,
            "detail": "No conclusion spans found (may be labeled differently)",
            "total": 0,
        }

    macro_count = sum(1 for s in conclusion_spans if s["label"] == "macro")
    fraction = macro_count / len(conclusion_spans)

    return {
        "name": "conclusion_macro",
        "passed": fraction >= threshold,
        "fraction_macro": round(fraction, 4),
        "total_conclusions": len(conclusion_spans),
        "threshold": threshold,
    }


def check_methods_micro(spans: list[dict], threshold: float) -> dict:
    """Check that >60% of methods/experiments sections contain micro spans."""
    methods_pattern = re.compile(
        r"(?i)^(experiment(s|al)?(\s+(setup|results|details))?|"
        r"results|implementation(\s+details)?|"
        r"method(s|ology)?|approach)$"
    )

    # Group spans by (paper_id, section_name)
    sections = defaultdict(list)
    for s in spans:
        if s["section_name"] and _any_segment_matches(s["section_name"], methods_pattern):
            key = (s["paper_id"], s["section_name"])
            sections[key].append(s)

    if not sections:
        return {
            "name": "methods_micro",
            "passed": True,
            "detail": "No methods/experiments sections found",
            "total_sections": 0,
        }

    sections_with_micro = sum(
        1 for spans_in_sec in sections.values()
        if any(s["label"] == "micro" for s in spans_in_sec)
    )
    fraction = sections_with_micro / len(sections)

    return {
        "name": "methods_micro",
        "passed": fraction >= threshold,
        "fraction_with_micro": round(fraction, 4),
        "total_sections": len(sections),
        "sections_with_micro": sections_with_micro,
        "threshold": threshold,
    }


def generate_distribution_plot(spans: list[dict], output_path: Path) -> None:
    """Generate label distribution plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Label count bar chart
    counts = Counter(s["label"] for s in spans)
    ax = axes[0]
    colors = {"macro": "#2196F3", "meso": "#FF9800", "micro": "#4CAF50"}
    bars = ax.bar(LABELS, [counts.get(l, 0) for l in LABELS],
                  color=[colors[l] for l in LABELS])
    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    for bar, lbl in zip(bars, LABELS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{counts.get(lbl, 0)}', ha='center', va='bottom')

    # 2. Word count distribution per label
    ax = axes[1]
    for lbl in LABELS:
        wcs = [s["word_count"] for s in spans if s["label"] == lbl]
        if wcs:
            ax.hist(wcs, bins=50, alpha=0.5, label=lbl, color=colors[lbl])
    ax.set_title("Word Count Distribution by Label")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_xlim(0, 500)

    # 3. Confidence distribution
    ax = axes[2]
    for lbl in LABELS:
        confs = [s["confidence"] for s in spans if s["label"] == lbl]
        if confs:
            ax.hist(confs, bins=20, alpha=0.5, label=lbl, color=colors[lbl])
    ax.set_title("Confidence Distribution by Label")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def run_all_checks(spans: list[dict], config: dict) -> list[dict]:
    """Run all automated sanity checks. Returns list of check results."""
    qc = config["quality_checks"]

    checks = [
        check_label_distribution(spans, qc["min_class_fraction"]),
        check_mean_word_count(spans),
        check_section_coverage(spans, qc["section_coverage_threshold"]),
        check_abstract_macro(spans, qc["abstract_macro_threshold"]),
        check_conclusion_macro(spans, qc["conclusion_macro_threshold"]),
        check_methods_micro(spans, qc["methods_micro_threshold"]),
    ]

    return checks


def main():
    parser = argparse.ArgumentParser(description="Run automated quality checks")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["paths"]["labeled_spans"])
    output_path = resolve_path(args.output or config["paths"]["validation_report"])
    plot_path = resolve_path(config["paths"]["label_distribution_plot"])

    logger.info(f"Loading spans from {input_path}")
    spans = read_jsonl(input_path)
    logger.info(f"Loaded {len(spans)} spans")

    # Run checks
    checks = run_all_checks(spans, config)

    # Report results
    all_passed = True
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        logger.info(f"  [{status}] {check['name']}")
        if not check["passed"]:
            all_passed = False
            # Log details for failed checks
            for k, v in check.items():
                if k not in ("name", "passed"):
                    logger.info(f"    {k}: {v}")

    report = {
        "total_spans": len(spans),
        "all_passed": all_passed,
        "checks": checks,
    }

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote validation report to {output_path}")

    # Generate plot
    generate_distribution_plot(spans, plot_path)

    if all_passed:
        logger.info("All quality checks PASSED.")
    else:
        logger.warning("Some quality checks FAILED. Review the report.")

    return all_passed


if __name__ == "__main__":
    main()
