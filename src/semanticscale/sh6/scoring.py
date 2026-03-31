"""Accuracy scoring for SH6 results."""

from collections import defaultdict


def score_results(results: list[dict]) -> dict:
    """Compute overall and per-subject accuracy from a list of result records.

    Returns a summary dict with overall accuracy and per-subject breakdown.
    """
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    answered = total - errors

    correct = sum(r.get("is_correct", False) for r in results)

    by_subject: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "errors": 0}
    )
    for r in results:
        subj = r.get("subject", "unknown")
        by_subject[subj]["total"] += 1
        if r.get("error"):
            by_subject[subj]["errors"] += 1
        elif r.get("is_correct"):
            by_subject[subj]["correct"] += 1

    subject_accuracy = {
        subj: {
            "total": v["total"],
            "correct": v["correct"],
            "errors": v["errors"],
            "accuracy": v["correct"] / max(v["total"] - v["errors"], 1),
        }
        for subj, v in sorted(by_subject.items())
    }

    # Token usage summary
    total_input_tokens = 0
    total_output_tokens = 0
    for r in results:
        usage = r.get("usage") or {}
        total_input_tokens += usage.get("input_tokens") or 0
        total_output_tokens += usage.get("output_tokens") or 0

    model_slug = results[0]["model_slug"] if results else ""

    return {
        "model_slug": model_slug,
        "model": results[0]["model"] if results else "",
        "reasoning_effort": results[0]["reasoning_effort"] if results else "",
        "service_tier": results[0]["service_tier"] if results else "",
        "total": total,
        "answered": answered,
        "errors": errors,
        "correct": correct,
        "accuracy": correct / max(answered, 1),
        "by_subject": subject_accuracy,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
