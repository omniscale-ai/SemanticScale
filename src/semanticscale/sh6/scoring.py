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

    first = results[0] if results else {}
    # Prefer the new dataset-agnostic run_slug; fall back to model_slug for
    # frontierscience records written before the rename.
    run_slug = first.get("run_slug") or first.get("model_slug") or ""

    return {
        "run_slug": run_slug,
        "dataset": first.get("dataset", "frontierscience"),
        "model": first.get("model", ""),
        "reasoning_effort": first.get("reasoning_effort", ""),
        "service_tier": first.get("service_tier", ""),
        "total": total,
        "answered": answered,
        "errors": errors,
        "correct": correct,
        "accuracy": correct / max(answered, 1),
        "by_subject": subject_accuracy,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
