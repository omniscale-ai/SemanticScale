"""GPQA-Diamond dataset loader.

Loads the 198-question Diamond split of Idavidrein/gpqa, formats each row as
a 4-option multiple-choice problem with a seeded shuffle of the answer
choices, and runs LLM inference + grading via the shared SH6 utilities.

The dataset is gated on Hugging Face. Authentication relies on the standard
``huggingface-cli login`` token cache (``~/.cache/huggingface/token``) or the
``HF_TOKEN`` env var.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

from semanticscale.sh6.inference import make_model_slug, run_inference

logger = logging.getLogger(__name__)

DATASET_NAME = "gpqa-diamond"
# GPQA-Diamond runs are already separated by generator at Stage 1 (one model
# per directory), so analysis-time slicing is a no-op.
SLICE_NAME: str | None = None


def slice_label(trace: dict) -> str | None:  # noqa: ARG001
    return None


_DEFAULT_HF_PATH = "Idavidrein/gpqa"
_DEFAULT_CONFIG = "gpqa_diamond"
_DEFAULT_SPLIT = "train"

_LETTERS = ("A", "B", "C", "D")

_LETTER_RE = re.compile(r"\b([ABCDabcd])\b")


def _extract_letter(text: str) -> str | None:
    """Return the first standalone A/B/C/D found in *text*, uppercased."""
    if not text:
        return None
    match = _LETTER_RE.search(text)
    return match.group(1).upper() if match else None


def _grade_deterministic(results: list[dict]) -> list[dict]:
    """MCQ grading: compare extracted letter against ``correct_letter``."""
    graded = []
    for result in results:
        record = dict(result)
        if record.get("error"):
            graded.append(record)
            continue
        candidate = record.get("predicted_answer") or record.get("answer_text", "")
        predicted_letter = _extract_letter(candidate)
        correct_letter = record.get("correct_letter")
        if predicted_letter is None:
            record["grade"] = {
                "type": "final_answer",
                "passed": False,
                "explanation": "No A/B/C/D letter found in the model's final answer.",
            }
            record["is_correct"] = False
        else:
            passed = predicted_letter == correct_letter
            record["grade"] = {
                "type": "final_answer",
                "passed": passed,
                "explanation": f"Predicted {predicted_letter}; correct {correct_letter}.",
            }
            record["is_correct"] = passed
            record["predicted_letter"] = predicted_letter
        graded.append(record)
    return graded

_PROMPT_TEMPLATE = """{question}

A) {a}
B) {b}
C) {c}
D) {d}

Reason step by step, then write your final answer (a single letter A, B, C, or D) on a new line after "FINAL ANSWER:"."""


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a GPQA-Diamond run: same convention as frontierscience."""
    traces_cfg = config["traces"]
    model = overrides.get("model") or traces_cfg["model"]["name"]
    reasoning = traces_cfg["model"].get("reasoning", {})
    return make_model_slug(model, reasoning)


def _build_item(row: dict, fallback_index: int) -> dict:
    """Format one HF row into the unified Stage-1 item dict."""
    correct = str(row["Correct Answer"]).strip()
    distractors = [
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]

    record_id = row.get("Record ID")
    item_id = str(record_id) if record_id not in (None, "") else f"gpqa-{fallback_index:04d}"

    options = [correct, *distractors]
    rng = random.Random(item_id)
    rng.shuffle(options)
    correct_index = options.index(correct)
    correct_letter = _LETTERS[correct_index]

    problem = _PROMPT_TEMPLATE.format(
        question=str(row["Question"]).strip(),
        a=options[0],
        b=options[1],
        c=options[2],
        d=options[3],
    )

    subject = str(
        row.get("High-level domain") or row.get("Subdomain") or "unknown"
    )

    return {
        "id": item_id,
        "problem": problem,
        "correct_answer": f"{correct_letter}) {correct}",
        "subject": subject,
        "origin": "gpqa-diamond",
        "has_final_answer": True,
        # Extras kept for downstream slicing / debugging.
        "subdomain": str(row.get("Subdomain") or ""),
        "correct_letter": correct_letter,
    }


def _load_problems(
    hf_path: str,
    config_name: str,
    split: str,
    max_samples: int | None,
) -> list[dict]:
    from datasets import load_dataset  # deferred: heavy dep

    logger.info(
        "Loading dataset %s (config=%s, split=%s) from the Hugging Face Hub",
        hf_path, config_name, split,
    )
    ds = load_dataset(hf_path, name=config_name, split=split)
    rows = list(ds)
    if max_samples is not None:
        rows = rows[:max_samples]
        logger.info("Truncated to %d samples", len(rows))

    items = [_build_item(row, i) for i, row in enumerate(rows)]
    by_subject: dict[str, int] = {}
    for it in items:
        by_subject[it["subject"]] = by_subject.get(it["subject"], 0) + 1
    logger.info(
        "Loaded %d GPQA-Diamond items (subjects: %s)",
        len(items),
        ", ".join(f"{k}={v}" for k, v in sorted(by_subject.items())),
    )
    return items


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict,
) -> list[dict]:
    """Stage 1: load problems, run inference, grade, and normalise to traces."""
    ds_cfg = config.get("dataset", {})
    max_samples = overrides.get("max_samples")
    if max_samples is None:
        max_samples = ds_cfg.get("max_samples")

    items = _load_problems(
        hf_path=ds_cfg.get("hf_path", _DEFAULT_HF_PATH),
        config_name=ds_cfg.get("config", _DEFAULT_CONFIG),
        split=ds_cfg.get("split", _DEFAULT_SPLIT),
        max_samples=max_samples,
    )

    results = run_inference(
        items=items,
        config=config,
        model_override=overrides.get("model"),
        service_tier_override=overrides.get("service_tier"),
    )

    # Carry over per-item extras the grader/inference layers don't propagate.
    extras_by_id = {it["id"]: it for it in items}
    for r in results:
        extra = extras_by_id.get(r["id"], {})
        r["correct_letter"] = extra.get("correct_letter")
        r["subdomain"] = extra.get("subdomain")
        r["origin"] = extra.get("origin", "gpqa-diamond")

    logger.info("Grading %d GPQA-Diamond results deterministically (MCQ letter match)", len(results))
    results = _grade_deterministic(results)

    slug = run_slug(config, overrides)
    for r in results:
        r["dataset"] = DATASET_NAME
        r["run_slug"] = slug
        r.setdefault("reasoning_chunks", None)
        r.setdefault("answer_chunks", None)
        r.setdefault("error_step_index", None)
    return results
