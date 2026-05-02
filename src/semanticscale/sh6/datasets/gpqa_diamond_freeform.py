"""GPQA-Diamond free-form dataset loader.

Loads the same 198-question Diamond split used by :mod:`gpqa_diamond`, but
requires the model to answer with the answer text rather than a letter. Each
item is formatted as a FrontierScience/Olympiad-style free-form problem with a
``FINAL ANSWER`` directive, then graded by an LLM against the ground-truth
answer text.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from semanticscale.sh6.grading import grade_results
from semanticscale.sh6.inference import make_model_slug, run_inference

logger = logging.getLogger(__name__)

DATASET_NAME = "gpqa-diamond-freeform"
# GPQA-Diamond free-form runs are already separated by generator at Stage 1
# (one model per directory), so analysis-time slicing is a no-op.
SLICE_NAME: str | None = None


def slice_label(trace: dict) -> str | None:  # noqa: ARG001
    return None


_DEFAULT_HF_PATH = "Idavidrein/gpqa"
_DEFAULT_CONFIG = "gpqa_diamond"
_DEFAULT_SPLIT = "train"

_LETTERS = ("A", "B", "C", "D")

_PROMPT_TEMPLATE = """{question}

A) {a}
B) {b}
C) {c}
D) {d}

Reason step by step, then write your final answer on a new line after "FINAL ANSWER:".

Your final answer must be the free-form answer text itself. Do not answer with only a multiple-choice letter."""


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a GPQA-Diamond free-form run."""
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
    item_id = str(record_id) if record_id not in (None, "") else f"gpqa-freeform-{fallback_index:04d}"

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
        "correct_answer": correct,
        "subject": subject,
        "origin": "gpqa-diamond-freeform",
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
        "Loaded %d GPQA-Diamond free-form items (subjects: %s)",
        len(items),
        ", ".join(f"{k}={v}" for k, v in sorted(by_subject.items())),
    )
    return items


def produce_traces(
    config: dict,
    project_root: Path,  # noqa: ARG001
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

    # Carry over per-item extras the inference layer does not propagate.
    extras_by_id = {it["id"]: it for it in items}
    for r in results:
        extra = extras_by_id.get(r["id"], {})
        r["subdomain"] = extra.get("subdomain")
        r["origin"] = extra.get("origin", "gpqa-diamond-freeform")
        r["correct_letter"] = extra.get("correct_letter")

    grader_model = config.get("grader", config["pairwise_slod"])["model"]["name"]
    logger.info("Running free-form grading with %s", grader_model)
    results = grade_results(results, config=config)

    slug = run_slug(config, overrides)
    for r in results:
        r["dataset"] = DATASET_NAME
        r["run_slug"] = slug
        r.setdefault("reasoning_chunks", None)
        r.setdefault("answer_chunks", None)
        r.setdefault("error_step_index", None)
    return results
