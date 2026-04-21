"""FrontierScience dataset loader.

Stage 1 for FrontierScience: download problems, run LLM inference, grade
with a second LLM call, and return a list of trace records.
"""

from __future__ import annotations

import logging
from pathlib import Path

from semanticscale.sh6.grading import grade_results
from semanticscale.sh6.inference import make_model_slug, run_inference

logger = logging.getLogger(__name__)

_ALL_TYPES = ("olympiad", "research")

DATASET_NAME = "frontierscience"


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a FrontierScience run: model, reasoning effort, types."""
    traces_cfg = config["traces"]
    model = overrides.get("model") or traces_cfg["model"]["name"]
    reasoning = traces_cfg["model"].get("reasoning", {})
    question_types = overrides.get("question_types")
    return make_model_slug(model, reasoning, question_types)


def _has_final_answer_prompt(problem: str) -> bool:
    return "FINAL ANSWER" in problem


def _load_rows_for_type(snapshot_path: Path, qtype: str, split: str) -> list[dict]:
    """Load rows from a single question-type subdirectory and inject 'origin'."""
    from datasets import load_dataset  # deferred: heavy dep

    data_file = snapshot_path / qtype / f"{split}.jsonl"
    if not data_file.exists():
        logger.warning("No file found for type=%s split=%s at %s", qtype, split, data_file)
        return []
    ds = load_dataset("json", data_files=str(data_file), trust_remote_code=False)
    rows = [dict(row) | {"origin": qtype} for row in ds["train"]]
    logger.info("Loaded %d rows for type=%s", len(rows), qtype)
    return rows


def _load_problems(
    hf_path: str,
    split: str,
    max_samples: int | None,
    question_types: list[str] | None,
) -> list[dict]:
    """Download the FrontierScience dataset and return one dict per problem."""
    from huggingface_hub import snapshot_download

    logger.info("Downloading dataset %s from the Hugging Face Hub", hf_path)
    snapshot_path = Path(snapshot_download(repo_id=hf_path, repo_type="dataset"))
    logger.info("Loading dataset %s (split=%s) from %s", hf_path, split, snapshot_path)

    types_to_load = (
        [qt.lower() for qt in question_types]
        if question_types is not None
        else list(_ALL_TYPES)
    )

    rows: list[dict] = []
    for qtype in types_to_load:
        rows.extend(_load_rows_for_type(snapshot_path, qtype, split))

    if max_samples is not None:
        rows = rows[:max_samples]
        logger.info("Truncated to %d samples", len(rows))

    items = [
        {
            "id": str(row["task_group_id"]),
            "problem": str(row["problem"]),
            "correct_answer": str(row["answer"]),
            "subject": str(row["subject"]),
            "origin": str(row["origin"]),
            "has_final_answer": _has_final_answer_prompt(str(row["problem"])),
        }
        for row in rows
    ]

    n_final_answer = sum(1 for it in items if it["has_final_answer"])
    logger.info(
        "Loaded %d items from %s/%s (%d with FINAL ANSWER, %d without)",
        len(items), hf_path, split, n_final_answer, len(items) - n_final_answer,
    )
    return items


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict,
) -> list[dict]:
    """Stage 1: load problems, run inference, grade, and normalise to traces."""
    ds_cfg = config["dataset"]
    max_samples = overrides.get("max_samples")
    if max_samples is None:
        max_samples = ds_cfg.get("max_samples")
    question_types = overrides.get("question_types") or ds_cfg.get("question_types")

    items = _load_problems(
        hf_path=ds_cfg["hf_path"],
        split=ds_cfg.get("split", "test"),
        max_samples=max_samples,
        question_types=question_types,
    )

    results = run_inference(
        items=items,
        config=config,
        model_override=overrides.get("model"),
        service_tier_override=overrides.get("service_tier"),
    )

    grader_model = config["pairwise_slod"]["model"]["name"]
    logger.info("Running advanced grading with %s", grader_model)
    results = grade_results(results, config=config)

    slug = run_slug(config, overrides)
    for r in results:
        r["dataset"] = DATASET_NAME
        r["run_slug"] = slug
        # Fields that FrontierScience doesn't populate but ProcessBench does —
        # keep them explicit so consumers can rely on the unified schema.
        r.setdefault("reasoning_chunks", None)
        r.setdefault("answer_chunks", None)
        r.setdefault("error_step_index", None)
    return results
