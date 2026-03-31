"""Load the openai/frontierscience dataset."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ALL_TYPES = ("olympiad", "research")


def _has_final_answer_prompt(problem: str) -> bool:
    return "FINAL ANSWER" in problem


def _load_rows_for_type(snapshot_path: Path, qtype: str, split: str) -> list[dict]:
    """Load rows from a single question-type subdirectory and inject 'origin'."""
    from datasets import load_dataset  # deferred import — heavy dependency

    data_file = snapshot_path / qtype / f"{split}.jsonl"
    if not data_file.exists():
        logger.warning("No file found for type=%s split=%s at %s", qtype, split, data_file)
        return []
    ds = load_dataset("json", data_files=str(data_file), trust_remote_code=False)
    rows = [dict(row) | {"origin": qtype} for row in ds["train"]]
    logger.info("Loaded %d rows for type=%s", len(rows), qtype)
    return rows


def load_frontierscience(
    hf_path: str = "openai/frontierscience",
    split: str = "test",
    max_samples: int | None = None,
    question_types: list[str] | None = None,
) -> list[dict]:
    """Load the FrontierScience dataset and return records.

    Each record has keys: id, problem, correct_answer, subject, origin,
    has_final_answer.  has_final_answer is True when the problem prompt
    requires a FINAL ANSWER line.

    Args:
        question_types: If given, load only the specified question type
            subdirectories (case-insensitive), e.g. ``["olympiad"]`` or
            ``["research"]``.  Valid values are ``"olympiad"`` and
            ``"research"``.  Pass ``None`` (default) to load all types.
    """
    from huggingface_hub import snapshot_download

    logger.info("Downloading dataset %s from the Hugging Face Hub", hf_path)
    snapshot_path = Path(
        snapshot_download(
            repo_id=hf_path,
            repo_type="dataset",
        )
    )
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
        len(items),
        hf_path,
        split,
        n_final_answer,
        len(items) - n_final_answer,
    )
    return items
