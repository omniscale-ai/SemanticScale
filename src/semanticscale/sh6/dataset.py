"""Load and normalise the openai/frontierscience dataset."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Candidate field names for each schema slot, in preference order.
_QUESTION_FIELDS = ("problem", "question", "Question", "Problem")
_ANSWER_FIELDS = ("answer", "Answer", "correct_answer", "solution", "Solution")
_SUBJECT_FIELDS = (
    "subject", "Subject", "category", "Category", "domain", "Domain", "field"
)
_CHOICES_FIELDS = ("choices", "Choices", "options", "Options")
_ID_FIELDS = ("id", "ID", "idx", "index", "problem_id", "question_id")


def _has_final_answer_prompt(problem: str) -> bool:
    """Return True if the prompt requires a FINAL ANSWER line."""
    return "FINAL ANSWER" in problem


def _pick(row: dict, candidates: tuple[str, ...], default: Any = None) -> Any:
    for key in candidates:
        if key in row:
            return row[key]
    return default


def _normalise_row(row: dict, idx: int) -> dict:
    """Map a raw dataset row to the canonical SH6 schema."""
    problem = _pick(row, _QUESTION_FIELDS, "")
    answer = _pick(row, _ANSWER_FIELDS, "")
    subject = _pick(row, _SUBJECT_FIELDS, "unknown")
    choices = _pick(row, _CHOICES_FIELDS, None)
    item_id = _pick(row, _ID_FIELDS, str(idx))

    # Coerce choices to a list of strings when present
    if choices is not None and not isinstance(choices, list):
        choices = list(choices)

    return {
        "id": str(item_id),
        "problem": str(problem),
        "choices": choices,
        "correct_answer": str(answer),
        "subject": str(subject),
    }


def load_frontierscience(
    hf_path: str = "openai/frontierscience",
    split: str = "test",
    max_samples: int | None = None,
) -> list[dict]:
    """Load the FrontierScience dataset and return normalised records.

    Only rows whose prompt explicitly requires a `FINAL ANSWER` line are kept.
    Each record has keys: id, problem, choices, correct_answer, subject.
    choices is None for free-form questions, otherwise a list of strings.
    """
    from datasets import load_dataset  # deferred import — heavy dependency
    from huggingface_hub import snapshot_download

    logger.info("Downloading dataset %s from the Hugging Face Hub", hf_path)
    snapshot_path = Path(
        snapshot_download(
            repo_id=hf_path,
            repo_type="dataset",
        )
    )
    logger.info("Loading dataset %s (split=%s) from %s", hf_path, split, snapshot_path)
    ds = load_dataset(str(snapshot_path), split=split, trust_remote_code=False)

    logger.info("Dataset columns: %s", ds.column_names)
    rows = [dict(row) for row in ds]
    filtered_rows = [row for row in rows if _has_final_answer_prompt(str(row.get("problem", "")))]
    dropped_rows = len(rows) - len(filtered_rows)
    logger.info(
        "Filtered to %d rows with FINAL ANSWER prompts (%d dropped)",
        len(filtered_rows),
        dropped_rows,
    )

    if max_samples is not None:
        filtered_rows = filtered_rows[:max_samples]
        logger.info("Truncated to %d samples", len(filtered_rows))

    items = [_normalise_row(row, i) for i, row in enumerate(filtered_rows)]
    logger.info("Loaded %d items from %s/%s", len(items), hf_path, split)
    return items
