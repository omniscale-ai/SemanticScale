"""Load the openai/frontierscience dataset."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_final_answer_prompt(problem: str) -> bool:
    return "FINAL ANSWER" in problem


def load_frontierscience(
    hf_path: str = "openai/frontierscience",
    split: str = "test",
    max_samples: int | None = None,
) -> list[dict]:
    """Load the FrontierScience dataset and return records.

    Each record has keys: id, problem, correct_answer, subject, has_final_answer.
    has_final_answer is True when the problem prompt requires a FINAL ANSWER line.
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

    rows = [dict(row) for row in ds]

    if max_samples is not None:
        rows = rows[:max_samples]
        logger.info("Truncated to %d samples", len(rows))

    items = [
        {
            "id": str(row["task_group_id"]),
            "problem": str(row["problem"]),
            "correct_answer": str(row["answer"]),
            "subject": str(row["subject"]),
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
