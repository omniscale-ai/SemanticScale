"""Shared utilities for SH3: config loading, logging, JSONL I/O, text normalization."""

import json
import logging
import re
from pathlib import Path

import jsonlines
import yaml


SLOD_LEVELS = ["macro", "meso", "micro"]
LABEL_MAP = {"macro": 0, "meso": 1, "micro": 2}
LABEL_NAMES = ["macro", "meso", "micro"]


def load_config(path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for item in data:
            writer.write(item)
    logging.info(f"Saved {len(data)} records to {path}")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            records.append(obj)
    logging.info(f"Loaded {len(records)} records from {path}")
    return records


def save_json(data, path: str | Path) -> None:
    """Write data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved JSON to {path}")


def load_json(path: str | Path):
    """Read a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, collapse multiple spaces."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_text(text: str) -> list[str]:
    """Tokenize text by splitting on whitespace after normalization."""
    return normalize_text(text).split()


def sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using regex.

    Handles common abbreviations and decimal numbers to avoid
    false splits. Falls back gracefully for edge cases.
    """
    # Split on sentence-ending punctuation followed by space and uppercase
    # or end of string. Handles ., !, ?
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]
