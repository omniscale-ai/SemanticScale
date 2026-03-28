"""Shared utilities for I/O, logging, and text processing."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import jsonlines
import yaml


def get_project_root() -> Path:
    """Return the SLoD-SH0 project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path against the project root."""
    return get_project_root() / relative_path


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    records = []
    with jsonlines.open(path, mode="r") as reader:
        for record in reader:
            records.append(record)
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in records:
            writer.write(record)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_chars(text: str) -> int:
    """Count characters in text."""
    return len(text)


def count_sentences_simple(text: str) -> int:
    """Simple sentence count based on punctuation. Used as fallback."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
