"""Utility functions for SH5: config, logging, I/O, text normalization."""

import json
import logging
import re
import string
from pathlib import Path

import yaml

# Constants
SLOD_LEVELS = ["macro", "meso", "micro"]
SLOD_MAP = {0: "macro", 1: "meso", 2: "micro"}
SLOD_REVERSE = {"macro": 0, "meso": 1, "micro": 2}

# Common English stopwords (minimal set for token-F1)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "me", "us", "him", "her", "them", "my", "our", "your", "his", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "nor", "as", "if", "than", "so", "very", "just",
    "about", "into", "over", "after", "before", "between", "under",
    "again", "then", "once", "here", "there", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "also", "too",
}


def load_config(path: str | Path) -> dict:
    """Load YAML config and return dict."""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    # Store project root for path resolution
    config["_project_root"] = str(path.parent)
    return config


def resolve_path(config: dict, *parts: str) -> Path:
    """Resolve a path relative to the project root."""
    root = Path(config["_project_root"])
    return root.joinpath(*parts)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_f1(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text for token-F1 computation.

    Split on whitespace and punctuation, lowercase, optionally remove stopwords.
    """
    text = normalize_text(text)
    # Split on whitespace and punctuation
    tokens = re.findall(r"\w+", text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def token_f1(pred_tokens: list[str], gold_tokens: list[str]) -> dict:
    """Compute token-level precision, recall, F1 using multiset (Counter) overlap.

    This follows the SQuAD-style token-F1 computation where duplicate tokens
    are counted with their frequency.

    Returns dict with 'precision', 'recall', 'f1'.
    """
    from collections import Counter

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    # Multiset intersection: min of counts for each token
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / sum(pred_counter.values()) if pred_counter else 0.0
    recall = overlap / sum(gold_counter.values()) if gold_counter else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
