"""Utility functions for SH2: config loading, I/O, constants."""
import json
import yaml
import re
from pathlib import Path


# SLoD label constants
LABEL_MAP = {0: "macro", 1: "meso", 2: "micro"}
LABEL_TO_INT = {"macro": 0, "meso": 1, "micro": 2}


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: str):
    """Save list of dicts to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data, path: str):
    """Save dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize text for token-F1 computation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference (SQuAD-style)."""
    from collections import Counter
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
