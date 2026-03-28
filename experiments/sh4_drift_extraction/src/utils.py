"""Shared utilities for SH4: config loading, logging, I/O, text normalization."""

import json
import logging
import re
import unicodedata
from pathlib import Path

import jsonlines
import yaml


def load_config(path: str | Path = None) -> dict:
    """Load YAML configuration file.

    If path is None, looks for config.yaml in project root.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def ensure_dirs(config: dict, project_root: Path = None) -> None:
    """Create all output directories specified in config."""
    if project_root is None:
        project_root = get_project_root()
    data_dir = project_root / config["data_dir"]
    for subdir in ["raw", "extractions", "features", "models", "results"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    reports_dir = project_root / config["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)


# --- JSONL I/O ---

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return list of dicts."""
    records = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            records.append(obj)
    logging.info(f"Loaded {len(records)} records from {path}")
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in records:
            writer.write(record)
    logging.info(f"Wrote {len(records)} records to {path}")


def append_jsonl(records: list[dict], path: str | Path) -> None:
    """Append list of dicts to an existing JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="a") as writer:
        for record in records:
            writer.write(record)


# --- JSON I/O ---

def read_json(path: str | Path) -> dict | list:
    """Read a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: dict | list, path: str | Path) -> None:
    """Write data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logging.info(f"Wrote JSON to {path}")


# --- Text normalization ---

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace, NFKD."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_arxiv_id(url: str) -> str | None:
    """Extract arxiv ID from a URL or string.

    Handles formats like:
    - https://arxiv.org/abs/2301.12345
    - https://arxiv.org/pdf/2301.12345v2
    - arxiv:2301.12345
    - http://arxiv.org/abs/2301.12345v1
    """
    if not url or not isinstance(url, str):
        return None
    # Match arxiv ID patterns
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv:(\d{4}\.\d{4,5}(?:v\d+)?)",
        # Older format: e.g., hep-ph/9901234
        r"arxiv\.org/(?:abs|pdf)/([\w-]+/\d{7}(?:v\d+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            # Strip version suffix for matching
            aid = re.sub(r"v\d+$", "", m.group(1))
            return aid
    return None


def parse_numeric_value(s: str) -> float | None:
    """Try to parse a numeric value from a string.

    Handles percentages, commas, etc.
    """
    if s is None:
        return None
    s = str(s).strip()
    # Remove percentage sign
    s = s.rstrip("%")
    # Remove commas
    s = s.replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
