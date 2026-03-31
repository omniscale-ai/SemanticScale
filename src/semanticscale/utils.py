"""Shared utilities for the semanticscale package."""

import json
import logging
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and inject _project_root for path resolution."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_project_root"] = str(path.parent)
    return config


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a timestamped format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSON Lines file and return a list of records."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """Write records to a JSON Lines file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in data)
