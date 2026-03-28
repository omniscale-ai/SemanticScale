#!/usr/bin/env python
"""Stage E: Score CoT final answers against gold (token-F1 + attribution F1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.score_answers import score_all

    score_all(config)


if __name__ == "__main__":
    main()
