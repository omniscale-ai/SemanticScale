#!/usr/bin/env python
"""Stage A: Sample 500 test questions and extract gold answers from QASPER."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.sample_questions import prepare_selected_questions

    prepare_selected_questions(config)


if __name__ == "__main__":
    main()
