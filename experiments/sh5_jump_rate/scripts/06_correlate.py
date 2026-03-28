#!/usr/bin/env python
"""Stage F: Correlation analysis, condition comparisons, regression."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.correlate import analyze_all

    analyze_all(config)


if __name__ == "__main__":
    main()
