#!/usr/bin/env python
"""Stage G: Generate all visualizations and the final report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.visualize import generate_all_visualizations

    generate_all_visualizations(config)


if __name__ == "__main__":
    main()
