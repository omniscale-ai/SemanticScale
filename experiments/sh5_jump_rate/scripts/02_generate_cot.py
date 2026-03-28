#!/usr/bin/env python
"""Stage B: Generate CoT reasoning traces via Claude Haiku for all conditions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.generate_cot import generate_all_cot

    generate_all_cot(config)


if __name__ == "__main__":
    main()
