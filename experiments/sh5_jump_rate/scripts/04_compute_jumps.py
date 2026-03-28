#!/usr/bin/env python
"""Stage D: Compute jump metrics from SLoD-tagged CoT traces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.compute_jumps import process_all_traces

    process_all_traces(config)


if __name__ == "__main__":
    main()
