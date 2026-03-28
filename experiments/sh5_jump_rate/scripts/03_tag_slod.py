#!/usr/bin/env python
"""Stage C: Retrain SH1 probe, embed CoT steps with SciBERT, classify SLoD levels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    from src.tag_slod import tag_all_steps

    tag_all_steps(config)


if __name__ == "__main__":
    main()
