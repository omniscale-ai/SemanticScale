#!/usr/bin/env python
"""Step 1: Load QASPER, build 3-level index, extract questions with gold evidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.data_prep import prepare_all


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    prepare_all(config, project_root=project_root)


if __name__ == "__main__":
    main()
