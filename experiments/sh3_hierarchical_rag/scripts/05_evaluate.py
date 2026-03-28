#!/usr/bin/env python
"""Step 5: Compute all evaluation metrics and run statistical significance tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.evaluate import evaluate_all


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    evaluate_all(config, project_root=project_root)


if __name__ == "__main__":
    main()
