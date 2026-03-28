#!/usr/bin/env python
"""Step 3: Classify query SLoD level using SciBERT+LogReg probe retrained from SH1."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.query_classifier import classify_queries


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    classify_queries(config, project_root=project_root)


if __name__ == "__main__":
    main()
