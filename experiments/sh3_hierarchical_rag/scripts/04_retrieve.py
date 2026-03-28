#!/usr/bin/env python
"""Step 4: Run retrieval for all 4 conditions at all k values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.retrieve import run_all_retrieval


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    run_all_retrieval(config, project_root=project_root)


if __name__ == "__main__":
    main()
