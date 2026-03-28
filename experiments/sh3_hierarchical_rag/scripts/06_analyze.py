#!/usr/bin/env python
"""Step 6: Run breakdown analyses, generate plots, and produce markdown report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.analyze import run_analysis


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    run_analysis(config, project_root=project_root)


if __name__ == "__main__":
    main()
