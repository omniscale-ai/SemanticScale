#!/usr/bin/env python
"""Stage F: Analysis, visualization, and report generation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, setup_logging, ensure_dirs
from src.analysis import run_analysis


def main():
    setup_logging()
    config = load_config()
    ensure_dirs(config, project_root)
    output_path = run_analysis(config, project_root)
    print(f"Stage F complete. Report: {output_path}")


if __name__ == "__main__":
    main()
