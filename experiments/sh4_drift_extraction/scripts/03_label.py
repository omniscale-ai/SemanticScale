#!/usr/bin/env python
"""Stage C: Silver label matching against PwC ground truth."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, setup_logging, ensure_dirs
from src.labeling import run_labeling


def main():
    setup_logging()
    config = load_config()
    ensure_dirs(config, project_root)
    output_path = run_labeling(config, project_root)
    print(f"Stage C complete. Output: {output_path}")


if __name__ == "__main__":
    main()
