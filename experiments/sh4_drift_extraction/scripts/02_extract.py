#!/usr/bin/env python
"""Stage B: LLM extraction using Claude Haiku API."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, setup_logging, ensure_dirs
from src.extraction import run_extraction


def main():
    setup_logging()
    config = load_config()
    ensure_dirs(config, project_root)
    output_path = run_extraction(config, project_root)
    print(f"Stage B complete. Output: {output_path}")


if __name__ == "__main__":
    main()
