#!/usr/bin/env python
"""Stage D: Feature engineering (SLoD drift + surface features)."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, setup_logging, ensure_dirs
from src.feature_engineering import run_feature_engineering


def main():
    setup_logging()
    config = load_config()
    ensure_dirs(config, project_root)
    output_path = run_feature_engineering(config, project_root)
    print(f"Stage D complete. Output: {output_path}")


if __name__ == "__main__":
    main()
