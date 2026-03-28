#!/usr/bin/env python
"""Stage A: Download and join data.

Supports two modes:
- Abstract-only (iter 1): PwC papers + abstracts
- Full-text (iter 2): QASPER full text + PwC eval tables
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, setup_logging, ensure_dirs
from src.data_acquisition import run_acquisition, acquire_qasper_pwc_data


def main():
    setup_logging()
    config = load_config()
    ensure_dirs(config, project_root)

    # Check if fulltext mode is enabled
    fulltext_cfg = config.get("fulltext_fallback", {})
    if fulltext_cfg.get("enabled", False):
        print("Using QASPER full-text mode (iteration 2)")
        output_path = acquire_qasper_pwc_data(config, project_root)
    else:
        print("Using abstract-only mode (iteration 1)")
        output_path = run_acquisition(config, project_root)

    print(f"Stage A complete. Output: {output_path}")


if __name__ == "__main__":
    main()
