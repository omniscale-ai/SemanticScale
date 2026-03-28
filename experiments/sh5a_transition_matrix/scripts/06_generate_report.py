#!/usr/bin/env python
"""Script 06: Generate auto-report."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report import generate_report

if __name__ == "__main__":
    generate_report()
    print("\nStage E (report) complete.")
