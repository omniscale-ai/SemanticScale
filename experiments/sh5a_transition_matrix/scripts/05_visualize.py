#!/usr/bin/env python
"""Script 05: Generate all visualizations."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualize import generate_all_plots

if __name__ == "__main__":
    generate_all_plots()
    print("\nStage E (visualization) complete.")
