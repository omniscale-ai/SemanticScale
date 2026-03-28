# SH5a -- Transition Matrix Predicts Reasoning Quality

SH5 found that scalar jump rate does not correlate with answer token-F1. SH5a decomposes the per-trace SLoD level sequence into a 3x3 transition matrix to test whether specific transition patterns (e.g., macro-to-micro shuttling, self-loop dominance) capture signal that scalar metrics missed.

## Quick Start

```bash
# Ensure data symlinks are in place:
#   data_sh5/ -> <SLOD_DATA_ROOT>/SH5
# Run pipeline stages in order:
python scripts/01_load_data.py
python scripts/02_build_matrices.py
python scripts/03_extract_features.py
python scripts/04_analyze.py
python scripts/05_visualize.py
python scripts/06_generate_report.py
```

## Key Results

See `reports/SH5a_auto_report.md` and `reports/SH5a_CONCLUSIVE_REPORT.md` for full analysis.
