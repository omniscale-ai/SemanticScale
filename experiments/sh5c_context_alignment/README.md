# SH5c -- Per-Question SLoD Consistency Predicts Quality

Tests whether context-reasoning SLoD alignment predicts answer quality: when the retrieval context's SLoD distribution matches the reasoning chain's SLoD distribution, answers should be better. SLoD-routed retrieval conditions are expected to produce better alignment than unrouted conditions.

## Quick Start

```bash
# Ensure data symlinks are in place:
#   data_sh5/ -> <SLOD_DATA_ROOT>/SH5
#   data_sh3/ -> <SLOD_DATA_ROOT>/SH3
# Run pipeline stages in order:
python scripts/01_load_and_extract.py
python scripts/02_alignment_features.py
python scripts/03_statistical_analysis.py
python scripts/04_subgroup_analysis.py
python scripts/05_visualize.py
python scripts/06_generate_report.py
```

## Key Results

See `reports/SH5c_auto_report.md` and `reports/SH5c_CONCLUSIVE_REPORT.md` for full analysis.
