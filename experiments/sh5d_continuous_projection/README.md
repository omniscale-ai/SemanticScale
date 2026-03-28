# SH5d -- Probe-Free Embedding Distance as Behavioral Signature

Tests whether continuous embedding-space metrics (cosine distances, SLoD axis projections) capture reasoning coherence better than discrete SLoD labels from a noisy probe. The SLoD axis (macro-to-micro direction in embedding space) is hypothesized to explain more variance in answer quality than orthogonal directions.

## Quick Start

```bash
# Ensure data symlinks are in place:
#   data_sh1/ -> <SLOD_DATA_ROOT>/SH1
#   data_sh3/ -> <SLOD_DATA_ROOT>/SH3
#   data_sh5/ -> <SLOD_DATA_ROOT>/SH5
# Run pipeline stages in order:
python scripts/01_load_data.py
python scripts/02_embed_steps.py
python scripts/03_slod_axis.py
python scripts/04_features.py
python scripts/05_analysis.py
python scripts/06_report.py
```

## Key Results

See `reports/SH5d_results.md` and `reports/SH5d_CONCLUSIVE_REPORT.md` for full analysis.
