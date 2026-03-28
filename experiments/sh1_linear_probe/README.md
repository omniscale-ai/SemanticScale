# SH1 -- Linear Decodability of SLoD from Frozen Embeddings

Tests whether SLoD levels (macro/meso/micro) are linearly decodable from frozen
transformer embeddings using logistic regression and linear SVM probes.

## Quick Start

```bash
pip install -r requirements.txt

# End-to-end: embed + train probes + analyze
python scripts/run_sh1.py
```

## Data

Requires SH0 outputs. Set up data directories:
```bash
ln -s /path/to/SLoD_data/SH1 data
ln -s /path/to/SLoD_data/SH0 data_sh0
```

Or set `SLOD_DATA_ROOT` environment variable.

## Architecture

```
src/
  embed_spans.py   -- Embedding generation (SciBERT, Specter2, MiniLM)
  train_probe.py   -- Linear probe training + evaluation + baselines
  analyze.py       -- Visualization (confusion matrices, t-SNE, PCA)
  utils.py         -- Config loading, data I/O, splits

scripts/
  run_sh1.py       -- End-to-end orchestrator
  run_sh1b.py      -- Grouped split variant (paper-level splits)
  run_sh1c.py      -- Section-controlled variant
  run_sh1_llm.py   -- LLM-based axis rerun

reports/
  SH1_CONCLUSIVE_REPORT.md  -- Final results and analysis
  figures/                   -- Confusion matrices, t-SNE, PCA plots
```

## Configuration

All hyperparameters in `config.yaml`:
- Embedding models and parameters
- Classifier types and C sweep values
- Split ratios and random seed
- Evaluation thresholds

## Key Results

See `reports/SH1_CONCLUSIVE_REPORT.md` for full results including:
- Per-model macro-F1 scores
- Confound analysis (length-matched vs full dataset)
- Baseline comparisons
- t-SNE visualizations showing SLoD cluster separation
