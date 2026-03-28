# SH2 -- Activation Steering Along the SLoD Axis

Tests whether activation steering in a generative LLM can controllably shift the semantic
level of detail of generated scientific answers while preserving factual accuracy.

## Quick Start

```bash
pip install -r requirements.txt

# Run stages sequentially
python scripts/01_setup_environment.py
python scripts/02_compute_slod_axis.py
python scripts/03_compute_steering_vector.py
python scripts/04_generate_baseline.py
python scripts/05_generate_steered.py
python scripts/06_evaluate.py
python scripts/07_report.py
```

Requires GPU. Total runtime: 2-4 hours on A100/V100.

## Data

Requires SH0 and SH1 outputs. Set up data directories:
```bash
ln -s /path/to/SLoD_data/SH2 data
```

Expected inputs:
- `data/sh0/qasper_slod_length_matched.jsonl` (37,278 spans from SH0)
- `data/sh1/embeddings/scibert_length_matched.npz` (SH1 embeddings)
- `data/sh1/splits.json` (SH1 train/val/test splits)
- `data/sh5/selected_questions.jsonl` (500 QASPER questions)

## Architecture

```
src/
  embedding.py      -- SciBERT [CLS] embedding
  slod_axis.py      -- SLoD axis computation + validation
  steering.py       -- Activation steering hooks + vector computation
  evaluate.py       -- SLoD shift, surface metrics, token-F1
  visualization.py  -- All plotting functions
  utils.py          -- Config, I/O, text normalization

scripts/
  01_setup_environment.py          -- GPU check, model download, data validation
  02_compute_slod_axis.py          -- SLoD evaluation axis from SH1 embeddings
  03_compute_steering_vector.py    -- Per-layer steering vectors in generative model
  04_generate_baseline.py          -- Baseline answers (no steering)
  05_generate_steered.py           -- Steered answers (both directions)
  06_evaluate.py                   -- Full evaluation pipeline
  07_report.py                     -- Figures and markdown report
  Additional variants: 03c, 04b, 04d, 06b, 06d, 06e, 07c (summarization experiments)

reports/
  SH2_CONCLUSIVE_REPORT.md        -- Final results and verdict
  figures/                         -- All generated plots
```

## Configuration

All parameters in `config.yaml`: generative model, steering alpha range, layer selection,
evaluation thresholds, SciBERT settings.

## Additional Files

- `RESULTS_IMPORT_SPEC.md` -- Documents the expected data format for importing results
  from remote GPU execution environments.

## Key Results

See `reports/SH2_CONCLUSIVE_REPORT.md` for full results.
