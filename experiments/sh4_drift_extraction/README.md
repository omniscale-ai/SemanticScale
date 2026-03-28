# SH4: Abstraction Drift Predicts TKH Extraction Quality

Validation of Sub-Hypothesis 4: SLoD drift between expected and realized abstraction level per extraction field predicts extraction correctness.

## Overview

This experiment tests whether the mismatch between expected and actual SLoD of LLM-extracted knowledge tuples can predict extraction quality. It uses Papers with Code silver labels, Claude Haiku for extraction, and the SH1 SciBERT probe for SLoD classification.

## Key Results

See `reports/SH4_CONCLUSIVE_REPORT.md` for the full analysis.

## Running

```bash
python scripts/01_acquire_data.py
python scripts/02_extract.py          # Requires ANTHROPIC_API_KEY
python scripts/03_label.py
python scripts/04_features.py
python scripts/05_train.py
python scripts/06_analyze.py
```

## Data Dependencies

- Papers with Code evaluation tables: HuggingFace (`felixleungsc/paperswithcode-data-evaluation-tables`)
- Papers with Code metadata: HuggingFace (`J0nasW/paperswithcode`)
- SH0 spans: `data_sh0/qasper_slod_length_matched.jsonl`
- SH1 embeddings: `data_sh1/embeddings/scibert_length_matched.npz`
- SH1 splits: `data_sh1/splits.json`

## Configuration

All parameters in `config.yaml`.
