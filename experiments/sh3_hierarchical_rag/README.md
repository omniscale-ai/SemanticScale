# SH3: SLoD-Routed Hierarchical RAG

Validation of Sub-Hypothesis 3: routing retrieval to the granularity level matching query SLoD improves evidence attribution on QASPER.

## Overview

This experiment builds a retrieval system with three document index levels (macro / meso / micro) and uses a trained SLoD classifier to route each query to the appropriate level. Performance is compared against fixed-level baselines and naive hybrid retrieval.

## Key Results

See `reports/SH3_CONCLUSIVE_REPORT.md` for the full analysis.

## Running

```bash
python scripts/01_prepare_data.py
python scripts/02_embed.py
python scripts/03_classify_queries.py
python scripts/04_retrieve.py
python scripts/05_evaluate.py
python scripts/06_analyze.py
```

## Data Dependencies

- QASPER: HuggingFace cache (`allenai/qasper`)
- SH0 spans: `data_sh0/qasper_slod_length_matched.jsonl`
- SH1 embeddings: `data_sh1/embeddings/scibert_length_matched.npz`
- SH1 splits: `data_sh1/splits.json`

## Configuration

All parameters in `config.yaml`. Key settings:
- `embedding_model`: all-MiniLM-L6-v2
- `query_classifier_model`: allenai/scibert_scivocab_uncased
- `top_k_values`: [1, 3, 5, 10, 20]
- `matching_threshold`: 0.5
