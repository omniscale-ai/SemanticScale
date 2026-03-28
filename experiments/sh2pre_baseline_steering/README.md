# SH2-pre: Baseline Activation Steering Along the SLoD Axis

Validation of Sub-Hypothesis 2 (preliminary): activation steering in a generative LLM's residual stream can controllably shift the Semantic Level of Detail (SLoD) of generated answers.

## Overview

This experiment computes steering vectors from macro/micro span centroids in a generative model's hidden states, then applies them during generation to shift answers toward more abstract (macro) or more detailed (micro) levels of description.

## Key Results

See `reports/` for figures and analysis.

## Running

```bash
# Stages run sequentially; each is idempotent
python scripts/01_setup_environment.py
python scripts/02_compute_slod_axis.py
python scripts/03_compute_steering_vector.py
python scripts/04_generate_baseline.py
python scripts/05_generate_steered.py
python scripts/06_evaluate.py
python scripts/07_report.py
```

## Data Dependencies

- SH0 spans: `data/sh0/qasper_slod_length_matched.jsonl`
- SH1 embeddings: `data/sh1/embeddings/scibert_length_matched.npz`
- SH1 splits: `data/sh1/splits.json`
- SH5 selected questions: `data/sh5/selected_questions.jsonl`

## Configuration

All parameters in `config.yaml`. Key settings:
- `generative_model.name`: Mistral-7B-Instruct (default)
- `steering.layer_fractions`: which layers to test
- `steering.alpha_values`: steering strength sweep
