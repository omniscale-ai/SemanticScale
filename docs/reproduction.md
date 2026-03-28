# Reproduction Guide

This guide explains how to reproduce each experiment end-to-end.

## Prerequisites

- Python 3.10+
- ~4 GB disk space for data
- GPU recommended for SH2 (activation steering on Mistral-7B)
- API access to Claude (for SH5 CoT generation) or use provided traces

```bash
git clone git@github.com:omniscale-ai/SemanticScale.git
cd SemanticScale
pip install -r requirements.txt
python setup_data.py  # downloads all data (~3.4 GB)
```

## Environment Setup

Set `SLOD_DATA_ROOT` to point to your data directory (defaults to `<repo>/data`):

```bash
export SLOD_DATA_ROOT=/path/to/SemanticScale/data
```

## Experiment Execution Order

Experiments have dependencies — run them in this order:

### 1. SH0 — Weak Label Bootstrap

```bash
cd experiments/sh0_weak_labels
python scripts/run_labeling.py      # Generate SLoD-labeled spans from QASPER
python scripts/run_validation.py    # Validate label quality
```

**Output:** `data/sh0/qasper_slod_spans.jsonl`, `data/sh0/qasper_slod_length_matched.jsonl`

### 2. SH1 — Linear Probe

```bash
cd experiments/sh1_linear_probe
python scripts/run_sh1.py           # End-to-end: embed → train → evaluate
```

**Variants:**
- `run_sh1b.py` — Paper-grouped splits (no data leakage between train/test)
- `run_sh1c.py` — Section-controlled analysis
- `run_sh1_llm.py` — LLM-based label rerun

**Output:** Probe models, confusion matrices, t-SNE plots in `reports/`

### 3. SH2 — Activation Steering (requires GPU)

```bash
cd experiments/sh2_activation_steering
python scripts/01_setup_environment.py
python scripts/02_compute_slod_axis.py
python scripts/03_compute_steering_vector.py
python scripts/04_generate_baseline.py
python scripts/05_generate_steered.py
python scripts/06_evaluate.py
python scripts/07_report.py
```

**Summarization variant (the confirmed result):**
```bash
python scripts/00_prepare_summarization_data.py
python scripts/03c_compute_steering_vector_summ.py
python scripts/04d_generate_summaries.py
python scripts/06d_evaluate_summaries.py
python scripts/07c_report_summaries.py
```

### 4. SH3 — Hierarchical RAG

```bash
cd experiments/sh3_hierarchical_rag
python scripts/01_prepare_data.py
python scripts/02_embed.py          # or 02b_embed_specter2.py
python scripts/03_build_index.py    # or 03b_train_question_probe.py
python scripts/04_retrieve.py
python scripts/05_evaluate.py
python scripts/06_analyze.py
```

### 5. SH4 — Drift Extraction

```bash
cd experiments/sh4_drift_extraction
python scripts/01_acquire_data.py
python scripts/02_extract.py
python scripts/03_label.py
python scripts/04_features.py
python scripts/05_train.py
python scripts/06_analyze.py
```

### 6. SH5 — Jump Rate (requires Claude API)

```bash
cd experiments/sh5_jump_rate
python scripts/01_sample_questions.py
python scripts/02_generate_cot.py    # Requires ANTHROPIC_API_KEY
python scripts/03_tag_slod.py
python scripts/04_compute_jumps.py
python scripts/05_score_answers.py
python scripts/06_correlate.py
python scripts/07_visualize.py
```

### 7. SH5a — Transition Matrix (reanalysis of SH5 data)

```bash
cd experiments/sh5a_transition_matrix
python scripts/01_load_data.py
python scripts/02_build_matrices.py
python scripts/03_extract_features.py
python scripts/04_analyze.py
python scripts/05_visualize.py
python scripts/06_generate_report.py
```

### 8. SH5c — Context Alignment

```bash
cd experiments/sh5c_context_alignment
python scripts/01_load_and_extract.py
python scripts/02_compute_alignment.py
python scripts/03_correlate.py
python scripts/04_subgroup_analysis.py
python scripts/05_visualize.py
python scripts/06_generate_report.py
```

### 9. SH5d — Continuous Projection

```bash
cd experiments/sh5d_continuous_projection
python scripts/01_load_data.py
python scripts/02_embed_steps.py
python scripts/03_slod_axis.py
python scripts/04_features.py
python scripts/05_analysis.py
python scripts/06_report.py
```

## Verification

After running all experiments, check that key results match:

| Experiment | Metric | Expected Value |
|---|---|---|
| SH1 | SciBERT macro-F1 | ~0.72 |
| SH2-summ | Cohen's d | ~0.679 |
| SH3 | Soft F1 gain (slod_weighted_parent) | ~+0.031 |
| SH5 | Jump rate ↔ token-F1 | ρ ≈ 0.003 (null) |
| SH5a | Macro self-loop ↔ attr-F1 | ρ ≈ −0.197 |
| SH5d | SLoD axis mean ↔ attr-F1 | ρ ≈ +0.219 |

Note: exact values may vary slightly due to random seeds and API model versions.
