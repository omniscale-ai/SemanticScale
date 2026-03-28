# SH4 Design: Abstraction Drift Predicts TKH Extraction Quality

## Hypothesis

Drift between expected and realized SLoD per extraction field, combined with entity/citation/temporal density features, predicts extraction correctness and enables automatic precision@k filtering.

## Method

### Pipeline Overview

```
Papers with Code (silver labels)
        |
        v
Match to paper abstracts (J0nasW/paperswithcode)
        |
        v
LLM extraction (Claude Haiku) -> (Method, Dataset, Metric, Value, Year, source_span)
        |
        v
Silver label matching (fuzzy match against PwC ground truth) -> correct={0,1}
        |
        v
SH1 probe classifies each source_span -> realized SLoD
        |
        v
Feature engineering: |expected_SLoD - realized_SLoD| + surface features
        |
        v
LogReg / (optional XGBoost) -> predict correct/incorrect
        |
        v
AUROC, precision@k, ablation
```

### Stage A: Data Acquisition

- **Papers with Code evaluation tables** (`felixleungsc/paperswithcode-data-evaluation-tables`): 326K rows of (task, dataset, model, paper_url, metric, value).
- **Paper abstracts** (`J0nasW/paperswithcode`): 55K papers with arxiv_id, title, abstract.
- Filter to papers with arxiv IDs and >= 3 tuples, sample 300 papers.
- Full text fallback if < 40% of extractions match ground truth.

### Stage B: LLM Extraction

- Claude Haiku extracts (Method, Dataset, Metric, Value, Year, source_span, confidence) from title + abstract.
- Rate limited, checkpointed, ~$0.50 total cost.

### Stage C: Silver Label Generation

- Fuzzy match each extraction against PwC ground truth for same paper.
- Match on (Dataset, Metric, Value) triple: Levenshtein >= 0.8, numeric values within 0.5% tolerance.
- Expected: ~40-60% correct.

### Stage D: Feature Engineering

**SLoD Drift Features (core hypothesis):**
- Embed source_span with SciBERT, classify with SH1 LogReg probe.
- Expected SLoD per field: Method=meso(1), Dataset=meso(1), Metric=meso(1), Value=micro(2), Year=micro(2).
- Drift = |expected - realized|.
- Features: drift_mean, drift_max, drift_value, drift_method, realized_slod_raw, probe probabilities (soft signals), slod_entropy.

**Surface Features:**
- word_count, sentence_count, entity_density, citation_density, temporal_density, numeric_density, has_table_context.

**Total: ~15-20 features.**

### Stage E: Prediction Model

- Paper-level train/val/test split (70/15/15), stratified by label.
- Primary: Logistic Regression with StandardScaler, C sweep.
- Secondary: Gradient Boosted Trees (fallback if LogReg AUROC < 0.65).

**Ablation study (critical):**
1. Drift-only features
2. Surface-only features + confidence
3. Combined (all features)

## Analysis

### Metrics

- **AUROC** on test set (primary).
- **Precision@k:** Sort by P(correct), retain top k%, measure actual correctness (k = 25%, 50%, 75%).
- **Feature importance:** LogReg coefficients (standardized).
- **Calibration:** Brier score.

### Figures

1. AUROC curves per ablation group
2. Precision@k curve
3. Feature importance bar chart
4. Drift distribution for correct vs incorrect extractions
5. Confusion matrix at optimal threshold

### Exit Criteria

| Verdict | Condition |
|---|---|
| **CONFIRMED** | Combined AUROC >= 0.65, drift-only > surface-only by >= 0.03, precision@50% >= 0.70 |
| **PARTIAL** | Combined AUROC >= 0.60, drift features have nonzero importance |
| **NOT CONFIRMED** | Combined AUROC < 0.60 OR drift features show zero/negative contribution |

## Code Structure

```
src/
  utils.py             -- Shared utilities, config loading
  data_acquisition.py  -- HuggingFace data loading + joining (was data_loader.py)
  extraction.py        -- LLM extraction pipeline (Anthropic API)
  labeling.py          -- Silver label matching logic
  feature_engineering.py -- Feature computation (drift + surface)
  slod_probe.py        -- SH1 probe wrapper (SciBERT + LogReg)
  model_training.py    -- LogReg / GBT training + evaluation
  analysis.py          -- Visualization + report generation

scripts/
  01_acquire_data.py
  02_extract.py (was 02_extract_tuples.py)
  03_label.py (was 03_label_extractions.py)
  04_features.py (was 04_engineer_features.py)
  05_train.py (was 05_train_model.py)
  06_analyze.py
```

## Data Dependencies

- PwC evaluation tables: HuggingFace
- PwC paper metadata: HuggingFace
- SH1 embeddings: `data_sh1/embeddings/scibert_length_matched.npz`
- SH1 splits: `data_sh1/splits.json`
- SH0 spans: `data_sh0/qasper_slod_length_matched.jsonl`

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| PwC dataset incomplete/noisy | Pre-filter aggressively; check match rates early |
| Abstracts too short | Stage A3 fallback to full text |
| SH1 probe doesn't generalize to extraction spans | Check probe confidence; use soft probabilities |
| Too few matched papers | Check join cardinality early; arxiv API fallback |
