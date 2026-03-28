# SH1 -- Linear Decodability of SLoD from Frozen Embeddings: Design

## Hypothesis

A linear probe on frozen transformer embeddings classifies macro/meso/micro SLoD levels
above chance, with macro-F1 > 0.60 on 3-way classification.

## Method

### Data

- **Primary:** Length-matched subset from SH0 (37,278 spans, 12,426 per class)
- **Full:** Imbalanced SH0 dataset (83,135 spans) for confound checking
- Split: 70% train / 15% validation / 15% test, stratified by label

### Embedding Models

| Model | HuggingFace ID | Dim | Pooling |
|-------|---------------|-----|---------|
| SciBERT | `allenai/scibert_scivocab_uncased` | 768 | [CLS] last hidden |
| Specter2 | `allenai/specter2_base` + classification adapter | 768 | [CLS] with adapter |
| MiniLM-L6 | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean pooling |

All embeddings extracted with `torch.no_grad()` on frozen models. No fine-tuning.

### Classifiers

1. **Logistic Regression** -- `solver='lbfgs'`, `multi_class='multinomial'`, C sweep [0.01, 0.1, 1.0, 10.0]
2. **Linear SVM** -- `LinearSVC`, C sweep [0.01, 0.1, 1.0, 10.0]

All embeddings pass through `StandardScaler` (fit on train only).

### Baselines

- Random baseline (expected macro-F1 = 0.333)
- Word-count-only baseline (measures length confound residual)
- Random-embedding baseline (confirms probe uses learned structure)

### Confound Checks

- Compare length-matched vs full dataset performance
- If gap > 0.10, length is a major confound
- Word-count-only baseline quantifies length leakage

## Analysis / Exit Criteria

| Criterion | Threshold |
|---|---|
| 3-way macro-F1, length-matched test | > 0.60 |
| Binary macro-vs-micro F1 (fallback) | > 0.75 |
| Best model minus random baseline | > 0.15 |
| Length-matched F1 within 0.10 of full F1 | Yes |

## Visualization

- Confusion matrices per model/classifier
- t-SNE of embeddings colored by SLoD label
- PCA explained variance curves
- Confidence calibration (SH0 confidence vs probe accuracy)
