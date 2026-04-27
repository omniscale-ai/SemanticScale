# SH6 gpqa-diamond/deepseek/deepseek-v3.2_reasoning-auto — Advanced Failure Prediction

## Setup

- Items: 192 (positive=147, negative=45)
- Cross-validation: stratified 5-fold, random_state=42
- Length features (3): reasoning_n_chunks, answer_n_chunks, total_n_chunks
- Shape features: 60

## Results

| Model | Features | # Features | ROC-AUC (mean ± std) |
|---|---|---|---|
| logreg | length only | 3 | 0.761 ± 0.051 |
| logreg | length + shape | 63 | 0.702 ± 0.108 |
| logreg | length-residualized shape (length dropped) | 60 | 0.593 ± 0.089 |
| gbm (HGB) | length only | 3 | 0.681 ± 0.068 |
| gbm (HGB) | length + shape | 63 | 0.689 ± 0.039 |
| gbm (HGB) | length + shape + subject | 64 | 0.704 ± 0.030 |

## Interpretation

- **Length baseline (logreg)**: AUC 0.761. This is the bar.
- **Length-residualized shape**: AUC 0.593. Shape carries Δ = +0.093 above chance after removing length.
- **Gradient boosting (length + shape)**: AUC 0.689 vs logreg 0.702. Non-linearity does not lift the model meaningfully on this run.
- **GBM + subject**: AUC 0.704. Δ vs GBM-without-subject = +0.014: domain stratification helps.

## Caveats

- Residualization is performed per fold to avoid leakage; on small samples, fold-to-fold residualizer fits add variance, so the residualized AUC has slightly wider standard deviation than a naive whole-dataset residualization would.
- HistGradientBoostingClassifier uses native NaN handling, no imputation is applied to GBM inputs.
