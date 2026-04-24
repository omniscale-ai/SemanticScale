# SH6 processbench/gsm8k — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 400
- Positive class (`final_answer_correct=true`): 200
- Negative class (`final_answer_correct=false`): 200
- Label agreement (`is_correct` vs `final_answer_correct`): 98.2% over 400 items
- Final answer correct but reasoning wrong: 7
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.544 +/- 0.080 | 0.539 +/- 0.051 | 0.520 +/- 0.074 | 0.520 +/- 0.074 | 0.579 +/- 0.063 |
| trajectory_shape | 29 | 0.504 +/- 0.050 | 0.518 +/- 0.054 | 0.498 +/- 0.044 | 0.498 +/- 0.044 | 0.529 +/- 0.036 |
| trajectory_full | 29 | 0.504 +/- 0.050 | 0.518 +/- 0.054 | 0.498 +/- 0.044 | 0.498 +/- 0.044 | 0.529 +/- 0.036 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_start | shape | 0.562 | higher -> correct |
| reasoning_negative_mass | shape | 0.558 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.552 | higher -> wrong |
| reasoning_range | commitment | 0.551 | higher -> correct |
| reasoning_n_chunks | length | 0.544 | higher -> correct |
| reasoning_monotonicity | commitment | 0.542 | higher -> wrong |
| reasoning_std | shape | 0.542 | higher -> correct |
| reasoning_end | landing | 0.542 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.540 | higher -> correct |
| reasoning_max_rise | shape | 0.540 | higher -> correct |
| reasoning_max_drop | derailment | 0.539 | higher -> wrong |
| reasoning_mid_mean | shape | 0.538 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_std | shape | 0.721 | higher -> correct |
| reasoning_start | shape | 0.529 | higher -> correct |
| reasoning_positive_mass | shape | -0.489 | higher -> wrong |
| reasoning_n_chunks | length | -0.453 | higher -> wrong |
| reasoning_peak_pos | timing | 0.369 | higher -> correct |
| reasoning_negative_mass | shape | -0.345 | higher -> wrong |
| reasoning_end_minus_start | landing | -0.285 | higher -> wrong |
| reasoning_max_rise | shape | 0.281 | higher -> correct |
| reasoning_trough_pos | timing | -0.250 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.211 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.138 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.137 | higher -> correct |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
