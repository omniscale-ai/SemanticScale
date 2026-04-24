# SH6 processbench/olympiadbench — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 400
- Positive class (`final_answer_correct=true`): 200
- Negative class (`final_answer_correct=false`): 200
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 400 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.485 +/- 0.045 | 0.505 +/- 0.024 | 0.510 +/- 0.040 | 0.510 +/- 0.040 | 0.541 +/- 0.061 |
| trajectory_shape | 29 | 0.471 +/- 0.042 | 0.515 +/- 0.029 | 0.460 +/- 0.020 | 0.460 +/- 0.020 | 0.458 +/- 0.059 |
| trajectory_full | 29 | 0.471 +/- 0.042 | 0.515 +/- 0.029 | 0.460 +/- 0.020 | 0.460 +/- 0.020 | 0.458 +/- 0.059 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_negative_mass | shape | 0.564 | higher -> wrong |
| reasoning_start | shape | 0.559 | higher -> wrong |
| reasoning_max | shape | 0.547 | higher -> wrong |
| reasoning_std | shape | 0.545 | higher -> wrong |
| reasoning_max_rise | shape | 0.545 | higher -> wrong |
| reasoning_range | commitment | 0.543 | higher -> wrong |
| reasoning_end | landing | 0.539 | higher -> correct |
| reasoning_peak_pos | timing | 0.539 | higher -> wrong |
| reasoning_min | shape | 0.536 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.533 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.532 | higher -> wrong |
| reasoning_trough_pos | timing | 0.532 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_std | shape | -0.578 | higher -> wrong |
| reasoning_negative_mass | shape | 0.379 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.288 | higher -> wrong |
| reasoning_start | shape | -0.273 | higher -> wrong |
| reasoning_late_mean | landing | -0.257 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.246 | higher -> correct |
| reasoning_max_drop_pos | derailment | -0.198 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.185 | higher -> correct |
| reasoning_late_minus_early | transition | -0.139 | higher -> wrong |
| reasoning_mid_mean | shape | -0.137 | higher -> wrong |
| reasoning_positive_mass | shape | -0.124 | higher -> wrong |
| reasoning_n_chunks | length | -0.122 | higher -> wrong |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
