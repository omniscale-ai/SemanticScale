# SH6 processbench/omnimath — Failure Prediction

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
| length_only | 1 | 0.631 +/- 0.024 | 0.635 +/- 0.027 | 0.577 +/- 0.015 | 0.577 +/- 0.015 | 0.604 +/- 0.029 |
| trajectory_shape | 29 | 0.632 +/- 0.056 | 0.655 +/- 0.047 | 0.578 +/- 0.040 | 0.578 +/- 0.040 | 0.569 +/- 0.041 |
| trajectory_full | 29 | 0.632 +/- 0.056 | 0.655 +/- 0.047 | 0.578 +/- 0.040 | 0.578 +/- 0.040 | 0.569 +/- 0.041 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | 0.631 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.628 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.627 | higher -> correct |
| reasoning_total_variation | thrashing | 0.607 | higher -> correct |
| reasoning_mid_mean | shape | 0.607 | higher -> correct |
| reasoning_max_rise | shape | 0.604 | higher -> correct |
| reasoning_negative_mass | shape | 0.586 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.585 | higher -> correct |
| reasoning_max_drop | derailment | 0.578 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.569 | higher -> correct |
| reasoning_time_positive | shape | 0.569 | higher -> correct |
| reasoning_time_negative | shape | 0.564 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | -0.687 | higher -> wrong |
| reasoning_positive_mass | shape | -0.574 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.498 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.422 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.371 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.288 | higher -> correct |
| reasoning_mid_mean | shape | -0.277 | higher -> wrong |
| reasoning_late_mean | landing | -0.265 | higher -> wrong |
| reasoning_negative_mass | shape | 0.207 | higher -> correct |
| reasoning_peak_pos | timing | 0.189 | higher -> correct |
| reasoning_late_minus_early | transition | -0.172 | higher -> wrong |
| reasoning_max_rise | shape | -0.164 | higher -> wrong |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
