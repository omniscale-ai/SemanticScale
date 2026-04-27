# SH6 agenthallu/framework-all — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 693
- Positive class (`final_answer_correct=true`): 250
- Negative class (`final_answer_correct=false`): 443
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 693 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 3 | 0.528 +/- 0.076 | 0.399 +/- 0.056 | 0.525 +/- 0.050 | 0.531 +/- 0.065 | 0.438 +/- 0.041 |
| trajectory_shape | 60 | 0.505 +/- 0.024 | 0.375 +/- 0.019 | 0.502 +/- 0.025 | 0.492 +/- 0.034 | 0.432 +/- 0.025 |
| trajectory_full | 63 | 0.504 +/- 0.025 | 0.371 +/- 0.016 | 0.503 +/- 0.024 | 0.496 +/- 0.036 | 0.430 +/- 0.023 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | 0.552 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.551 | higher -> wrong |
| answer_monotonicity | commitment | 0.547 | higher -> correct |
| reasoning_late_minus_early | transition | 0.546 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.546 | higher -> wrong |
| answer_time_positive | shape | 0.543 | higher -> correct |
| reasoning_max_rise | shape | 0.543 | higher -> wrong |
| answer_positive_mass | shape | 0.540 | higher -> correct |
| reasoning_end_minus_start | landing | 0.540 | higher -> wrong |
| answer_max | shape | 0.540 | higher -> correct |
| answer_range | commitment | 0.540 | higher -> correct |
| answer_time_negative | shape | 0.539 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_std | shape | 0.455 | higher -> correct |
| answer_total_variation | thrashing | -0.422 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | -0.324 | higher -> wrong |
| answer_peak_pos | timing | -0.323 | higher -> wrong |
| reasoning_max_rise | shape | -0.314 | higher -> wrong |
| reasoning_negative_mass | shape | -0.300 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.281 | higher -> correct |
| reasoning_monotonicity | commitment | -0.259 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.253 | higher -> correct |
| answer_end_minus_reasoning_end | landing | 0.237 | higher -> correct |
| reasoning_zero_crossings | thrashing | -0.224 | higher -> wrong |
| answer_std | shape | 0.214 | higher -> correct |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
