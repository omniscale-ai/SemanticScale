# SH6 swe-agent-trajectories/model-all_steps-50 — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 200
- Positive class (`final_answer_correct=true`): 17
- Negative class (`final_answer_correct=false`): 183
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 200 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.770 +/- 0.047 | 0.233 +/- 0.021 | 0.814 +/- 0.032 | 0.660 +/- 0.056 | 0.335 +/- 0.033 |
| trajectory_shape | 29 | 0.924 +/- 0.020 | 0.389 +/- 0.095 | 0.924 +/- 0.020 | 0.860 +/- 0.037 | 0.554 +/- 0.092 |
| trajectory_full | 29 | 0.924 +/- 0.020 | 0.389 +/- 0.095 | 0.924 +/- 0.020 | 0.860 +/- 0.037 | 0.554 +/- 0.092 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_end_minus_start | landing | 0.924 | higher -> correct |
| reasoning_total_variation | thrashing | 0.924 | higher -> correct |
| reasoning_peak_pos | timing | 0.924 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.924 | higher -> correct |
| reasoning_max_rise | shape | 0.924 | higher -> correct |
| reasoning_time_positive | shape | 0.924 | higher -> correct |
| reasoning_time_negative | shape | 0.924 | higher -> correct |
| reasoning_monotonicity | commitment | 0.896 | higher -> correct |
| reasoning_min | shape | 0.858 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.839 | higher -> correct |
| reasoning_positive_mass | shape | 0.836 | higher -> correct |
| reasoning_late_mean | landing | 0.815 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_monotonicity | commitment | 0.606 | higher -> correct |
| reasoning_time_negative | shape | 0.563 | higher -> correct |
| reasoning_time_positive | shape | -0.563 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.493 | higher -> correct |
| reasoning_max_rise | shape | -0.463 | higher -> wrong |
| reasoning_start | shape | 0.435 | higher -> correct |
| reasoning_peak_pos | timing | -0.377 | higher -> wrong |
| reasoning_end_minus_start | landing | -0.347 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.331 | higher -> wrong |
| reasoning_max_drop_pos | derailment | -0.319 | higher -> wrong |
| reasoning_n_chunks | length | 0.279 | higher -> correct |
| reasoning_positive_mass | shape | -0.277 | higher -> wrong |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
