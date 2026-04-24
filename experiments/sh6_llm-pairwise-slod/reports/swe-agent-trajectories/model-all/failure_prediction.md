# SH6 swe-agent-trajectories/model-all — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 645
- Positive class (`final_answer_correct=true`): 202
- Negative class (`final_answer_correct=false`): 443
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 645 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.736 +/- 0.050 | 0.552 +/- 0.067 | 0.668 +/- 0.018 | 0.614 +/- 0.043 | 0.569 +/- 0.012 |
| trajectory_shape | 29 | 0.865 +/- 0.028 | 0.692 +/- 0.043 | 0.776 +/- 0.032 | 0.775 +/- 0.030 | 0.683 +/- 0.037 |
| trajectory_full | 29 | 0.866 +/- 0.028 | 0.693 +/- 0.043 | 0.776 +/- 0.032 | 0.775 +/- 0.030 | 0.683 +/- 0.037 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | 0.777 | higher -> correct |
| reasoning_std | shape | 0.751 | higher -> correct |
| reasoning_min | shape | 0.746 | higher -> correct |
| reasoning_negative_mass | shape | 0.743 | higher -> correct |
| reasoning_positive_mass | shape | 0.739 | higher -> correct |
| reasoning_n_chunks | length | 0.736 | higher -> correct |
| reasoning_range | commitment | 0.732 | higher -> correct |
| reasoning_max | shape | 0.722 | higher -> correct |
| reasoning_late_minus_early | transition | 0.720 | higher -> correct |
| reasoning_late_mean | landing | 0.719 | higher -> correct |
| reasoning_total_variation | thrashing | 0.705 | higher -> correct |
| reasoning_monotonicity | commitment | 0.704 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | -2.348 | higher -> wrong |
| reasoning_max | shape | -2.135 | higher -> wrong |
| reasoning_max_rise_pos | timing | -1.510 | higher -> wrong |
| reasoning_late_mean | landing | 1.113 | higher -> correct |
| reasoning_peak_pos | timing | 1.023 | higher -> correct |
| reasoning_fall_from_peak | derailment | -1.012 | higher -> wrong |
| reasoning_start | shape | 0.949 | higher -> correct |
| reasoning_end_minus_start | landing | -0.947 | higher -> wrong |
| reasoning_min | shape | -0.848 | higher -> wrong |
| reasoning_range | commitment | -0.681 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.661 | higher -> correct |
| reasoning_max_rise | shape | -0.608 | higher -> wrong |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
