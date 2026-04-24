# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 153
- Positive class (`is_correct=true`): 61
- Negative class (`is_correct=false`): 92

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 3 | 0.765 +/- 0.065 | 0.593 +/- 0.097 | 0.785 +/- 0.048 | 0.751 +/- 0.064 | 0.756 +/- 0.044 |
| trajectory_shape | 60 | 0.848 +/- 0.089 | 0.756 +/- 0.151 | 0.789 +/- 0.098 | 0.790 +/- 0.094 | 0.748 +/- 0.117 |
| trajectory_full | 63 | 0.835 +/- 0.090 | 0.742 +/- 0.153 | 0.776 +/- 0.086 | 0.777 +/- 0.084 | 0.733 +/- 0.104 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_max_rise | shape | 0.848 | higher -> correct |
| answer_direction_changes | thrashing | 0.838 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.820 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.801 | higher -> correct |
| answer_monotonicity | commitment | 0.796 | higher -> correct |
| answer_max_drop | derailment | 0.782 | higher -> correct |
| answer_zero_crossings | thrashing | 0.774 | higher -> correct |
| answer_total_variation | thrashing | 0.756 | higher -> correct |
| answer_end | landing | 0.742 | higher -> correct |
| answer_rebound_from_trough | shape | 0.736 | higher -> correct |
| answer_minus_reasoning_mean | answer_alignment | 0.716 | higher -> correct |
| answer_end_minus_start | landing | 0.710 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_monotonicity | commitment | 1.620 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | -1.060 | higher -> wrong |
| answer_end | landing | -0.983 | higher -> wrong |
| answer_direction_changes | thrashing | -0.766 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.732 | higher -> correct |
| answer_zero_crossings | thrashing | -0.705 | higher -> wrong |
| answer_early_mean | shape | -0.699 | higher -> wrong |
| answer_max_drop_pos | derailment | -0.664 | higher -> wrong |
| reasoning_max_rise_pos | timing | -0.650 | higher -> wrong |
| answer_mid_mean | shape | -0.642 | higher -> wrong |
| reasoning_rebound_from_trough | shape | -0.617 | higher -> wrong |
| answer_end_minus_start | landing | -0.614 | higher -> wrong |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
