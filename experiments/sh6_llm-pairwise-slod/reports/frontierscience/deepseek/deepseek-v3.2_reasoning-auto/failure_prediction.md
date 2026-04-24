# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 153
- Positive class (`is_correct=true`): 61
- Negative class (`is_correct=false`): 92

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 3 | 0.771 +/- 0.065 | 0.602 +/- 0.095 | 0.801 +/- 0.036 | 0.771 +/- 0.048 | 0.770 +/- 0.037 |
| trajectory_shape | 60 | 0.884 +/- 0.062 | 0.806 +/- 0.123 | 0.774 +/- 0.065 | 0.778 +/- 0.052 | 0.724 +/- 0.085 |
| trajectory_full | 63 | 0.891 +/- 0.052 | 0.817 +/- 0.111 | 0.781 +/- 0.084 | 0.784 +/- 0.069 | 0.728 +/- 0.112 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_max_rise | shape | 0.849 | higher -> correct |
| answer_direction_changes | thrashing | 0.837 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.837 | higher -> correct |
| answer_max_drop | derailment | 0.826 | higher -> correct |
| answer_monotonicity | commitment | 0.815 | higher -> correct |
| answer_total_variation | thrashing | 0.814 | higher -> correct |
| answer_zero_crossings | thrashing | 0.787 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.776 | higher -> correct |
| answer_rebound_from_trough | shape | 0.765 | higher -> correct |
| answer_end | landing | 0.726 | higher -> correct |
| answer_n_chunks | length | 0.700 | higher -> correct |
| answer_late_mean | landing | 0.696 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_fall_from_peak | derailment | 1.471 | higher -> correct |
| answer_monotonicity | commitment | 1.421 | higher -> correct |
| answer_early_mean | shape | -1.150 | higher -> wrong |
| reasoning_total_variation | thrashing | -1.068 | higher -> wrong |
| answer_time_positive | shape | -0.930 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.828 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | 0.745 | higher -> correct |
| answer_end | landing | -0.713 | higher -> wrong |
| answer_mid_mean | shape | -0.697 | higher -> wrong |
| answer_negative_mass | shape | 0.683 | higher -> correct |
| answer_late_minus_early | transition | 0.671 | higher -> correct |
| answer_trough_pos | timing | 0.670 | higher -> correct |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
