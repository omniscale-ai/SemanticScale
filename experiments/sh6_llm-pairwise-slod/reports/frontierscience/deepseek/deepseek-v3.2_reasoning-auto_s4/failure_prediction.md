# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s4 — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 153
- Positive class (`is_correct=true`): 59
- Negative class (`is_correct=false`): 94

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.658 +/- 0.144 | 0.495 +/- 0.107 | 0.767 +/- 0.096 | 0.721 +/- 0.111 | 0.735 +/- 0.091 |
| trajectory_shape (logreg) | 96 | 0.844 +/- 0.053 | 0.779 +/- 0.099 | 0.736 +/- 0.029 | 0.745 +/- 0.058 | 0.678 +/- 0.030 |
| trajectory_full (logreg) | 99 | 0.834 +/- 0.050 | 0.754 +/- 0.103 | 0.706 +/- 0.055 | 0.712 +/- 0.064 | 0.642 +/- 0.073 |
| reasoning_traj (MiniRocket) | 20 | 0.472 +/- 0.021 | 0.425 +/- 0.057 | 0.495 +/- 0.030 | 0.523 +/- 0.043 | 0.360 +/- 0.090 |
| trajectory_full (lightgbm) | 99 | 0.849 +/- 0.021 | 0.780 +/- 0.087 | 0.786 +/- 0.035 | 0.791 +/- 0.048 | 0.738 +/- 0.045 |
| mode_stack (logreg) | 13 | 0.769 +/- 0.098 | 0.652 +/- 0.114 | 0.726 +/- 0.072 | 0.694 +/- 0.085 | 0.690 +/- 0.061 |
| mode_stack (lightgbm) | 13 | 0.822 +/- 0.038 | 0.791 +/- 0.049 | 0.739 +/- 0.035 | 0.752 +/- 0.041 | 0.672 +/- 0.060 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_max_rise | shape | 0.838 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.799 | higher -> correct |
| answer_max_drop | derailment | 0.792 | higher -> correct |
| answer_total_variation | thrashing | 0.785 | higher -> correct |
| answer_monotonicity | commitment | 0.771 | higher -> correct |
| answer_end | landing | 0.769 | higher -> correct |
| answer_direction_changes | thrashing | 0.760 | higher -> correct |
| answer_rebound_from_trough | shape | 0.749 | higher -> correct |
| answer_zero_crossings | thrashing | 0.740 | higher -> correct |
| answer_late_mean | landing | 0.739 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.736 | higher -> correct |
| answer_late_minus_early | transition | 0.726 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_zero_crossings | thrashing | 1.115 | higher -> correct |
| answer_end | landing | -1.069 | higher -> wrong |
| reasoning_traj_t09 | shape | 0.990 | higher -> correct |
| answer_peak_pos | timing | 0.985 | higher -> correct |
| reasoning_trough_pos | timing | -0.924 | higher -> wrong |
| answer_traj_t03 | shape | -0.905 | higher -> wrong |
| reasoning_max_rise | shape | -0.776 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.764 | higher -> correct |
| answer_time_positive | shape | 0.733 | higher -> correct |
| answer_late_mean | landing | -0.729 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | -0.700 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.655 | higher -> correct |

## Interpretable Failure-Mode Detectors

Each detector encodes a pre-registered hypothesis: *higher detector score implies a higher probability of failure*.
For every run we compute a 95% percentile-bootstrap CI on the score's failure-AUC and assign one of four verdicts:

- `confirmed` — CI lower bound above 0.5; the directional claim holds.
- `inverted` — CI upper bound below 0.5; on this run the score actually predicts *success*. The hypothesis is falsified in the opposite direction.
- `inconclusive` — CI straddles 0.5; there is no evidence either way on this run.
- `insufficient_data` — too few scored rows or only one class present.

Flag-level metrics (precision / recall / F1 / lift) are reported only when the verdict is `confirmed`. When the hypothesis is falsified or unclear, those numbers would be actively misleading, so they render as `-`. The raw AUC and CI are always reported so the call can be audited.

### What each detector catches

| Mode | What it catches |
|---|---|
| premature_exit | Very short reasoning trace — the model answered or gave up before exploring. |
| rambling_overlong | Reasoning runs much longer than on successful traces, often without ever synthesising. |
| thrashing | Many SLoD direction changes — the model flip-flops between abstraction levels instead of committing. |
| no_commitment | Low end-to-end monotonicity — the trace never commits to a clear abstraction arc. |
| derailment_late | Trace peaks high on the SLoD axis and then falls away, failing to land. |
| answer_drift | Answer SLoD is far from where the reasoning ended — the answer does not follow the chain. |
| answer_meandering | Answer trajectory has many SLoD direction changes — long, oscillating answer instead of a clean statement (FrontierScience-style hedging). |
| answer_volatility | Large single-step SLoD jump in the answer — the response leaps between abstraction levels, a confabulation pattern. |
| answer_uncommitted | Low monotonicity inside the answer trajectory — the answer never commits to a clear arc. |
| answer_overrange | Answer covers a wider SLoD range than the reasoning did — the answer claims abstraction breadth the reasoning never built up. |
| truncation_abort | Agent exited on context/budget/format, not a clean submit (SWE-agent style). |

### Detector performance

| Mode | Verdict | Scored | Score AUC (95% CI) | Base Fail Rate | Flagged | Precision | Recall | F1 | Lift |
|---|---|---|---|---|---|---|---|---|---|
| premature_exit | inconclusive | 153 | 0.523 [0.421, 0.614] | 0.614 | 21 | - | - | - | - |
| rambling_overlong | inconclusive | 153 | 0.477 [0.386, 0.579] | 0.614 | 19 | - | - | - | - |
| thrashing | inconclusive | 153 | 0.444 [0.356, 0.539] | 0.614 | 2 | - | - | - | - |
| no_commitment | inconclusive | 153 | 0.474 [0.382, 0.567] | 0.614 | 15 | - | - | - | - |
| derailment_late | inconclusive | 153 | 0.469 [0.377, 0.569] | 0.614 | 11 | - | - | - | - |
| answer_drift | inconclusive | 136 | 0.514 [0.414, 0.616] | 0.566 | 13 | - | - | - | - |
| answer_meandering | confirmed | 136 | 0.821 [0.759, 0.883] | 0.566 | 60 | 0.900 | 0.701 | 0.788 | 1.590 |
| answer_volatility | confirmed | 136 | 0.846 [0.784, 0.906] | 0.566 | 60 | 0.900 | 0.701 | 0.788 | 1.590 |
| answer_uncommitted | confirmed | 136 | 0.804 [0.736, 0.872] | 0.566 | 61 | 0.902 | 0.714 | 0.797 | 1.593 |
| answer_overrange | confirmed | 136 | 0.747 [0.659, 0.828] | 0.566 | 45 | 0.867 | 0.506 | 0.639 | 1.531 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_meandering, answer_volatility, answer_uncommitted, answer_overrange.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.769**
- `trajectory_full` ROC-AUC: **0.834**
- Above-chance discrimination preserved by the mode stack: **80.7%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 55 / 94 | 66 | 0.585 | 0.833 |
| any | 11 | 74 / 94 | 101 | 0.787 | 0.733 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
