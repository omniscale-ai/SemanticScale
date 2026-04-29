# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s2 — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 156
- Positive class (`is_correct=true`): 58
- Negative class (`is_correct=false`): 98

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.743 +/- 0.057 | 0.534 +/- 0.058 | 0.777 +/- 0.080 | 0.737 +/- 0.079 | 0.727 +/- 0.070 |
| trajectory_shape (logreg) | 60 | 0.776 +/- 0.084 | 0.661 +/- 0.115 | 0.719 +/- 0.114 | 0.717 +/- 0.126 | 0.665 +/- 0.124 |
| trajectory_full (logreg) | 63 | 0.799 +/- 0.081 | 0.694 +/- 0.100 | 0.748 +/- 0.116 | 0.749 +/- 0.117 | 0.691 +/- 0.139 |
| trajectory_full (lightgbm) | 63 | 0.836 +/- 0.091 | 0.758 +/- 0.151 | 0.769 +/- 0.102 | 0.781 +/- 0.104 | 0.715 +/- 0.129 |
| mode_stack (logreg) | 13 | 0.824 +/- 0.066 | 0.692 +/- 0.102 | 0.738 +/- 0.089 | 0.711 +/- 0.083 | 0.685 +/- 0.088 |
| mode_stack (lightgbm) | 13 | 0.857 +/- 0.093 | 0.797 +/- 0.126 | 0.728 +/- 0.105 | 0.750 +/- 0.101 | 0.654 +/- 0.135 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_total_variation | thrashing | 0.834 | higher -> correct |
| answer_max_drop | derailment | 0.824 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.814 | higher -> correct |
| answer_direction_changes | thrashing | 0.809 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.803 | higher -> correct |
| answer_max_rise | shape | 0.799 | higher -> correct |
| answer_monotonicity | commitment | 0.769 | higher -> correct |
| answer_rebound_from_trough | shape | 0.766 | higher -> correct |
| answer_zero_crossings | thrashing | 0.761 | higher -> correct |
| answer_range | commitment | 0.720 | higher -> correct |
| answer_max | shape | 0.712 | higher -> correct |
| answer_n_chunks | length | 0.707 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_end_minus_reasoning_end | landing | -1.255 | higher -> wrong |
| answer_rebound_from_trough | shape | 1.150 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | -0.952 | higher -> wrong |
| answer_early_mean | shape | -0.827 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.762 | higher -> wrong |
| answer_time_negative | shape | -0.630 | higher -> wrong |
| answer_trough_pos | timing | 0.621 | higher -> correct |
| answer_max_rise | shape | -0.603 | higher -> wrong |
| answer_mid_mean | shape | -0.598 | higher -> wrong |
| answer_min | shape | -0.575 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.557 | higher -> correct |
| answer_max | shape | -0.556 | higher -> wrong |

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
| premature_exit | inconclusive | 156 | 0.483 [0.390, 0.573] | 0.628 | 21 | - | - | - | - |
| rambling_overlong | inconclusive | 156 | 0.517 [0.427, 0.610] | 0.628 | 24 | - | - | - | - |
| thrashing | inconclusive | 156 | 0.551 [0.463, 0.634] | 0.628 | 18 | - | - | - | - |
| no_commitment | inconclusive | 156 | 0.467 [0.369, 0.565] | 0.628 | 20 | - | - | - | - |
| derailment_late | inconclusive | 156 | 0.473 [0.375, 0.573] | 0.628 | 14 | - | - | - | - |
| answer_drift | inconclusive | 142 | 0.496 [0.403, 0.600] | 0.592 | 12 | - | - | - | - |
| answer_meandering | confirmed | 142 | 0.816 [0.755, 0.875] | 0.592 | 63 | 0.905 | 0.679 | 0.776 | 1.529 |
| answer_volatility | confirmed | 142 | 0.807 [0.736, 0.874] | 0.592 | 65 | 0.908 | 0.702 | 0.792 | 1.534 |
| answer_uncommitted | confirmed | 142 | 0.763 [0.689, 0.840] | 0.592 | 58 | 0.897 | 0.619 | 0.732 | 1.516 |
| answer_overrange | confirmed | 142 | 0.809 [0.735, 0.872] | 0.592 | 52 | 0.885 | 0.548 | 0.676 | 1.495 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_meandering, answer_volatility, answer_uncommitted, answer_overrange.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.824**
- `trajectory_full` ROC-AUC: **0.799**
- Above-chance discrimination preserved by the mode stack: **108.5%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 60 / 98 | 72 | 0.612 | 0.833 |
| any | 11 | 83 / 98 | 119 | 0.847 | 0.697 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
