# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1 — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 148
- Positive class (`is_correct=true`): 56
- Negative class (`is_correct=false`): 92

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 3 | 0.658 +/- 0.158 | 0.500 +/- 0.144 | 0.747 +/- 0.069 | 0.695 +/- 0.094 | 0.713 +/- 0.066 |
| trajectory_shape | 60 | 0.763 +/- 0.079 | 0.620 +/- 0.100 | 0.672 +/- 0.074 | 0.669 +/- 0.078 | 0.609 +/- 0.080 |
| trajectory_full | 63 | 0.763 +/- 0.079 | 0.618 +/- 0.103 | 0.699 +/- 0.058 | 0.689 +/- 0.070 | 0.643 +/- 0.058 |
| mode_stack | 10 | 0.792 +/- 0.080 | 0.665 +/- 0.107 | 0.723 +/- 0.067 | 0.689 +/- 0.081 | 0.680 +/- 0.064 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_max_drop | derailment | 0.809 | higher -> correct |
| answer_total_variation | thrashing | 0.801 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.794 | higher -> correct |
| answer_max_rise | shape | 0.765 | higher -> correct |
| answer_direction_changes | thrashing | 0.758 | higher -> correct |
| answer_zero_crossings | thrashing | 0.751 | higher -> correct |
| answer_monotonicity | commitment | 0.751 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.744 | higher -> correct |
| answer_rebound_from_trough | shape | 0.706 | higher -> correct |
| answer_range | commitment | 0.691 | higher -> correct |
| answer_max | shape | 0.685 | higher -> correct |
| answer_n_chunks | length | 0.683 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_rebound_from_trough | shape | 0.930 | higher -> correct |
| answer_range | commitment | -0.908 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.859 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.792 | higher -> wrong |
| answer_end_minus_reasoning_end | landing | -0.737 | higher -> wrong |
| answer_negative_mass | shape | 0.698 | higher -> correct |
| answer_zero_crossings | thrashing | -0.694 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | -0.679 | higher -> wrong |
| answer_max | shape | -0.641 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.633 | higher -> wrong |
| answer_peak_pos | timing | 0.609 | higher -> correct |
| answer_direction_changes | thrashing | -0.581 | higher -> wrong |

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
| premature_exit | inconclusive | 148 | 0.514 [0.415, 0.599] | 0.622 | 15 | - | - | - | - |
| rambling_overlong | inconclusive | 148 | 0.486 [0.401, 0.585] | 0.622 | 21 | - | - | - | - |
| thrashing | inconclusive | 148 | 0.544 [0.444, 0.646] | 0.622 | 10 | - | - | - | - |
| no_commitment | inconclusive | 148 | 0.465 [0.376, 0.564] | 0.622 | 23 | - | - | - | - |
| derailment_late | inconclusive | 148 | 0.414 [0.314, 0.517] | 0.622 | 6 | - | - | - | - |
| answer_drift | inconclusive | 135 | 0.496 [0.387, 0.601] | 0.585 | 8 | - | - | - | - |
| answer_meandering | confirmed | 135 | 0.791 [0.722, 0.854] | 0.585 | 56 | 0.911 | 0.646 | 0.756 | 1.556 |
| answer_volatility | confirmed | 135 | 0.768 [0.679, 0.843] | 0.585 | 56 | 0.893 | 0.633 | 0.741 | 1.526 |
| answer_uncommitted | confirmed | 135 | 0.755 [0.671, 0.830] | 0.585 | 56 | 0.893 | 0.633 | 0.741 | 1.526 |
| answer_overrange | confirmed | 135 | 0.736 [0.653, 0.818] | 0.585 | 42 | 0.857 | 0.456 | 0.595 | 1.465 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_meandering, answer_volatility, answer_uncommitted, answer_overrange.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.792**
- `trajectory_full` ROC-AUC: **0.763**
- Above-chance discrimination preserved by the mode stack: **110.7%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 52 / 92 | 64 | 0.565 | 0.812 |
| any | 11 | 67 / 92 | 97 | 0.728 | 0.691 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
