# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1 — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 148
- Positive class (`is_correct=true`): 56
- Negative class (`is_correct=false`): 92

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.658 +/- 0.158 | 0.500 +/- 0.144 | 0.747 +/- 0.069 | 0.695 +/- 0.094 | 0.713 +/- 0.066 |
| trajectory_shape (logreg) | 96 | 0.752 +/- 0.084 | 0.617 +/- 0.102 | 0.654 +/- 0.077 | 0.676 +/- 0.086 | 0.577 +/- 0.092 |
| trajectory_full (logreg) | 99 | 0.745 +/- 0.081 | 0.611 +/- 0.102 | 0.637 +/- 0.069 | 0.662 +/- 0.081 | 0.551 +/- 0.076 |
| reasoning_traj (MiniRocket) | 20 | 0.494 +/- 0.076 | 0.407 +/- 0.042 | 0.469 +/- 0.035 | 0.514 +/- 0.047 | 0.302 +/- 0.056 |
| trajectory_full (lightgbm) | 99 | 0.812 +/- 0.058 | 0.680 +/- 0.089 | 0.676 +/- 0.083 | 0.696 +/- 0.077 | 0.593 +/- 0.108 |
| mode_stack (logreg) | 13 | 0.786 +/- 0.081 | 0.647 +/- 0.111 | 0.715 +/- 0.078 | 0.682 +/- 0.087 | 0.669 +/- 0.081 |
| mode_stack (lightgbm) | 13 | 0.829 +/- 0.025 | 0.719 +/- 0.048 | 0.698 +/- 0.071 | 0.723 +/- 0.042 | 0.597 +/- 0.128 |

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
| reasoning_traj_t01 | shape | 0.863 | higher -> correct |
| reasoning_traj_t06 | shape | -0.858 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.822 | higher -> correct |
| reasoning_traj_t03 | shape | -0.790 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.762 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.756 | higher -> wrong |
| answer_traj_t18 | shape | 0.725 | higher -> correct |
| answer_range | commitment | -0.686 | higher -> wrong |
| answer_peak_pos | timing | 0.681 | higher -> correct |
| answer_late_mean | landing | -0.678 | higher -> wrong |
| answer_traj_t16 | shape | -0.638 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.619 | higher -> wrong |

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

- `mode_stack` ROC-AUC: **0.786**
- `trajectory_full` ROC-AUC: **0.745**
- Above-chance discrimination preserved by the mode stack: **117.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 52 / 92 | 64 | 0.565 | 0.812 |
| any | 11 | 67 / 92 | 97 | 0.728 | 0.691 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
