# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 96
- Positive class (`is_correct=true`): 56
- Negative class (`is_correct=false`): 40

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.521 +/- 0.026 | 0.630 +/- 0.031 | 0.544 +/- 0.099 | 0.573 +/- 0.103 | 0.655 +/- 0.093 |
| trajectory_shape (logreg) | 96 | 0.512 +/- 0.085 | 0.619 +/- 0.055 | 0.548 +/- 0.090 | 0.561 +/- 0.090 | 0.616 +/- 0.098 |
| trajectory_full (logreg) | 99 | 0.511 +/- 0.097 | 0.619 +/- 0.062 | 0.542 +/- 0.099 | 0.551 +/- 0.102 | 0.589 +/- 0.128 |
| reasoning_traj (MiniRocket) | 20 | 0.509 +/- 0.132 | 0.658 +/- 0.107 | 0.527 +/- 0.094 | 0.541 +/- 0.089 | 0.604 +/- 0.078 |
| trajectory_full (lightgbm) | 99 | 0.580 +/- 0.099 | 0.688 +/- 0.079 | 0.534 +/- 0.133 | 0.539 +/- 0.131 | 0.581 +/- 0.140 |
| mode_stack (logreg) | 13 | 0.620 +/- 0.098 | 0.724 +/- 0.107 | 0.622 +/- 0.072 | 0.614 +/- 0.058 | 0.627 +/- 0.066 |
| mode_stack (lightgbm) | 13 | 0.681 +/- 0.113 | 0.776 +/- 0.088 | 0.620 +/- 0.124 | 0.635 +/- 0.111 | 0.699 +/- 0.082 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_n_chunks | length | 0.647 | higher -> wrong |
| answer_max | shape | 0.634 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.634 | higher -> wrong |
| reasoning_traj_t06 | shape | 0.626 | higher -> correct |
| answer_traj_t13 | shape | 0.624 | higher -> wrong |
| answer_max_rise_pos | timing | 0.620 | higher -> wrong |
| reasoning_max | shape | 0.619 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.616 | higher -> wrong |
| reasoning_start | shape | 0.611 | higher -> wrong |
| reasoning_traj_t14 | shape | 0.609 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.605 | higher -> correct |
| reasoning_traj_t13 | shape | 0.605 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_range_minus_reasoning_range | commitment | 1.288 | higher -> correct |
| answer_rebound_from_trough | shape | 1.186 | higher -> correct |
| reasoning_traj_t01 | shape | 1.105 | higher -> correct |
| reasoning_traj_t06 | shape | -0.932 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.912 | higher -> wrong |
| reasoning_traj_t03 | shape | -0.754 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.721 | higher -> wrong |
| reasoning_max_rise | shape | 0.674 | higher -> correct |
| answer_trough_pos | timing | -0.672 | higher -> wrong |
| reasoning_std | shape | 0.586 | higher -> correct |
| reasoning_total_variation | thrashing | -0.555 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.539 | higher -> wrong |

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
| premature_exit | inconclusive | 96 | 0.432 [0.316, 0.559] | 0.417 | 6 | - | - | - | - |
| rambling_overlong | inconclusive | 96 | 0.568 [0.441, 0.684] | 0.417 | 14 | - | - | - | - |
| thrashing | inconclusive | 96 | 0.560 [0.440, 0.673] | 0.417 | 7 | - | - | - | - |
| no_commitment | inconclusive | 96 | 0.439 [0.312, 0.564] | 0.417 | 14 | - | - | - | - |
| derailment_late | inconclusive | 96 | 0.405 [0.294, 0.521] | 0.417 | 6 | - | - | - | - |
| answer_drift | inconclusive | 87 | 0.514 [0.392, 0.638] | 0.356 | 7 | - | - | - | - |
| answer_meandering | inconclusive | 87 | 0.488 [0.408, 0.578] | 0.356 | 9 | - | - | - | - |
| answer_volatility | inconclusive | 87 | 0.435 [0.308, 0.561] | 0.356 | 9 | - | - | - | - |
| answer_uncommitted | inconclusive | 87 | 0.484 [0.396, 0.580] | 0.356 | 9 | - | - | - | - |
| answer_overrange | inconclusive | 87 | 0.430 [0.311, 0.549] | 0.356 | 6 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.620**
- `trajectory_full` ROC-AUC: **0.511**
- Above-chance discrimination preserved by the mode stack: **1130.4%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 40 | 0 | - | - |
| any | 11 | 17 / 40 | 47 | 0.425 | 0.362 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
