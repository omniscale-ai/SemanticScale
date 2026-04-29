# SH6 frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 145
- Positive class (`is_correct=true`): 31
- Negative class (`is_correct=false`): 114

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.639 +/- 0.106 | 0.379 +/- 0.121 | 0.555 +/- 0.098 | 0.517 +/- 0.087 | 0.348 +/- 0.089 |
| trajectory_shape (logreg) | 60 | 0.596 +/- 0.134 | 0.441 +/- 0.092 | 0.569 +/- 0.078 | 0.655 +/- 0.072 | 0.339 +/- 0.104 |
| trajectory_full (logreg) | 63 | 0.603 +/- 0.135 | 0.448 +/- 0.090 | 0.570 +/- 0.078 | 0.641 +/- 0.056 | 0.340 +/- 0.112 |
| trajectory_full (lightgbm) | 63 | 0.649 +/- 0.093 | 0.416 +/- 0.073 | 0.582 +/- 0.059 | 0.772 +/- 0.052 | 0.319 +/- 0.114 |
| mode_stack (logreg) | 13 | 0.677 +/- 0.062 | 0.523 +/- 0.109 | 0.616 +/- 0.015 | 0.641 +/- 0.064 | 0.408 +/- 0.022 |
| mode_stack (lightgbm) | 13 | 0.634 +/- 0.085 | 0.350 +/- 0.092 | 0.534 +/- 0.047 | 0.717 +/- 0.074 | 0.247 +/- 0.077 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | 0.684 | higher -> correct |
| reasoning_positive_mass | shape | 0.662 | higher -> wrong |
| answer_early_mean | shape | 0.643 | higher -> wrong |
| reasoning_max_rise | shape | 0.628 | higher -> correct |
| reasoning_trough_pos | timing | 0.621 | higher -> correct |
| answer_late_minus_early | transition | 0.617 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.609 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.609 | higher -> correct |
| reasoning_late_mean | landing | 0.605 | higher -> correct |
| reasoning_late_minus_early | transition | 0.605 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.604 | higher -> correct |
| answer_n_chunks | length | 0.603 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_range_minus_reasoning_range | commitment | -1.269 | higher -> wrong |
| reasoning_max_drop | derailment | -1.214 | higher -> wrong |
| reasoning_direction_changes | thrashing | 1.180 | higher -> correct |
| answer_n_chunks | length | -0.799 | higher -> wrong |
| answer_negative_mass | shape | -0.794 | higher -> wrong |
| answer_time_negative | shape | 0.695 | higher -> correct |
| answer_time_positive | shape | -0.695 | higher -> wrong |
| answer_positive_mass | shape | 0.677 | higher -> correct |
| reasoning_peak_pos | timing | 0.631 | higher -> correct |
| answer_early_mean | shape | 0.630 | higher -> correct |
| answer_zero_crossings | thrashing | 0.617 | higher -> correct |
| answer_min | shape | -0.604 | higher -> wrong |

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
| premature_exit | inconclusive | 145 | 0.595 [0.477, 0.704] | 0.786 | 35 | - | - | - | - |
| rambling_overlong | inconclusive | 145 | 0.405 [0.296, 0.523] | 0.786 | 11 | - | - | - | - |
| thrashing | inverted | 145 | 0.329 [0.218, 0.434] | 0.786 | 0 | - | - | - | - |
| no_commitment | inconclusive | 145 | 0.507 [0.384, 0.634] | 0.786 | 11 | - | - | - | - |
| derailment_late | inconclusive | 145 | 0.426 [0.314, 0.547] | 0.786 | 4 | - | - | - | - |
| answer_drift | inconclusive | 142 | 0.496 [0.384, 0.615] | 0.782 | 13 | - | - | - | - |
| answer_meandering | inconclusive | 142 | 0.602 [0.489, 0.711] | 0.782 | 22 | - | - | - | - |
| answer_volatility | inconclusive | 142 | 0.576 [0.463, 0.685] | 0.782 | 21 | - | - | - | - |
| answer_uncommitted | inconclusive | 142 | 0.588 [0.478, 0.702] | 0.782 | 25 | - | - | - | - |
| answer_overrange | inconclusive | 142 | 0.614 [0.487, 0.741] | 0.782 | 17 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Hypothesis falsified (inverted)**: thrashing. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.677**
- `trajectory_full` ROC-AUC: **0.603**
- Above-chance discrimination preserved by the mode stack: **170.8%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 114 | 0 | - | - |
| any | 11 | 70 / 114 | 85 | 0.614 | 0.824 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
