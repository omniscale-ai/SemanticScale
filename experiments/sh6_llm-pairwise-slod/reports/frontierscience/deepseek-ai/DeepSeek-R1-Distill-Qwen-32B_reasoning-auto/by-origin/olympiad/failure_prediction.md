# SH6 frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 86
- Positive class (`is_correct=true`): 28
- Negative class (`is_correct=false`): 58

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.487 +/- 0.130 | 0.478 +/- 0.149 | 0.522 +/- 0.081 | 0.545 +/- 0.075 | 0.386 +/- 0.118 |
| trajectory_shape (logreg) | 96 | 0.470 +/- 0.059 | 0.412 +/- 0.031 | 0.448 +/- 0.017 | 0.500 +/- 0.059 | 0.283 +/- 0.064 |
| trajectory_full (logreg) | 99 | 0.471 +/- 0.065 | 0.427 +/- 0.062 | 0.418 +/- 0.032 | 0.476 +/- 0.066 | 0.234 +/- 0.127 |
| reasoning_traj (MiniRocket) | 20 | 0.481 +/- 0.150 | 0.377 +/- 0.093 | 0.529 +/- 0.117 | 0.572 +/- 0.102 | 0.366 +/- 0.136 |
| trajectory_full (lightgbm) | 99 | 0.538 +/- 0.085 | 0.447 +/- 0.061 | 0.487 +/- 0.075 | 0.570 +/- 0.087 | 0.276 +/- 0.091 |
| mode_stack (logreg) | 13 | 0.695 +/- 0.047 | 0.605 +/- 0.083 | 0.656 +/- 0.098 | 0.663 +/- 0.078 | 0.544 +/- 0.128 |
| mode_stack (lightgbm) | 13 | 0.592 +/- 0.113 | 0.497 +/- 0.141 | 0.470 +/- 0.070 | 0.570 +/- 0.044 | 0.213 +/- 0.176 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_traj_t18 | shape | 0.709 | higher -> wrong |
| answer_positive_mass | shape | 0.691 | higher -> wrong |
| answer_end | landing | 0.672 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.663 | higher -> correct |
| answer_traj_t17 | shape | 0.658 | higher -> wrong |
| reasoning_max_rise | shape | 0.647 | higher -> correct |
| reasoning_traj_t13 | shape | 0.643 | higher -> correct |
| reasoning_start | shape | 0.642 | higher -> correct |
| answer_fall_from_peak | derailment | 0.639 | higher -> wrong |
| reasoning_traj_t08 | shape | 0.634 | higher -> wrong |
| answer_min | shape | 0.632 | higher -> wrong |
| total_n_chunks | length | 0.632 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_range_minus_reasoning_range | commitment | -1.273 | higher -> wrong |
| reasoning_direction_changes | thrashing | 1.215 | higher -> correct |
| reasoning_traj_t13 | shape | 0.849 | higher -> correct |
| reasoning_traj_t01 | shape | 0.832 | higher -> correct |
| reasoning_traj_t09 | shape | -0.799 | higher -> wrong |
| answer_max_rise | shape | 0.734 | higher -> correct |
| answer_zero_crossings | thrashing | -0.722 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.720 | higher -> correct |
| reasoning_max | shape | 0.701 | higher -> correct |
| answer_max_drop | derailment | 0.635 | higher -> correct |
| answer_trough_pos | timing | -0.623 | higher -> wrong |
| reasoning_max_drop | derailment | -0.607 | higher -> wrong |

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
| premature_exit | inconclusive | 86 | 0.479 [0.352, 0.617] | 0.674 | 8 | - | - | - | - |
| rambling_overlong | inconclusive | 86 | 0.521 [0.383, 0.648] | 0.674 | 11 | - | - | - | - |
| thrashing | inverted | 86 | 0.330 [0.214, 0.467] | 0.674 | 0 | - | - | - | - |
| no_commitment | inconclusive | 86 | 0.511 [0.359, 0.645] | 0.674 | 8 | - | - | - | - |
| derailment_late | inconclusive | 86 | 0.432 [0.302, 0.564] | 0.674 | 3 | - | - | - | - |
| answer_drift | inconclusive | 84 | 0.481 [0.335, 0.620] | 0.667 | 6 | - | - | - | - |
| answer_meandering | inconclusive | 84 | 0.414 [0.274, 0.546] | 0.667 | 5 | - | - | - | - |
| answer_volatility | inconclusive | 84 | 0.406 [0.258, 0.541] | 0.667 | 5 | - | - | - | - |
| answer_uncommitted | inconclusive | 84 | 0.483 [0.344, 0.608] | 0.667 | 13 | - | - | - | - |
| answer_overrange | inconclusive | 84 | 0.501 [0.354, 0.647] | 0.667 | 5 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Hypothesis falsified (inverted)**: thrashing. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.695**
- `trajectory_full` ROC-AUC: **0.471**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 58 | 0 | - | - |
| any | 11 | 26 / 58 | 41 | 0.448 | 0.634 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
