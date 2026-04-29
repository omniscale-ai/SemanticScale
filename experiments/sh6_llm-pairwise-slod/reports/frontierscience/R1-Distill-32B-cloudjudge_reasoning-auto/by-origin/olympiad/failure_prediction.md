# SH6 frontierscience/R1-Distill-32B-cloudjudge_reasoning-auto/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 88
- Positive class (`is_correct=true`): 27
- Negative class (`is_correct=false`): 61

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.524 +/- 0.129 | 0.461 +/- 0.104 | 0.564 +/- 0.090 | 0.580 +/- 0.088 | 0.432 +/- 0.108 |
| trajectory_shape (logreg) | 60 | 0.489 +/- 0.110 | 0.384 +/- 0.059 | 0.468 +/- 0.108 | 0.524 +/- 0.067 | 0.258 +/- 0.182 |
| trajectory_full (logreg) | 63 | 0.493 +/- 0.110 | 0.384 +/- 0.048 | 0.504 +/- 0.128 | 0.558 +/- 0.077 | 0.285 +/- 0.214 |
| trajectory_full (lightgbm) | 63 | 0.484 +/- 0.100 | 0.376 +/- 0.072 | 0.503 +/- 0.076 | 0.580 +/- 0.043 | 0.293 +/- 0.121 |
| mode_stack (logreg) | 13 | 0.505 +/- 0.153 | 0.401 +/- 0.100 | 0.535 +/- 0.114 | 0.571 +/- 0.107 | 0.387 +/- 0.138 |
| mode_stack (lightgbm) | 13 | 0.468 +/- 0.088 | 0.346 +/- 0.065 | 0.460 +/- 0.112 | 0.546 +/- 0.102 | 0.221 +/- 0.189 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_end_minus_reasoning_end | landing | 0.722 | higher -> wrong |
| answer_time_positive | shape | 0.681 | higher -> wrong |
| answer_time_negative | shape | 0.681 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.664 | higher -> correct |
| answer_max | shape | 0.654 | higher -> wrong |
| answer_trough_pos | timing | 0.653 | higher -> correct |
| answer_max_rise_pos | timing | 0.645 | higher -> wrong |
| reasoning_max | shape | 0.643 | higher -> correct |
| reasoning_range | commitment | 0.641 | higher -> correct |
| reasoning_late_minus_early | transition | 0.640 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.636 | higher -> wrong |
| answer_min | shape | 0.631 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_positive_mass | shape | -0.992 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.990 | higher -> correct |
| answer_monotonicity | commitment | 0.913 | higher -> correct |
| answer_total_variation | thrashing | 0.766 | higher -> correct |
| answer_trough_pos | timing | 0.739 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.637 | higher -> correct |
| answer_max_drop | derailment | 0.634 | higher -> correct |
| reasoning_trough_pos | timing | -0.629 | higher -> wrong |
| answer_zero_crossings | thrashing | 0.607 | higher -> correct |
| answer_start | shape | 0.549 | higher -> correct |
| reasoning_max | shape | 0.549 | higher -> correct |
| answer_max_drop_pos | derailment | 0.517 | higher -> correct |

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
| premature_exit | inconclusive | 88 | 0.503 [0.368, 0.641] | 0.693 | 10 | - | - | - | - |
| rambling_overlong | inconclusive | 88 | 0.497 [0.359, 0.632] | 0.693 | 9 | - | - | - | - |
| thrashing | inconclusive | 88 | 0.526 [0.413, 0.656] | 0.693 | 18 | - | - | - | - |
| no_commitment | confirmed | 88 | 0.662 [0.533, 0.786] | 0.693 | 26 | 0.885 | 0.377 | 0.529 | 1.276 |
| derailment_late | inconclusive | 88 | 0.414 [0.277, 0.556] | 0.693 | 4 | - | - | - | - |
| answer_drift | inconclusive | 86 | 0.435 [0.300, 0.569] | 0.686 | 8 | - | - | - | - |
| answer_meandering | inconclusive | 86 | 0.475 [0.344, 0.606] | 0.686 | 6 | - | - | - | - |
| answer_volatility | inconclusive | 86 | 0.482 [0.342, 0.620] | 0.686 | 11 | - | - | - | - |
| answer_uncommitted | inconclusive | 86 | 0.494 [0.362, 0.636] | 0.686 | 11 | - | - | - | - |
| answer_overrange | inconclusive | 86 | 0.457 [0.325, 0.581] | 0.686 | 6 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: no_commitment.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.505**
- `trajectory_full` ROC-AUC: **0.493**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 23 / 61 | 26 | 0.377 | 0.885 |
| any | 11 | 50 / 61 | 69 | 0.820 | 0.725 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
