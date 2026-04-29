# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s3/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 99
- Positive class (`is_correct=true`): 58
- Negative class (`is_correct=false`): 41

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.670 +/- 0.137 | 0.745 +/- 0.110 | 0.677 +/- 0.127 | 0.688 +/- 0.122 | 0.742 +/- 0.095 |
| trajectory_shape (logreg) | 60 | 0.524 +/- 0.097 | 0.664 +/- 0.071 | 0.512 +/- 0.080 | 0.504 +/- 0.091 | 0.503 +/- 0.151 |
| trajectory_full (logreg) | 63 | 0.618 +/- 0.058 | 0.718 +/- 0.061 | 0.563 +/- 0.062 | 0.555 +/- 0.068 | 0.578 +/- 0.094 |
| trajectory_full (lightgbm) | 63 | 0.629 +/- 0.139 | 0.731 +/- 0.091 | 0.602 +/- 0.126 | 0.626 +/- 0.134 | 0.689 +/- 0.126 |
| mode_stack (logreg) | 13 | 0.689 +/- 0.100 | 0.784 +/- 0.085 | 0.642 +/- 0.084 | 0.645 +/- 0.093 | 0.679 +/- 0.121 |
| mode_stack (lightgbm) | 13 | 0.573 +/- 0.205 | 0.676 +/- 0.160 | 0.596 +/- 0.129 | 0.605 +/- 0.125 | 0.651 +/- 0.132 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_start | shape | 0.666 | higher -> wrong |
| reasoning_negative_mass | shape | 0.655 | higher -> wrong |
| reasoning_std | shape | 0.640 | higher -> wrong |
| answer_n_chunks | length | 0.636 | higher -> correct |
| reasoning_range | commitment | 0.632 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.620 | higher -> correct |
| answer_max | shape | 0.618 | higher -> wrong |
| reasoning_trough_pos | timing | 0.614 | higher -> correct |
| reasoning_total_variation | thrashing | 0.605 | higher -> wrong |
| answer_range | commitment | 0.604 | higher -> wrong |
| reasoning_rebound_from_trough | shape | 0.604 | higher -> correct |
| answer_std | shape | 0.598 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | 1.692 | higher -> correct |
| reasoning_std | shape | 0.861 | higher -> correct |
| answer_rebound_from_trough | shape | 0.748 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.633 | higher -> wrong |
| reasoning_max | shape | -0.624 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.619 | higher -> correct |
| answer_start_minus_reasoning_end | landing | -0.597 | higher -> wrong |
| answer_time_positive | shape | -0.493 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.483 | higher -> correct |
| reasoning_min | shape | -0.452 | higher -> wrong |
| answer_monotonicity | commitment | 0.431 | higher -> correct |
| answer_max_rise_pos | timing | -0.429 | higher -> wrong |

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
| premature_exit | inconclusive | 99 | 0.404 [0.294, 0.525] | 0.414 | 11 | - | - | - | - |
| rambling_overlong | inconclusive | 99 | 0.596 [0.475, 0.706] | 0.414 | 12 | - | - | - | - |
| thrashing | inconclusive | 99 | 0.474 [0.355, 0.585] | 0.414 | 3 | - | - | - | - |
| no_commitment | inconclusive | 99 | 0.602 [0.490, 0.712] | 0.414 | 16 | - | - | - | - |
| derailment_late | inconclusive | 99 | 0.611 [0.497, 0.718] | 0.414 | 13 | - | - | - | - |
| answer_drift | inconclusive | 89 | 0.560 [0.432, 0.681] | 0.348 | 11 | - | - | - | - |
| answer_meandering | inconclusive | 89 | 0.495 [0.420, 0.578] | 0.348 | 4 | - | - | - | - |
| answer_volatility | inconclusive | 89 | 0.480 [0.368, 0.604] | 0.348 | 7 | - | - | - | - |
| answer_uncommitted | inconclusive | 89 | 0.496 [0.414, 0.577] | 0.348 | 7 | - | - | - | - |
| answer_overrange | inconclusive | 89 | 0.461 [0.330, 0.598] | 0.348 | 7 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.689**
- `trajectory_full` ROC-AUC: **0.618**
- Above-chance discrimination preserved by the mode stack: **160.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 41 | 0 | - | - |
| any | 11 | 25 / 41 | 56 | 0.610 | 0.446 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
