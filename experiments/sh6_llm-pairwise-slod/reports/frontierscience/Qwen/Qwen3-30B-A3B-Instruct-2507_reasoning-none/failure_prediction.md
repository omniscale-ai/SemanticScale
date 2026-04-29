# SH6 frontierscience/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 158
- Positive class (`is_correct=true`): 82
- Negative class (`is_correct=false`): 76

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.713 +/- 0.055 | 0.693 +/- 0.046 | 0.663 +/- 0.053 | 0.664 +/- 0.053 | 0.693 +/- 0.063 |
| trajectory_shape (logreg) | 29 | 0.599 +/- 0.075 | 0.594 +/- 0.067 | 0.602 +/- 0.042 | 0.601 +/- 0.041 | 0.616 +/- 0.049 |
| trajectory_full (logreg) | 29 | 0.599 +/- 0.075 | 0.594 +/- 0.067 | 0.602 +/- 0.042 | 0.601 +/- 0.041 | 0.616 +/- 0.049 |
| trajectory_full (lightgbm) | 29 | 0.619 +/- 0.021 | 0.644 +/- 0.049 | 0.576 +/- 0.034 | 0.576 +/- 0.033 | 0.610 +/- 0.062 |
| mode_stack (logreg) | 4 | 0.698 +/- 0.076 | 0.675 +/- 0.076 | 0.655 +/- 0.054 | 0.658 +/- 0.055 | 0.692 +/- 0.068 |
| mode_stack (lightgbm) | 4 | 0.641 +/- 0.044 | 0.693 +/- 0.053 | 0.599 +/- 0.073 | 0.601 +/- 0.073 | 0.609 +/- 0.083 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_n_chunks | length | 0.713 | higher -> correct |
| answer_monotonicity | commitment | 0.632 | higher -> wrong |
| answer_end_minus_start | landing | 0.598 | higher -> wrong |
| answer_peak_pos | timing | 0.585 | higher -> wrong |
| answer_max_drop | derailment | 0.577 | higher -> wrong |
| answer_range | commitment | 0.571 | higher -> correct |
| answer_max | shape | 0.570 | higher -> correct |
| answer_direction_changes | thrashing | 0.565 | higher -> wrong |
| answer_rebound_from_trough | shape | 0.564 | higher -> correct |
| answer_negative_mass | shape | 0.557 | higher -> wrong |
| answer_start | shape | 0.552 | higher -> wrong |
| answer_early_mean | shape | 0.552 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | -0.902 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | 0.807 | higher -> correct |
| answer_std | shape | -0.511 | higher -> wrong |
| answer_total_variation | thrashing | -0.442 | higher -> wrong |
| answer_max | shape | 0.327 | higher -> correct |
| answer_range | commitment | 0.214 | higher -> correct |
| answer_max_rise | shape | -0.203 | higher -> wrong |
| answer_zero_crossings | thrashing | -0.194 | higher -> wrong |
| answer_time_negative | shape | 0.166 | higher -> correct |
| answer_time_positive | shape | -0.166 | higher -> wrong |
| answer_peak_pos | timing | -0.166 | higher -> wrong |
| answer_start | shape | 0.157 | higher -> correct |

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
| premature_exit | inconclusive | 158 | 0.500 [0.500, 0.500] | 0.481 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 158 | 0.500 [0.500, 0.500] | 0.481 | 0 | - | - | - | - |
| thrashing | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| no_commitment | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| derailment_late | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | inconclusive | 158 | 0.489 [0.402, 0.574] | 0.481 | 2 | - | - | - | - |
| answer_volatility | inconclusive | 158 | 0.505 [0.417, 0.591] | 0.481 | 17 | - | - | - | - |
| answer_uncommitted | inconclusive | 158 | 0.526 [0.435, 0.611] | 0.481 | 17 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, answer_meandering, answer_volatility, answer_uncommitted. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.698**
- `trajectory_full` ROC-AUC: **0.599**
- Above-chance discrimination preserved by the mode stack: **199.0%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 76 | 0 | - | - |
| any | 11 | 16 / 76 | 35 | 0.211 | 0.457 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
