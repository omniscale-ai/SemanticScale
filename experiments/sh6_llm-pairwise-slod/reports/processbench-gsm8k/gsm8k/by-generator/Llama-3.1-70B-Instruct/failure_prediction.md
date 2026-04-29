# SH6 processbench/gsm8k/by-generator/Llama-3.1-70B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 35
- Positive class (`final_answer_correct=true`): 29
- Negative class (`final_answer_correct=false`): 6
- Label agreement (`is_correct` vs `final_answer_correct`): 97.1% over 35 items
- Final answer correct but reasoning wrong: 1
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.603 +/- 0.373 | 0.861 +/- 0.148 | 0.527 +/- 0.233 | 0.600 +/- 0.167 | 0.718 +/- 0.126 |
| trajectory_shape (logreg) | 29 | 0.573 +/- 0.336 | 0.888 +/- 0.115 | 0.583 +/- 0.183 | 0.686 +/- 0.107 | 0.790 +/- 0.088 |
| trajectory_full (logreg) | 29 | 0.573 +/- 0.336 | 0.888 +/- 0.115 | 0.583 +/- 0.183 | 0.686 +/- 0.107 | 0.790 +/- 0.088 |
| trajectory_full (lightgbm) | 29 | 0.607 +/- 0.300 | 0.899 +/- 0.110 | 0.660 +/- 0.287 | 0.686 +/- 0.190 | 0.785 +/- 0.130 |
| mode_stack (logreg) | 6 | 0.453 +/- 0.336 | 0.847 +/- 0.140 | 0.507 +/- 0.193 | 0.571 +/- 0.202 | 0.674 +/- 0.185 |
| mode_stack (lightgbm) | 6 | 0.187 +/- 0.248 | 0.776 +/- 0.114 | 0.257 +/- 0.172 | 0.286 +/- 0.090 | 0.405 +/- 0.126 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max | shape | 0.810 | higher -> wrong |
| reasoning_mid_mean | shape | 0.727 | higher -> wrong |
| reasoning_negative_mass | shape | 0.727 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.707 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.707 | higher -> wrong |
| reasoning_std | shape | 0.693 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.687 | higher -> correct |
| reasoning_max_drop | derailment | 0.683 | higher -> correct |
| reasoning_peak_pos | timing | 0.677 | higher -> correct |
| reasoning_time_positive | shape | 0.670 | higher -> wrong |
| reasoning_time_negative | shape | 0.670 | higher -> wrong |
| reasoning_trough_pos | timing | 0.653 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_curvature_abs_mean | thrashing | -1.077 | higher -> wrong |
| reasoning_trough_pos | timing | 0.855 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.838 | higher -> correct |
| reasoning_positive_mass | shape | 0.770 | higher -> correct |
| reasoning_peak_pos | timing | 0.619 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.617 | higher -> correct |
| reasoning_max | shape | 0.499 | higher -> correct |
| reasoning_std | shape | 0.489 | higher -> correct |
| reasoning_monotonicity | commitment | -0.427 | higher -> wrong |
| reasoning_range | commitment | 0.365 | higher -> correct |
| reasoning_end | landing | 0.340 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.307 | higher -> correct |

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
| premature_exit | inconclusive | 35 | 0.486 [0.145, 0.813] | 0.171 | 3 | - | - | - | - |
| rambling_overlong | inconclusive | 35 | 0.514 [0.187, 0.855] | 0.171 | 2 | - | - | - | - |
| thrashing | inconclusive | 35 | 0.598 [0.281, 0.907] | 0.171 | 2 | - | - | - | - |
| no_commitment | inconclusive | 35 | 0.445 [0.219, 0.667] | 0.171 | 3 | - | - | - | - |
| derailment_late | inconclusive | 35 | 0.566 [0.375, 0.765] | 0.171 | 3 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.453**
- `trajectory_full` ROC-AUC: **0.573**
- Above-chance discrimination preserved by the mode stack: **-63.6%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 6 | 0 | - | - |
| any | 11 | 3 / 6 | 12 | 0.500 | 0.250 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
