# SH6 processbench/olympiadbench/by-generator/Qwen2-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 48
- Positive class (`final_answer_correct=true`): 31
- Negative class (`final_answer_correct=false`): 17
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 48 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.416 +/- 0.158 | 0.667 +/- 0.101 | 0.488 +/- 0.117 | 0.547 +/- 0.105 | 0.646 +/- 0.129 |
| trajectory_shape (logreg) | 29 | 0.446 +/- 0.134 | 0.688 +/- 0.118 | 0.513 +/- 0.183 | 0.544 +/- 0.161 | 0.613 +/- 0.204 |
| trajectory_full (logreg) | 29 | 0.446 +/- 0.134 | 0.688 +/- 0.118 | 0.513 +/- 0.183 | 0.544 +/- 0.161 | 0.613 +/- 0.204 |
| trajectory_full (lightgbm) | 29 | 0.577 +/- 0.145 | 0.759 +/- 0.090 | 0.608 +/- 0.050 | 0.689 +/- 0.085 | 0.776 +/- 0.080 |
| mode_stack (logreg) | 6 | 0.442 +/- 0.322 | 0.708 +/- 0.178 | 0.513 +/- 0.124 | 0.584 +/- 0.084 | 0.699 +/- 0.042 |
| mode_stack (lightgbm) | 6 | 0.528 +/- 0.165 | 0.715 +/- 0.095 | 0.532 +/- 0.175 | 0.584 +/- 0.145 | 0.685 +/- 0.110 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_std | shape | 0.726 | higher -> wrong |
| reasoning_negative_mass | shape | 0.712 | higher -> wrong |
| reasoning_min | shape | 0.705 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.685 | higher -> wrong |
| reasoning_max_rise | shape | 0.661 | higher -> wrong |
| reasoning_early_mean | shape | 0.642 | higher -> correct |
| reasoning_late_minus_early | transition | 0.633 | higher -> correct |
| reasoning_range | commitment | 0.632 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.602 | higher -> wrong |
| reasoning_peak_pos | timing | 0.598 | higher -> correct |
| reasoning_mid_mean | shape | 0.596 | higher -> correct |
| reasoning_late_mean | landing | 0.591 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_peak_pos | timing | -0.604 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.581 | higher -> correct |
| reasoning_max_drop | derailment | -0.573 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.537 | higher -> wrong |
| reasoning_early_mean | shape | 0.457 | higher -> correct |
| reasoning_max_rise | shape | -0.401 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.369 | higher -> correct |
| reasoning_late_minus_early | transition | -0.313 | higher -> wrong |
| reasoning_max_rise_pos | timing | -0.309 | higher -> wrong |
| reasoning_time_positive | shape | -0.275 | higher -> wrong |
| reasoning_rebound_from_trough | shape | 0.266 | higher -> correct |
| reasoning_fall_from_peak | derailment | -0.231 | higher -> wrong |

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
| premature_exit | inconclusive | 48 | 0.528 [0.346, 0.707] | 0.354 | 4 | - | - | - | - |
| rambling_overlong | inconclusive | 48 | 0.472 [0.293, 0.654] | 0.354 | 5 | - | - | - | - |
| thrashing | inconclusive | 48 | 0.461 [0.290, 0.646] | 0.354 | 4 | - | - | - | - |
| no_commitment | inconclusive | 48 | 0.501 [0.299, 0.699] | 0.354 | 7 | - | - | - | - |
| derailment_late | inconclusive | 48 | 0.510 [0.319, 0.677] | 0.354 | 7 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.442**
- `trajectory_full` ROC-AUC: **0.446**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 17 | 0 | - | - |
| any | 11 | 9 / 17 | 20 | 0.529 | 0.450 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
