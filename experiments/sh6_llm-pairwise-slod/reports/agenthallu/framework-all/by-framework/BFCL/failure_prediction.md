# SH6 agenthallu/framework-all/by-framework/BFCL — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 164
- Positive class (`final_answer_correct=true`): 61
- Negative class (`final_answer_correct=false`): 103
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 164 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.467 +/- 0.108 | 0.399 +/- 0.105 | 0.497 +/- 0.075 | 0.500 +/- 0.079 | 0.418 +/- 0.098 |
| trajectory_shape | 29 | 0.502 +/- 0.065 | 0.391 +/- 0.040 | 0.512 +/- 0.088 | 0.530 +/- 0.062 | 0.402 +/- 0.135 |
| trajectory_full | 29 | 0.502 +/- 0.065 | 0.391 +/- 0.040 | 0.512 +/- 0.088 | 0.530 +/- 0.062 | 0.402 +/- 0.135 |
| mode_stack | 5 | 0.484 +/- 0.097 | 0.448 +/- 0.105 | 0.447 +/- 0.105 | 0.445 +/- 0.095 | 0.373 +/- 0.127 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_trough_pos | timing | 0.644 | higher -> correct |
| reasoning_range | commitment | 0.601 | higher -> wrong |
| reasoning_late_mean | landing | 0.587 | higher -> correct |
| reasoning_max | shape | 0.584 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.582 | higher -> correct |
| reasoning_std | shape | 0.580 | higher -> wrong |
| reasoning_positive_mass | shape | 0.573 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.568 | higher -> correct |
| reasoning_late_minus_early | transition | 0.559 | higher -> correct |
| reasoning_end_minus_start | landing | 0.558 | higher -> wrong |
| reasoning_end | landing | 0.557 | higher -> correct |
| reasoning_early_mean | shape | 0.555 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_curvature_abs_mean | thrashing | 0.563 | higher -> correct |
| reasoning_trough_pos | timing | 0.501 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.374 | higher -> wrong |
| reasoning_n_chunks | length | 0.361 | higher -> correct |
| reasoning_std | shape | 0.349 | higher -> correct |
| reasoning_total_variation | thrashing | -0.345 | higher -> wrong |
| reasoning_peak_pos | timing | 0.335 | higher -> correct |
| reasoning_positive_mass | shape | -0.258 | higher -> wrong |
| reasoning_late_mean | landing | -0.256 | higher -> wrong |
| reasoning_negative_mass | shape | 0.207 | higher -> correct |
| reasoning_min | shape | 0.206 | higher -> correct |
| reasoning_max_rise | shape | -0.203 | higher -> wrong |

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
| premature_exit | inconclusive | 164 | 0.528 [0.438, 0.625] | 0.628 | 14 | - | - | - | - |
| rambling_overlong | inconclusive | 164 | 0.472 [0.375, 0.562] | 0.628 | 10 | - | - | - | - |
| thrashing | inconclusive | 164 | 0.483 [0.382, 0.576] | 0.628 | 16 | - | - | - | - |
| no_commitment | inconclusive | 164 | 0.473 [0.380, 0.560] | 0.628 | 16 | - | - | - | - |
| derailment_late | inconclusive | 164 | 0.530 [0.437, 0.614] | 0.628 | 20 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.484**
- `trajectory_full` ROC-AUC: **0.502**
- Above-chance discrimination preserved by the mode stack: **-816.6%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 103 | 0 | - | - |
| any | 11 | 41 / 103 | 59 | 0.398 | 0.695 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
