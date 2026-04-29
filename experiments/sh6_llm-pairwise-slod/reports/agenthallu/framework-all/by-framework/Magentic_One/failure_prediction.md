# SH6 agenthallu/framework-all/by-framework/Magentic_One — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 94
- Positive class (`final_answer_correct=true`): 40
- Negative class (`final_answer_correct=false`): 54
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 94 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.389 +/- 0.132 | 0.411 +/- 0.055 | 0.471 +/- 0.103 | 0.445 +/- 0.098 | 0.497 +/- 0.088 |
| trajectory_shape (logreg) | 60 | 0.397 +/- 0.092 | 0.399 +/- 0.036 | 0.447 +/- 0.133 | 0.445 +/- 0.119 | 0.391 +/- 0.173 |
| trajectory_full (logreg) | 63 | 0.399 +/- 0.096 | 0.401 +/- 0.038 | 0.459 +/- 0.147 | 0.456 +/- 0.136 | 0.411 +/- 0.181 |
| trajectory_full (lightgbm) | 63 | 0.385 +/- 0.147 | 0.438 +/- 0.096 | 0.403 +/- 0.078 | 0.415 +/- 0.076 | 0.320 +/- 0.090 |
| mode_stack (logreg) | 13 | 0.417 +/- 0.122 | 0.457 +/- 0.065 | 0.404 +/- 0.106 | 0.404 +/- 0.106 | 0.359 +/- 0.115 |
| mode_stack (lightgbm) | 13 | 0.487 +/- 0.102 | 0.473 +/- 0.096 | 0.535 +/- 0.102 | 0.554 +/- 0.111 | 0.441 +/- 0.094 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_minus_reasoning_mean | answer_alignment | 0.686 | higher -> wrong |
| answer_end_minus_start | landing | 0.678 | higher -> wrong |
| reasoning_mid_mean | shape | 0.674 | higher -> wrong |
| reasoning_end | landing | 0.650 | higher -> wrong |
| reasoning_rebound_from_trough | shape | 0.649 | higher -> wrong |
| reasoning_negative_mass | shape | 0.643 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.641 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | 0.637 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.623 | higher -> correct |
| reasoning_time_positive | shape | 0.622 | higher -> wrong |
| reasoning_time_negative | shape | 0.622 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.621 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max | shape | 0.798 | higher -> correct |
| reasoning_max_rise | shape | -0.780 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -0.684 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | -0.655 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.572 | higher -> correct |
| reasoning_max_rise_pos | timing | -0.539 | higher -> wrong |
| answer_direction_changes | thrashing | 0.526 | higher -> correct |
| answer_early_mean | shape | 0.443 | higher -> correct |
| reasoning_negative_mass | shape | -0.439 | higher -> wrong |
| answer_negative_mass | shape | -0.404 | higher -> wrong |
| reasoning_min | shape | 0.385 | higher -> correct |
| answer_max_rise_pos | timing | -0.383 | higher -> wrong |

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
| premature_exit | inconclusive | 94 | 0.535 [0.421, 0.643] | 0.574 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 94 | 0.465 [0.357, 0.579] | 0.574 | 11 | - | - | - | - |
| thrashing | inconclusive | 94 | 0.492 [0.381, 0.606] | 0.574 | 10 | - | - | - | - |
| no_commitment | inconclusive | 94 | 0.526 [0.406, 0.640] | 0.574 | 8 | - | - | - | - |
| derailment_late | inconclusive | 94 | 0.458 [0.346, 0.574] | 0.574 | 5 | - | - | - | - |
| answer_drift | inconclusive | 94 | 0.485 [0.367, 0.606] | 0.574 | 6 | - | - | - | - |
| answer_meandering | inconclusive | 94 | 0.511 [0.415, 0.602] | 0.574 | 5 | - | - | - | - |
| answer_volatility | inconclusive | 94 | 0.529 [0.430, 0.638] | 0.574 | 13 | - | - | - | - |
| answer_uncommitted | inconclusive | 94 | 0.478 [0.368, 0.589] | 0.574 | 0 | - | - | - | - |
| answer_overrange | inconclusive | 94 | 0.529 [0.412, 0.642] | 0.574 | 20 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.417**
- `trajectory_full` ROC-AUC: **0.399**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 54 | 0 | - | - |
| any | 11 | 26 / 54 | 48 | 0.481 | 0.542 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
