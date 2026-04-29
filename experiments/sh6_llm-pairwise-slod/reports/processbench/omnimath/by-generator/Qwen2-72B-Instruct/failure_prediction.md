# SH6 processbench/omnimath/by-generator/Qwen2-72B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 37
- Positive class (`final_answer_correct=true`): 20
- Negative class (`final_answer_correct=false`): 17
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 37 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.596 +/- 0.213 | 0.693 +/- 0.114 | 0.592 +/- 0.239 | 0.596 +/- 0.243 | 0.570 +/- 0.327 |
| trajectory_shape (logreg) | 29 | 0.596 +/- 0.064 | 0.693 +/- 0.105 | 0.525 +/- 0.086 | 0.539 +/- 0.068 | 0.601 +/- 0.063 |
| trajectory_full (logreg) | 29 | 0.596 +/- 0.064 | 0.693 +/- 0.105 | 0.525 +/- 0.086 | 0.539 +/- 0.068 | 0.601 +/- 0.063 |
| trajectory_full (lightgbm) | 29 | 0.637 +/- 0.156 | 0.729 +/- 0.145 | 0.600 +/- 0.146 | 0.600 +/- 0.154 | 0.597 +/- 0.194 |
| mode_stack (logreg) | 6 | 0.567 +/- 0.164 | 0.675 +/- 0.141 | 0.450 +/- 0.113 | 0.436 +/- 0.116 | 0.328 +/- 0.215 |
| mode_stack (lightgbm) | 6 | 0.692 +/- 0.206 | 0.742 +/- 0.170 | 0.708 +/- 0.246 | 0.707 +/- 0.255 | 0.679 +/- 0.305 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_min | shape | 0.775 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.758 | higher -> correct |
| reasoning_range | commitment | 0.737 | higher -> wrong |
| reasoning_start | shape | 0.729 | higher -> correct |
| reasoning_negative_mass | shape | 0.725 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.688 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.642 | higher -> correct |
| reasoning_late_mean | landing | 0.638 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.637 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.629 | higher -> wrong |
| reasoning_std | shape | 0.625 | higher -> wrong |
| reasoning_max_drop | derailment | 0.625 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_positive_mass | shape | -0.848 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.710 | higher -> correct |
| reasoning_max_rise | shape | 0.678 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.602 | higher -> correct |
| reasoning_start | shape | 0.589 | higher -> correct |
| reasoning_max_drop | derailment | -0.523 | higher -> wrong |
| reasoning_late_mean | landing | -0.481 | higher -> wrong |
| reasoning_n_chunks | length | -0.460 | higher -> wrong |
| reasoning_end | landing | 0.385 | higher -> correct |
| reasoning_std | shape | -0.367 | higher -> wrong |
| reasoning_trough_pos | timing | -0.351 | higher -> wrong |
| reasoning_late_minus_early | transition | -0.317 | higher -> wrong |

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
| premature_exit | inconclusive | 37 | 0.396 [0.205, 0.593] | 0.459 | 2 | - | - | - | - |
| rambling_overlong | inconclusive | 37 | 0.604 [0.407, 0.795] | 0.459 | 5 | - | - | - | - |
| thrashing | inconclusive | 37 | 0.626 [0.439, 0.814] | 0.459 | 7 | - | - | - | - |
| no_commitment | inconclusive | 37 | 0.576 [0.381, 0.756] | 0.459 | 5 | - | - | - | - |
| derailment_late | inconclusive | 37 | 0.578 [0.373, 0.769] | 0.459 | 2 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.567**
- `trajectory_full` ROC-AUC: **0.596**
- Above-chance discrimination preserved by the mode stack: **69.6%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 17 | 0 | - | - |
| any | 11 | 8 / 17 | 15 | 0.471 | 0.533 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
