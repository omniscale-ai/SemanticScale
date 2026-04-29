# SH6 processbench/omnimath — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 400
- Positive class (`final_answer_correct=true`): 200
- Negative class (`final_answer_correct=false`): 200
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 400 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.631 +/- 0.024 | 0.635 +/- 0.027 | 0.577 +/- 0.015 | 0.577 +/- 0.015 | 0.604 +/- 0.029 |
| trajectory_shape (logreg) | 29 | 0.632 +/- 0.056 | 0.655 +/- 0.047 | 0.578 +/- 0.040 | 0.578 +/- 0.040 | 0.569 +/- 0.041 |
| trajectory_full (logreg) | 29 | 0.632 +/- 0.056 | 0.655 +/- 0.047 | 0.578 +/- 0.040 | 0.578 +/- 0.040 | 0.569 +/- 0.041 |
| trajectory_full (lightgbm) | 29 | 0.554 +/- 0.057 | 0.588 +/- 0.056 | 0.540 +/- 0.048 | 0.540 +/- 0.048 | 0.534 +/- 0.033 |
| mode_stack (logreg) | 6 | 0.630 +/- 0.047 | 0.639 +/- 0.039 | 0.592 +/- 0.050 | 0.592 +/- 0.050 | 0.584 +/- 0.056 |
| mode_stack (lightgbm) | 6 | 0.503 +/- 0.064 | 0.504 +/- 0.055 | 0.520 +/- 0.059 | 0.520 +/- 0.059 | 0.507 +/- 0.064 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | 0.631 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.628 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.627 | higher -> correct |
| reasoning_total_variation | thrashing | 0.607 | higher -> correct |
| reasoning_mid_mean | shape | 0.607 | higher -> correct |
| reasoning_max_rise | shape | 0.604 | higher -> correct |
| reasoning_negative_mass | shape | 0.586 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.585 | higher -> correct |
| reasoning_max_drop | derailment | 0.578 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.569 | higher -> correct |
| reasoning_time_positive | shape | 0.569 | higher -> correct |
| reasoning_time_negative | shape | 0.564 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | -0.687 | higher -> wrong |
| reasoning_positive_mass | shape | -0.574 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.498 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.422 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.371 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.288 | higher -> correct |
| reasoning_mid_mean | shape | -0.277 | higher -> wrong |
| reasoning_late_mean | landing | -0.265 | higher -> wrong |
| reasoning_negative_mass | shape | 0.207 | higher -> correct |
| reasoning_peak_pos | timing | 0.189 | higher -> correct |
| reasoning_late_minus_early | transition | -0.172 | higher -> wrong |
| reasoning_max_rise | shape | -0.164 | higher -> wrong |

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
| premature_exit | inverted | 400 | 0.368 [0.311, 0.418] | 0.500 | 11 | - | - | - | - |
| rambling_overlong | confirmed | 400 | 0.632 [0.582, 0.689] | 0.500 | 40 | 0.650 | 0.130 | 0.217 | 1.300 |
| thrashing | confirmed | 400 | 0.627 [0.573, 0.677] | 0.500 | 29 | 0.586 | 0.085 | 0.148 | 1.172 |
| no_commitment | inconclusive | 400 | 0.546 [0.493, 0.599] | 0.500 | 37 | - | - | - | - |
| derailment_late | inconclusive | 400 | 0.535 [0.480, 0.587] | 0.500 | 46 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.630**
- `trajectory_full` ROC-AUC: **0.632**
- Above-chance discrimination preserved by the mode stack: **97.9%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 2 | 28 / 200 | 47 | 0.140 | 0.596 |
| any | 11 | 68 / 200 | 129 | 0.340 | 0.527 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
