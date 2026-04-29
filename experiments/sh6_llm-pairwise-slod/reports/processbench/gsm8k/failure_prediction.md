# SH6 processbench-gsm8k/gsm8k — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 400
- Positive class (`final_answer_correct=true`): 200
- Negative class (`final_answer_correct=false`): 200
- Label agreement (`is_correct` vs `final_answer_correct`): 98.2% over 400 items
- Final answer correct but reasoning wrong: 7
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.544 +/- 0.080 | 0.539 +/- 0.051 | 0.520 +/- 0.074 | 0.520 +/- 0.074 | 0.579 +/- 0.063 |
| trajectory_shape (logreg) | 29 | 0.504 +/- 0.050 | 0.518 +/- 0.054 | 0.498 +/- 0.044 | 0.498 +/- 0.044 | 0.529 +/- 0.036 |
| trajectory_full (logreg) | 29 | 0.504 +/- 0.050 | 0.518 +/- 0.054 | 0.498 +/- 0.044 | 0.498 +/- 0.044 | 0.529 +/- 0.036 |
| trajectory_full (lightgbm) | 29 | 0.464 +/- 0.039 | 0.504 +/- 0.037 | 0.485 +/- 0.054 | 0.485 +/- 0.054 | 0.487 +/- 0.056 |
| mode_stack (logreg) | 6 | 0.483 +/- 0.066 | 0.503 +/- 0.054 | 0.487 +/- 0.054 | 0.487 +/- 0.054 | 0.509 +/- 0.046 |
| mode_stack (lightgbm) | 6 | 0.474 +/- 0.055 | 0.509 +/- 0.055 | 0.503 +/- 0.055 | 0.503 +/- 0.055 | 0.507 +/- 0.036 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_start | shape | 0.562 | higher -> correct |
| reasoning_negative_mass | shape | 0.558 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.552 | higher -> wrong |
| reasoning_range | commitment | 0.551 | higher -> correct |
| reasoning_n_chunks | length | 0.544 | higher -> correct |
| reasoning_monotonicity | commitment | 0.542 | higher -> wrong |
| reasoning_std | shape | 0.542 | higher -> correct |
| reasoning_end | landing | 0.542 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.540 | higher -> correct |
| reasoning_max_rise | shape | 0.540 | higher -> correct |
| reasoning_max_drop | derailment | 0.539 | higher -> wrong |
| reasoning_mid_mean | shape | 0.538 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_std | shape | 0.721 | higher -> correct |
| reasoning_start | shape | 0.529 | higher -> correct |
| reasoning_positive_mass | shape | -0.489 | higher -> wrong |
| reasoning_n_chunks | length | -0.453 | higher -> wrong |
| reasoning_peak_pos | timing | 0.369 | higher -> correct |
| reasoning_negative_mass | shape | -0.345 | higher -> wrong |
| reasoning_end_minus_start | landing | -0.285 | higher -> wrong |
| reasoning_max_rise | shape | 0.281 | higher -> correct |
| reasoning_trough_pos | timing | -0.250 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.211 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.138 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.137 | higher -> correct |

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
| premature_exit | inconclusive | 400 | 0.455 [0.401, 0.507] | 0.500 | 39 | - | - | - | - |
| rambling_overlong | inconclusive | 400 | 0.545 [0.493, 0.599] | 0.500 | 34 | - | - | - | - |
| thrashing | inconclusive | 400 | 0.531 [0.475, 0.583] | 0.500 | 32 | - | - | - | - |
| no_commitment | inconclusive | 400 | 0.534 [0.478, 0.590] | 0.500 | 38 | - | - | - | - |
| derailment_late | inconclusive | 400 | 0.494 [0.433, 0.551] | 0.500 | 43 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.483**
- `trajectory_full` ROC-AUC: **0.504**
- Above-chance discrimination preserved by the mode stack: **-373.2%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 200 | 0 | - | - |
| any | 11 | 79 / 200 | 154 | 0.395 | 0.513 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
