# SH6 agenthallu/framework-all — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 693
- Positive class (`final_answer_correct=true`): 250
- Negative class (`final_answer_correct=false`): 443
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 693 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.528 +/- 0.076 | 0.399 +/- 0.056 | 0.525 +/- 0.050 | 0.531 +/- 0.065 | 0.438 +/- 0.041 |
| trajectory_shape (logreg) | 60 | 0.505 +/- 0.024 | 0.375 +/- 0.019 | 0.502 +/- 0.025 | 0.492 +/- 0.034 | 0.432 +/- 0.025 |
| trajectory_full (logreg) | 63 | 0.504 +/- 0.025 | 0.371 +/- 0.016 | 0.503 +/- 0.024 | 0.496 +/- 0.036 | 0.430 +/- 0.023 |
| trajectory_full (lightgbm) | 63 | 0.512 +/- 0.058 | 0.375 +/- 0.061 | 0.487 +/- 0.055 | 0.550 +/- 0.053 | 0.294 +/- 0.079 |
| mode_stack (logreg) | 13 | 0.494 +/- 0.041 | 0.368 +/- 0.025 | 0.495 +/- 0.033 | 0.469 +/- 0.036 | 0.444 +/- 0.030 |
| mode_stack (lightgbm) | 13 | 0.466 +/- 0.039 | 0.352 +/- 0.011 | 0.459 +/- 0.039 | 0.511 +/- 0.038 | 0.286 +/- 0.051 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | 0.552 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.551 | higher -> wrong |
| answer_monotonicity | commitment | 0.547 | higher -> correct |
| reasoning_late_minus_early | transition | 0.546 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.546 | higher -> wrong |
| answer_time_positive | shape | 0.543 | higher -> correct |
| reasoning_max_rise | shape | 0.543 | higher -> wrong |
| answer_positive_mass | shape | 0.540 | higher -> correct |
| reasoning_end_minus_start | landing | 0.540 | higher -> wrong |
| answer_max | shape | 0.540 | higher -> correct |
| answer_range | commitment | 0.540 | higher -> correct |
| answer_time_negative | shape | 0.539 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_std | shape | 0.455 | higher -> correct |
| answer_total_variation | thrashing | -0.422 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | -0.324 | higher -> wrong |
| answer_peak_pos | timing | -0.323 | higher -> wrong |
| reasoning_max_rise | shape | -0.314 | higher -> wrong |
| reasoning_negative_mass | shape | -0.300 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.281 | higher -> correct |
| reasoning_monotonicity | commitment | -0.259 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.253 | higher -> correct |
| answer_end_minus_reasoning_end | landing | 0.237 | higher -> correct |
| reasoning_zero_crossings | thrashing | -0.224 | higher -> wrong |
| answer_std | shape | 0.214 | higher -> correct |

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
| premature_exit | inconclusive | 693 | 0.531 [0.486, 0.575] | 0.639 | 87 | - | - | - | - |
| rambling_overlong | inconclusive | 693 | 0.469 [0.425, 0.514] | 0.639 | 44 | - | - | - | - |
| thrashing | inconclusive | 693 | 0.481 [0.436, 0.526] | 0.639 | 57 | - | - | - | - |
| no_commitment | inconclusive | 693 | 0.493 [0.451, 0.535] | 0.639 | 69 | - | - | - | - |
| derailment_late | inconclusive | 693 | 0.518 [0.469, 0.561] | 0.639 | 83 | - | - | - | - |
| answer_drift | inconclusive | 504 | 0.524 [0.471, 0.576] | 0.641 | 64 | - | - | - | - |
| answer_meandering | inconclusive | 504 | 0.527 [0.480, 0.571] | 0.641 | 35 | - | - | - | - |
| answer_volatility | inconclusive | 504 | 0.524 [0.476, 0.572] | 0.641 | 51 | - | - | - | - |
| answer_uncommitted | inverted | 504 | 0.432 [0.386, 0.479] | 0.641 | 0 | - | - | - | - |
| answer_overrange | inconclusive | 504 | 0.522 [0.472, 0.578] | 0.641 | 64 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Hypothesis falsified (inverted)**: answer_uncommitted. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.494**
- `trajectory_full` ROC-AUC: **0.504**
- Above-chance discrimination preserved by the mode stack: **-153.9%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 443 | 0 | - | - |
| any | 11 | 219 / 443 | 333 | 0.494 | 0.658 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
