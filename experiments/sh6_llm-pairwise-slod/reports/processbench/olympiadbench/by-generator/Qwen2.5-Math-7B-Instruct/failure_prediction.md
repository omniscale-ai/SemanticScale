# SH6 processbench/olympiadbench/by-generator/Qwen2.5-Math-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 54
- Positive class (`final_answer_correct=true`): 24
- Negative class (`final_answer_correct=false`): 30
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 54 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.666 +/- 0.099 | 0.632 +/- 0.053 | 0.577 +/- 0.084 | 0.595 +/- 0.086 | 0.473 +/- 0.124 |
| trajectory_shape (logreg) | 29 | 0.497 +/- 0.180 | 0.533 +/- 0.133 | 0.563 +/- 0.113 | 0.558 +/- 0.098 | 0.515 +/- 0.121 |
| trajectory_full (logreg) | 29 | 0.497 +/- 0.180 | 0.533 +/- 0.133 | 0.563 +/- 0.113 | 0.558 +/- 0.098 | 0.515 +/- 0.121 |
| trajectory_full (lightgbm) | 29 | 0.442 +/- 0.269 | 0.527 +/- 0.184 | 0.450 +/- 0.223 | 0.449 +/- 0.208 | 0.368 +/- 0.244 |
| mode_stack (logreg) | 6 | 0.585 +/- 0.174 | 0.645 +/- 0.137 | 0.535 +/- 0.128 | 0.538 +/- 0.124 | 0.467 +/- 0.157 |
| mode_stack (lightgbm) | 6 | 0.518 +/- 0.138 | 0.596 +/- 0.133 | 0.513 +/- 0.113 | 0.518 +/- 0.106 | 0.429 +/- 0.223 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_mid_mean | shape | 0.687 | higher -> wrong |
| reasoning_late_mean | landing | 0.670 | higher -> correct |
| reasoning_n_chunks | length | 0.666 | higher -> correct |
| reasoning_max_drop | derailment | 0.665 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.665 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.664 | higher -> correct |
| reasoning_trough_pos | timing | 0.658 | higher -> correct |
| reasoning_max_rise | shape | 0.657 | higher -> correct |
| reasoning_peak_pos | timing | 0.653 | higher -> correct |
| reasoning_late_minus_early | transition | 0.643 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.642 | higher -> correct |
| reasoning_end | landing | 0.620 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_negative_mass | shape | -1.025 | higher -> wrong |
| reasoning_max_drop_pos | derailment | -0.711 | higher -> wrong |
| reasoning_max_drop | derailment | 0.641 | higher -> correct |
| reasoning_monotonicity | commitment | 0.620 | higher -> correct |
| reasoning_max_rise | shape | 0.565 | higher -> correct |
| reasoning_early_mean | shape | 0.486 | higher -> correct |
| reasoning_peak_pos | timing | 0.448 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.422 | higher -> wrong |
| reasoning_n_chunks | length | -0.376 | higher -> wrong |
| reasoning_late_mean | landing | 0.374 | higher -> correct |
| reasoning_zero_crossings | thrashing | -0.158 | higher -> wrong |
| reasoning_std | shape | -0.152 | higher -> wrong |

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
| premature_exit | confirmed | 54 | 0.660 [0.510, 0.800] | 0.556 | 13 | 0.769 | 0.333 | 0.465 | 1.385 |
| rambling_overlong | inverted | 54 | 0.340 [0.200, 0.490] | 0.556 | 6 | - | - | - | - |
| thrashing | inverted | 54 | 0.340 [0.197, 0.481] | 0.556 | 6 | - | - | - | - |
| no_commitment | inconclusive | 54 | 0.440 [0.276, 0.596] | 0.556 | 3 | - | - | - | - |
| derailment_late | inconclusive | 54 | 0.537 [0.381, 0.689] | 0.556 | 6 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: premature_exit.
- **Hypothesis falsified (inverted)**: rambling_overlong, thrashing. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.585**
- `trajectory_full` ROC-AUC: **0.497**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 10 / 30 | 13 | 0.333 | 0.769 |
| any | 11 | 15 / 30 | 27 | 0.500 | 0.556 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
