# SH6 swe-agent-trajectories/model-all/by-generator/swe-agent-llama-405b — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 236
- Positive class (`final_answer_correct=true`): 106
- Negative class (`final_answer_correct=false`): 130
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 236 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Status

- The `lenght_abort` baseline includes both chunk-count features and `truncation_abort_score` on this run.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| lenght_abort (logreg) | 2 | 0.753 +/- 0.096 | 0.678 +/- 0.093 | 0.723 +/- 0.054 | 0.704 +/- 0.058 | 0.736 +/- 0.043 |
| trajectory_shape (logreg) | 29 | 0.717 +/- 0.043 | 0.676 +/- 0.046 | 0.665 +/- 0.060 | 0.665 +/- 0.061 | 0.640 +/- 0.070 |
| trajectory_full (logreg) | 29 | 0.718 +/- 0.043 | 0.679 +/- 0.047 | 0.665 +/- 0.060 | 0.665 +/- 0.061 | 0.640 +/- 0.070 |
| trajectory_full (lightgbm) | 29 | 0.758 +/- 0.045 | 0.711 +/- 0.057 | 0.723 +/- 0.043 | 0.724 +/- 0.044 | 0.698 +/- 0.045 |
| mode_stack (logreg) | 7 | 0.740 +/- 0.042 | 0.633 +/- 0.051 | 0.722 +/- 0.061 | 0.704 +/- 0.064 | 0.734 +/- 0.050 |
| mode_stack (lightgbm) | 7 | 0.827 +/- 0.034 | 0.792 +/- 0.039 | 0.789 +/- 0.035 | 0.788 +/- 0.035 | 0.771 +/- 0.042 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_mid_mean | shape | 0.595 | higher -> correct |
| reasoning_range | commitment | 0.576 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.574 | higher -> correct |
| reasoning_max_rise | shape | 0.572 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.568 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.567 | higher -> wrong |
| reasoning_end | landing | 0.564 | higher -> wrong |
| reasoning_start | shape | 0.562 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.556 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.556 | higher -> correct |
| reasoning_early_mean | shape | 0.550 | higher -> correct |
| reasoning_peak_pos | timing | 0.549 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_zero_crossings | thrashing | -1.311 | higher -> wrong |
| reasoning_start | shape | -1.090 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 1.044 | higher -> correct |
| reasoning_max_rise | shape | -0.899 | higher -> wrong |
| reasoning_mid_mean | shape | -0.788 | higher -> wrong |
| reasoning_max | shape | -0.773 | higher -> wrong |
| reasoning_std | shape | -0.738 | higher -> wrong |
| reasoning_min | shape | -0.620 | higher -> wrong |
| reasoning_end | landing | -0.606 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.564 | higher -> correct |
| reasoning_late_mean | landing | 0.539 | higher -> correct |
| reasoning_max_drop | derailment | -0.523 | higher -> wrong |

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
| premature_exit | inconclusive | 236 | 0.529 [0.458, 0.597] | 0.551 | 13 | - | - | - | - |
| rambling_overlong | inconclusive | 236 | 0.471 [0.403, 0.542] | 0.551 | 15 | - | - | - | - |
| thrashing | inconclusive | 236 | 0.503 [0.429, 0.576] | 0.551 | 15 | - | - | - | - |
| no_commitment | inconclusive | 236 | 0.540 [0.464, 0.617] | 0.551 | 14 | - | - | - | - |
| derailment_late | inconclusive | 236 | 0.509 [0.433, 0.585] | 0.551 | 15 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 236 | 0.723 [0.673, 0.775] | 0.551 | 78 | 0.885 | 0.531 | 0.663 | 1.606 |

### Verdict summary

- **Confirmed on this run**: truncation_abort.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.740**
- `trajectory_full` ROC-AUC: **0.718**
- Above-chance discrimination preserved by the mode stack: **109.7%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 69 / 130 | 78 | 0.531 | 0.885 |
| any | 11 | 87 / 130 | 125 | 0.669 | 0.696 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
