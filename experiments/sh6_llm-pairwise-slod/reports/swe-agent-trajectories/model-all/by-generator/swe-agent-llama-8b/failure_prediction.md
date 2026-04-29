# SH6 swe-agent-trajectories/model-all/by-generator/swe-agent-llama-8b — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 97
- Positive class (`final_answer_correct=true`): 29
- Negative class (`final_answer_correct=false`): 68
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 97 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.717 +/- 0.088 | 0.495 +/- 0.093 | 0.649 +/- 0.081 | 0.598 +/- 0.090 | 0.535 +/- 0.069 |
| trajectory_shape | 29 | 0.731 +/- 0.098 | 0.497 +/- 0.098 | 0.721 +/- 0.096 | 0.712 +/- 0.094 | 0.604 +/- 0.112 |
| trajectory_full | 29 | 0.741 +/- 0.100 | 0.506 +/- 0.104 | 0.721 +/- 0.096 | 0.712 +/- 0.094 | 0.604 +/- 0.112 |
| mode_stack | 6 | 0.871 +/- 0.102 | 0.773 +/- 0.164 | 0.854 +/- 0.091 | 0.846 +/- 0.104 | 0.783 +/- 0.110 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_end | landing | 0.755 | higher -> correct |
| reasoning_time_positive | shape | 0.740 | higher -> correct |
| reasoning_time_negative | shape | 0.740 | higher -> correct |
| reasoning_late_mean | landing | 0.729 | higher -> correct |
| reasoning_late_minus_early | transition | 0.727 | higher -> correct |
| reasoning_peak_pos | timing | 0.725 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.721 | higher -> correct |
| reasoning_max | shape | 0.720 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.719 | higher -> correct |
| reasoning_max_rise | shape | 0.719 | higher -> correct |
| reasoning_end_minus_start | landing | 0.718 | higher -> correct |
| reasoning_positive_mass | shape | 0.717 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_peak_pos | timing | 0.422 | higher -> correct |
| reasoning_rebound_from_trough | shape | -0.286 | higher -> wrong |
| reasoning_end | landing | -0.253 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.250 | higher -> correct |
| reasoning_mid_mean | shape | -0.207 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.182 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.177 | higher -> correct |
| reasoning_late_mean | landing | 0.176 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.142 | higher -> correct |
| reasoning_late_minus_early | transition | 0.137 | higher -> correct |
| reasoning_time_positive | shape | -0.135 | higher -> wrong |
| reasoning_time_negative | shape | 0.135 | higher -> correct |

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
| premature_exit | inverted | 97 | 0.288 [0.183, 0.401] | 0.701 | 0 | - | - | - | - |
| rambling_overlong | confirmed | 97 | 0.712 [0.599, 0.817] | 0.701 | 5 | 0.800 | 0.059 | 0.110 | 1.141 |
| thrashing | confirmed | 97 | 0.709 [0.595, 0.816] | 0.701 | 5 | 0.800 | 0.059 | 0.110 | 1.141 |
| no_commitment | inconclusive | 97 | 0.452 [0.346, 0.566] | 0.701 | 20 | - | - | - | - |
| derailment_late | inverted | 97 | 0.288 [0.182, 0.401] | 0.701 | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 97 | 0.809 [0.729, 0.886] | 0.701 | 52 | 0.942 | 0.721 | 0.817 | 1.344 |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, truncation_abort.
- **Hypothesis falsified (inverted)**: premature_exit, derailment_late. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: no_commitment. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.871**
- `trajectory_full` ROC-AUC: **0.741**
- Above-chance discrimination preserved by the mode stack: **153.7%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 3 | 50 / 68 | 53 | 0.735 | 0.943 |
| any | 11 | 60 / 68 | 64 | 0.882 | 0.938 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
