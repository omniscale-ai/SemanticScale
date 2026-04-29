# SH6 swe-agent-trajectories/model-all/by-generator/swe-agent-llama-405b — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 30
- Positive class (`final_answer_correct=true`): 17
- Negative class (`final_answer_correct=false`): 13
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 30 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.800 +/- 0.138 | 0.818 +/- 0.137 | 0.800 +/- 0.138 | 0.800 +/- 0.125 | 0.815 +/- 0.112 |
| trajectory_shape | 29 | 0.811 +/- 0.129 | 0.825 +/- 0.127 | 0.800 +/- 0.138 | 0.800 +/- 0.125 | 0.815 +/- 0.112 |
| trajectory_full | 29 | 0.787 +/- 0.133 | 0.818 +/- 0.137 | 0.800 +/- 0.138 | 0.800 +/- 0.125 | 0.815 +/- 0.112 |
| mode_stack | 6 | 0.835 +/- 0.130 | 0.841 +/- 0.119 | 0.800 +/- 0.138 | 0.800 +/- 0.125 | 0.815 +/- 0.112 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | 0.800 | higher -> correct |
| reasoning_start | shape | 0.800 | higher -> correct |
| reasoning_end | landing | 0.800 | higher -> correct |
| reasoning_end_minus_start | landing | 0.800 | higher -> correct |
| reasoning_std | shape | 0.800 | higher -> correct |
| reasoning_range | commitment | 0.800 | higher -> correct |
| reasoning_min | shape | 0.800 | higher -> correct |
| reasoning_max | shape | 0.800 | higher -> correct |
| reasoning_early_mean | shape | 0.800 | higher -> correct |
| reasoning_mid_mean | shape | 0.800 | higher -> correct |
| reasoning_late_mean | landing | 0.800 | higher -> correct |
| reasoning_late_minus_early | transition | 0.800 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_time_negative | shape | 0.049 | higher -> correct |
| reasoning_time_positive | shape | -0.049 | higher -> wrong |
| reasoning_late_minus_early | transition | 0.049 | higher -> correct |
| reasoning_positive_mass | shape | 0.049 | higher -> correct |
| reasoning_std | shape | 0.049 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.049 | higher -> correct |
| reasoning_n_chunks | length | -0.049 | higher -> wrong |
| reasoning_start | shape | 0.049 | higher -> correct |
| reasoning_late_mean | landing | 0.049 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.049 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -0.049 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.049 | higher -> correct |

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
| premature_exit | inverted | 30 | 0.195 [0.068, 0.350] | 0.433 | 0 | - | - | - | - |
| rambling_overlong | confirmed | 30 | 0.805 [0.650, 0.932] | 0.433 | 0 | - | - | - | - |
| thrashing | confirmed | 30 | 0.805 [0.650, 0.932] | 0.433 | 0 | - | - | - | - |
| no_commitment | inverted | 30 | 0.195 [0.068, 0.350] | 0.433 | 0 | - | - | - | - |
| derailment_late | inverted | 30 | 0.195 [0.068, 0.350] | 0.433 | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 30 | 0.835 [0.687, 0.952] | 0.433 | 14 | 0.786 | 0.846 | 0.815 | 1.813 |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, truncation_abort.
- **Hypothesis falsified (inverted)**: premature_exit, no_commitment, derailment_late. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.835**
- `trajectory_full` ROC-AUC: **0.787**
- Above-chance discrimination preserved by the mode stack: **116.4%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 3 | 11 / 13 | 14 | 0.846 | 0.786 |
| any | 11 | 11 / 13 | 14 | 0.846 | 0.786 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
