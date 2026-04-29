# SH6 swe-agent-trajectories/model-all/by-generator/swe-agent-llama-70b — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 518
- Positive class (`final_answer_correct=true`): 156
- Negative class (`final_answer_correct=false`): 362
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 518 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.744 +/- 0.036 | 0.541 +/- 0.040 | 0.698 +/- 0.026 | 0.647 +/- 0.033 | 0.585 +/- 0.028 |
| trajectory_shape | 29 | 0.901 +/- 0.022 | 0.800 +/- 0.046 | 0.832 +/- 0.027 | 0.817 +/- 0.030 | 0.742 +/- 0.037 |
| trajectory_full | 29 | 0.900 +/- 0.023 | 0.797 +/- 0.047 | 0.832 +/- 0.027 | 0.817 +/- 0.030 | 0.742 +/- 0.037 |
| mode_stack | 6 | 0.816 +/- 0.031 | 0.669 +/- 0.047 | 0.733 +/- 0.039 | 0.724 +/- 0.041 | 0.623 +/- 0.047 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | 0.787 | higher -> correct |
| reasoning_std | shape | 0.763 | higher -> correct |
| reasoning_monotonicity | commitment | 0.757 | higher -> correct |
| reasoning_min | shape | 0.754 | higher -> correct |
| reasoning_negative_mass | shape | 0.752 | higher -> correct |
| reasoning_positive_mass | shape | 0.751 | higher -> correct |
| reasoning_n_chunks | length | 0.744 | higher -> correct |
| reasoning_range | commitment | 0.735 | higher -> correct |
| reasoning_max | shape | 0.720 | higher -> correct |
| reasoning_late_minus_early | transition | 0.711 | higher -> correct |
| reasoning_late_mean | landing | 0.704 | higher -> correct |
| reasoning_max_drop | derailment | 0.698 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | -2.281 | higher -> wrong |
| reasoning_max | shape | -2.158 | higher -> wrong |
| reasoning_peak_pos | timing | 1.557 | higher -> correct |
| reasoning_max_rise_pos | timing | -1.554 | higher -> wrong |
| reasoning_end_minus_start | landing | -1.355 | higher -> wrong |
| reasoning_late_mean | landing | 1.256 | higher -> correct |
| reasoning_min | shape | -1.201 | higher -> wrong |
| reasoning_start | shape | 1.115 | higher -> correct |
| reasoning_end | landing | -0.936 | higher -> wrong |
| reasoning_fall_from_peak | derailment | -0.812 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.688 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.676 | higher -> wrong |

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
| premature_exit | inverted | 518 | 0.255 [0.202, 0.309] | 0.699 | 8 | - | - | - | - |
| rambling_overlong | confirmed | 518 | 0.745 [0.691, 0.798] | 0.699 | 42 | 0.881 | 0.102 | 0.183 | 1.261 |
| thrashing | confirmed | 518 | 0.786 [0.744, 0.825] | 0.699 | 147 | 0.966 | 0.392 | 0.558 | 1.382 |
| no_commitment | confirmed | 518 | 0.757 [0.708, 0.798] | 0.699 | 157 | 0.904 | 0.392 | 0.547 | 1.294 |
| derailment_late | inconclusive | 518 | 0.465 [0.408, 0.525] | 0.699 | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 518 | 0.623 [0.595, 0.649] | 0.699 | 106 | 0.953 | 0.279 | 0.432 | 1.363 |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, no_commitment, truncation_abort.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.816**
- `trajectory_full` ROC-AUC: **0.900**
- Above-chance discrimination preserved by the mode stack: **78.9%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 202 / 362 | 222 | 0.558 | 0.910 |
| any | 11 | 207 / 362 | 227 | 0.572 | 0.912 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
