# SH6 processbench/gsm8k/by-generator/Qwen2.5-Math-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 22
- Positive class (`final_answer_correct=true`): 16
- Negative class (`final_answer_correct=false`): 6
- Label agreement (`is_correct` vs `final_answer_correct`): 90.9% over 22 items
- Final answer correct but reasoning wrong: 2
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.217 +/- 0.125 | 0.706 +/- 0.057 | 0.183 +/- 0.033 | 0.270 +/- 0.068 | 0.421 +/- 0.080 |
| trajectory_shape (logreg) | 47 | 0.692 +/- 0.372 | 0.889 +/- 0.133 | 0.683 +/- 0.200 | 0.780 +/- 0.129 | 0.843 +/- 0.109 |
| trajectory_full (logreg) | 47 | 0.692 +/- 0.372 | 0.889 +/- 0.133 | 0.683 +/- 0.200 | 0.780 +/- 0.129 | 0.843 +/- 0.109 |
| reasoning_traj (MiniRocket) | 20 | 0.625 +/- 0.352 | 0.870 +/- 0.122 | 0.675 +/- 0.269 | 0.790 +/- 0.180 | 0.871 +/- 0.112 |
| trajectory_full (lightgbm) | 47 | 0.500 +/- 0.000 | 0.730 +/- 0.068 | 0.500 +/- 0.000 | 0.570 +/- 0.229 | 0.514 +/- 0.420 |
| mode_stack (logreg) | 6 | 0.592 +/- 0.388 | 0.845 +/- 0.151 | 0.467 +/- 0.221 | 0.500 +/- 0.247 | 0.538 +/- 0.329 |
| mode_stack (lightgbm) | 6 | 0.500 +/- 0.000 | 0.730 +/- 0.068 | 0.500 +/- 0.000 | 0.570 +/- 0.229 | 0.514 +/- 0.420 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t14 | shape | 0.825 | higher -> correct |
| reasoning_traj_t15 | shape | 0.825 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.817 | higher -> correct |
| reasoning_traj_t18 | shape | 0.792 | higher -> wrong |
| reasoning_n_chunks | length | 0.783 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.775 | higher -> correct |
| reasoning_trough_pos | timing | 0.775 | higher -> correct |
| reasoning_time_positive | shape | 0.775 | higher -> wrong |
| reasoning_traj_t07 | shape | 0.775 | higher -> correct |
| reasoning_traj_t09 | shape | 0.775 | higher -> wrong |
| reasoning_traj_t04 | shape | 0.758 | higher -> wrong |
| reasoning_traj_t16 | shape | 0.758 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | 0.691 | higher -> correct |
| reasoning_max_drop | derailment | -0.652 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.499 | higher -> wrong |
| reasoning_traj_t15 | shape | 0.417 | higher -> correct |
| reasoning_n_chunks | length | 0.414 | higher -> correct |
| reasoning_traj_t16 | shape | 0.363 | higher -> correct |
| reasoning_traj_t14 | shape | 0.347 | higher -> correct |
| reasoning_max_rise | shape | 0.330 | higher -> correct |
| reasoning_traj_t08 | shape | -0.317 | higher -> wrong |
| reasoning_time_positive | shape | 0.292 | higher -> correct |
| reasoning_min | shape | 0.247 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.236 | higher -> wrong |

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
| premature_exit | inconclusive | 22 | 0.427 [0.190, 0.684] | 0.273 | 1 | - | - | - | - |
| rambling_overlong | inconclusive | 22 | 0.573 [0.316, 0.810] | 0.273 | 2 | - | - | - | - |
| thrashing | inconclusive | 22 | 0.667 [0.396, 0.906] | 0.273 | 3 | - | - | - | - |
| no_commitment | inconclusive | 22 | 0.536 [0.256, 0.800] | 0.273 | 2 | - | - | - | - |
| derailment_late | inconclusive | 22 | 0.542 [0.295, 0.785] | 0.273 | 3 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.592**
- `trajectory_full` ROC-AUC: **0.692**
- Above-chance discrimination preserved by the mode stack: **47.8%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 6 | 0 | - | - |
| any | 11 | 2 / 6 | 7 | 0.333 | 0.286 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
