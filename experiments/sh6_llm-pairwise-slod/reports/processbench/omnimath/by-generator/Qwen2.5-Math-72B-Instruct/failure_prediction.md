# SH6 processbench/omnimath/by-generator/Qwen2.5-Math-72B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 45
- Positive class (`final_answer_correct=true`): 26
- Negative class (`final_answer_correct=false`): 19
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 45 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.458 +/- 0.144 | 0.613 +/- 0.121 | 0.485 +/- 0.118 | 0.511 +/- 0.113 | 0.608 +/- 0.082 |
| trajectory_shape (logreg) | 47 | 0.644 +/- 0.155 | 0.773 +/- 0.132 | 0.538 +/- 0.139 | 0.533 +/- 0.130 | 0.549 +/- 0.117 |
| trajectory_full (logreg) | 47 | 0.644 +/- 0.155 | 0.773 +/- 0.132 | 0.538 +/- 0.139 | 0.533 +/- 0.130 | 0.549 +/- 0.117 |
| reasoning_traj (MiniRocket) | 20 | 0.339 +/- 0.190 | 0.605 +/- 0.149 | 0.352 +/- 0.118 | 0.378 +/- 0.113 | 0.448 +/- 0.132 |
| trajectory_full (lightgbm) | 47 | 0.471 +/- 0.181 | 0.632 +/- 0.127 | 0.513 +/- 0.166 | 0.511 +/- 0.166 | 0.532 +/- 0.181 |
| mode_stack (logreg) | 6 | 0.587 +/- 0.160 | 0.729 +/- 0.131 | 0.555 +/- 0.149 | 0.556 +/- 0.157 | 0.557 +/- 0.194 |
| mode_stack (lightgbm) | 6 | 0.424 +/- 0.122 | 0.635 +/- 0.127 | 0.443 +/- 0.159 | 0.444 +/- 0.141 | 0.474 +/- 0.076 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t18 | shape | 0.697 | higher -> correct |
| reasoning_traj_t12 | shape | 0.693 | higher -> wrong |
| reasoning_traj_t01 | shape | 0.687 | higher -> wrong |
| reasoning_rebound_from_trough | shape | 0.679 | higher -> correct |
| reasoning_max_rise | shape | 0.678 | higher -> correct |
| reasoning_traj_t17 | shape | 0.672 | higher -> correct |
| reasoning_traj_t11 | shape | 0.661 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.661 | higher -> wrong |
| reasoning_end | landing | 0.656 | higher -> correct |
| reasoning_peak_pos | timing | 0.652 | higher -> correct |
| reasoning_traj_t05 | shape | 0.650 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.648 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise | shape | 1.278 | higher -> correct |
| reasoning_max_drop | derailment | -0.901 | higher -> wrong |
| reasoning_traj_t03 | shape | -0.829 | higher -> wrong |
| reasoning_traj_t05 | shape | 0.757 | higher -> correct |
| reasoning_max_rise_pos | timing | -0.746 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.694 | higher -> correct |
| reasoning_traj_t15 | shape | -0.604 | higher -> wrong |
| reasoning_traj_t07 | shape | -0.567 | higher -> wrong |
| reasoning_traj_t17 | shape | 0.554 | higher -> correct |
| reasoning_max | shape | 0.514 | higher -> correct |
| reasoning_traj_t12 | shape | 0.457 | higher -> correct |
| reasoning_peak_pos | timing | 0.408 | higher -> correct |

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
| premature_exit | inconclusive | 45 | 0.466 [0.297, 0.645] | 0.422 | 1 | - | - | - | - |
| rambling_overlong | inconclusive | 45 | 0.534 [0.355, 0.703] | 0.422 | 7 | - | - | - | - |
| thrashing | inconclusive | 45 | 0.530 [0.368, 0.699] | 0.422 | 3 | - | - | - | - |
| no_commitment | inconclusive | 45 | 0.478 [0.310, 0.656] | 0.422 | 3 | - | - | - | - |
| derailment_late | inconclusive | 45 | 0.611 [0.428, 0.776] | 0.422 | 10 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.587**
- `trajectory_full` ROC-AUC: **0.644**
- Above-chance discrimination preserved by the mode stack: **60.0%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 19 | 0 | - | - |
| any | 11 | 12 / 19 | 22 | 0.632 | 0.545 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
