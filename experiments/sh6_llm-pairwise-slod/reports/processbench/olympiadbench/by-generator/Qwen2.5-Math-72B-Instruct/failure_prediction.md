# SH6 processbench/olympiadbench/by-generator/Qwen2.5-Math-72B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 47
- Positive class (`final_answer_correct=true`): 20
- Negative class (`final_answer_correct=false`): 27
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 47 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.626 +/- 0.070 | 0.589 +/- 0.100 | 0.560 +/- 0.105 | 0.571 +/- 0.108 | 0.494 +/- 0.128 |
| trajectory_shape (logreg) | 47 | 0.425 +/- 0.069 | 0.487 +/- 0.075 | 0.510 +/- 0.147 | 0.513 +/- 0.136 | 0.449 +/- 0.185 |
| trajectory_full (logreg) | 47 | 0.425 +/- 0.069 | 0.487 +/- 0.075 | 0.510 +/- 0.147 | 0.513 +/- 0.136 | 0.449 +/- 0.185 |
| reasoning_traj (MiniRocket) | 20 | 0.652 +/- 0.157 | 0.699 +/- 0.146 | 0.567 +/- 0.147 | 0.571 +/- 0.129 | 0.453 +/- 0.258 |
| trajectory_full (lightgbm) | 47 | 0.407 +/- 0.231 | 0.514 +/- 0.185 | 0.430 +/- 0.143 | 0.431 +/- 0.152 | 0.381 +/- 0.136 |
| mode_stack (logreg) | 6 | 0.610 +/- 0.043 | 0.621 +/- 0.070 | 0.658 +/- 0.049 | 0.658 +/- 0.053 | 0.615 +/- 0.043 |
| mode_stack (lightgbm) | 6 | 0.611 +/- 0.202 | 0.580 +/- 0.199 | 0.507 +/- 0.118 | 0.516 +/- 0.124 | 0.373 +/- 0.213 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | 0.672 | higher -> wrong |
| reasoning_range | commitment | 0.668 | higher -> correct |
| reasoning_traj_t04 | shape | 0.668 | higher -> wrong |
| reasoning_traj_t18 | shape | 0.660 | higher -> wrong |
| reasoning_max | shape | 0.653 | higher -> correct |
| reasoning_min | shape | 0.652 | higher -> correct |
| reasoning_traj_t02 | shape | 0.652 | higher -> wrong |
| reasoning_std | shape | 0.650 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.647 | higher -> correct |
| reasoning_positive_mass | shape | 0.645 | higher -> wrong |
| reasoning_negative_mass | shape | 0.643 | higher -> wrong |
| reasoning_traj_t17 | shape | 0.640 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_drop | derailment | 1.380 | higher -> correct |
| reasoning_positive_mass | shape | 0.793 | higher -> correct |
| reasoning_peak_pos | timing | 0.729 | higher -> correct |
| reasoning_monotonicity | commitment | 0.665 | higher -> correct |
| reasoning_min | shape | 0.664 | higher -> correct |
| reasoning_range | commitment | -0.647 | higher -> wrong |
| reasoning_traj_t11 | shape | 0.520 | higher -> correct |
| reasoning_traj_t07 | shape | 0.473 | higher -> correct |
| reasoning_traj_t04 | shape | 0.457 | higher -> correct |
| reasoning_max | shape | -0.404 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -0.404 | higher -> wrong |
| reasoning_traj_t05 | shape | -0.400 | higher -> wrong |

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
| premature_exit | inconclusive | 47 | 0.648 [0.483, 0.809] | 0.574 | 10 | - | - | - | - |
| rambling_overlong | inconclusive | 47 | 0.352 [0.191, 0.517] | 0.574 | 3 | - | - | - | - |
| thrashing | inconclusive | 47 | 0.377 [0.215, 0.550] | 0.574 | 0 | - | - | - | - |
| no_commitment | inconclusive | 47 | 0.452 [0.286, 0.632] | 0.574 | 4 | - | - | - | - |
| derailment_late | inconclusive | 47 | 0.621 [0.429, 0.788] | 0.574 | 4 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.610**
- `trajectory_full` ROC-AUC: **0.425**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 27 | 0 | - | - |
| any | 11 | 11 / 27 | 18 | 0.407 | 0.611 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
