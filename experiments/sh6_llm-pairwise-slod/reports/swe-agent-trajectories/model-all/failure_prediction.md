# SH6 swe-agent-trajectories/model-all — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 2017
- Positive class (`final_answer_correct=true`): 839
- Negative class (`final_answer_correct=false`): 1178
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 2017 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Status

- The `length_abort` baseline includes both chunk-count features and `truncation_abort_score` on this run.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_abort (logreg) | 2 | 0.675 +/- 0.015 | 0.535 +/- 0.016 | 0.647 +/- 0.013 | 0.601 +/- 0.012 | 0.657 +/- 0.014 |
| trajectory_shape (logreg) | 47 | 0.822 +/- 0.023 | 0.728 +/- 0.037 | 0.744 +/- 0.021 | 0.736 +/- 0.024 | 0.714 +/- 0.020 |
| trajectory_full (logreg) | 47 | 0.822 +/- 0.023 | 0.728 +/- 0.037 | 0.744 +/- 0.021 | 0.736 +/- 0.024 | 0.714 +/- 0.020 |
| reasoning_traj (MiniRocket) | 20 | 0.823 +/- 0.021 | 0.725 +/- 0.034 | 0.746 +/- 0.019 | 0.738 +/- 0.021 | 0.715 +/- 0.019 |
| trajectory_full (lightgbm) | 47 | 0.823 +/- 0.022 | 0.727 +/- 0.034 | 0.746 +/- 0.019 | 0.738 +/- 0.021 | 0.715 +/- 0.019 |
| mode_stack (logreg) | 7 | 0.697 +/- 0.017 | 0.579 +/- 0.022 | 0.673 +/- 0.013 | 0.637 +/- 0.012 | 0.670 +/- 0.014 |
| mode_stack (lightgbm) | 7 | 0.893 +/- 0.018 | 0.834 +/- 0.030 | 0.810 +/- 0.021 | 0.807 +/- 0.024 | 0.782 +/- 0.023 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t18 | shape | 0.657 | higher -> correct |
| reasoning_start | shape | 0.628 | higher -> correct |
| reasoning_late_mean | landing | 0.627 | higher -> correct |
| reasoning_traj_t17 | shape | 0.622 | higher -> correct |
| reasoning_late_minus_early | transition | 0.617 | higher -> correct |
| reasoning_early_mean | shape | 0.588 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.588 | higher -> correct |
| reasoning_end_minus_start | landing | 0.584 | higher -> correct |
| reasoning_traj_t09 | shape | 0.583 | higher -> correct |
| reasoning_traj_t06 | shape | 0.581 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.576 | higher -> correct |
| reasoning_traj_t15 | shape | 0.576 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | -2.445 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -2.346 | higher -> wrong |
| reasoning_total_variation | thrashing | 1.955 | higher -> correct |
| reasoning_traj_t05 | shape | -1.895 | higher -> wrong |
| reasoning_traj_t02 | shape | 1.755 | higher -> correct |
| reasoning_max_drop_pos | derailment | -1.515 | higher -> wrong |
| reasoning_max_rise | shape | -1.289 | higher -> wrong |
| reasoning_std | shape | -1.268 | higher -> wrong |
| reasoning_traj_t03 | shape | -1.256 | higher -> wrong |
| reasoning_start | shape | -1.252 | higher -> wrong |
| reasoning_monotonicity | commitment | -1.181 | higher -> wrong |
| reasoning_traj_t07 | shape | -1.082 | higher -> wrong |

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
| premature_exit | inverted | 2017 | 0.430 [0.404, 0.456] | 0.584 | 120 | - | - | - | - |
| rambling_overlong | confirmed | 2017 | 0.570 [0.544, 0.596] | 0.584 | 262 | 0.695 | 0.154 | 0.253 | 1.189 |
| thrashing | confirmed | 2017 | 0.577 [0.553, 0.603] | 0.584 | 241 | 0.722 | 0.148 | 0.245 | 1.236 |
| no_commitment | confirmed | 2017 | 0.541 [0.515, 0.567] | 0.584 | 196 | 0.638 | 0.106 | 0.182 | 1.092 |
| derailment_late | inconclusive | 2017 | 0.500 [0.473, 0.526] | 0.584 | 116 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 2017 | 0.648 [0.632, 0.664] | 0.584 | 469 | 0.893 | 0.356 | 0.509 | 1.530 |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, no_commitment, truncation_abort.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.697**
- `trajectory_full` ROC-AUC: **0.822**
- Above-chance discrimination preserved by the mode stack: **61.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 636 / 1178 | 878 | 0.540 | 0.724 |
| any | 11 | 729 / 1178 | 1081 | 0.619 | 0.674 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
