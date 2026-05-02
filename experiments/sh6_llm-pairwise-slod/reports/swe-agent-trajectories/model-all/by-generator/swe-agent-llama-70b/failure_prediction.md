# SH6 swe-agent-trajectories/model-all/by-generator/swe-agent-llama-70b — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 1232
- Positive class (`final_answer_correct=true`): 519
- Negative class (`final_answer_correct=false`): 713
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 1232 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Status

- The `length_abort` baseline includes both chunk-count features and `truncation_abort_score` on this run.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_abort (logreg) | 2 | 0.632 +/- 0.025 | 0.512 +/- 0.016 | 0.585 +/- 0.048 | 0.548 +/- 0.045 | 0.605 +/- 0.044 |
| trajectory_shape (logreg) | 47 | 0.896 +/- 0.008 | 0.847 +/- 0.008 | 0.809 +/- 0.010 | 0.812 +/- 0.011 | 0.780 +/- 0.011 |
| trajectory_full (logreg) | 47 | 0.896 +/- 0.008 | 0.848 +/- 0.008 | 0.809 +/- 0.010 | 0.812 +/- 0.011 | 0.780 +/- 0.011 |
| reasoning_traj (MiniRocket) | 20 | 0.899 +/- 0.007 | 0.853 +/- 0.009 | 0.809 +/- 0.011 | 0.809 +/- 0.012 | 0.780 +/- 0.012 |
| trajectory_full (lightgbm) | 47 | 0.899 +/- 0.008 | 0.853 +/- 0.012 | 0.807 +/- 0.008 | 0.807 +/- 0.008 | 0.779 +/- 0.010 |
| mode_stack (logreg) | 7 | 0.659 +/- 0.026 | 0.556 +/- 0.026 | 0.620 +/- 0.032 | 0.590 +/- 0.030 | 0.623 +/- 0.032 |
| mode_stack (lightgbm) | 7 | 0.907 +/- 0.016 | 0.872 +/- 0.012 | 0.824 +/- 0.015 | 0.822 +/- 0.013 | 0.798 +/- 0.017 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t18 | shape | 0.698 | higher -> correct |
| reasoning_start | shape | 0.671 | higher -> correct |
| reasoning_late_mean | landing | 0.658 | higher -> correct |
| reasoning_late_minus_early | transition | 0.641 | higher -> correct |
| reasoning_traj_t17 | shape | 0.629 | higher -> correct |
| reasoning_traj_t06 | shape | 0.627 | higher -> correct |
| reasoning_early_mean | shape | 0.607 | higher -> correct |
| reasoning_traj_t15 | shape | 0.607 | higher -> correct |
| reasoning_end_minus_start | landing | 0.586 | higher -> correct |
| reasoning_traj_t16 | shape | 0.586 | higher -> correct |
| reasoning_traj_t09 | shape | 0.585 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.584 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | -2.728 | higher -> wrong |
| reasoning_total_variation | thrashing | 2.227 | higher -> correct |
| reasoning_zero_crossings | thrashing | -1.888 | higher -> wrong |
| reasoning_traj_t05 | shape | -1.718 | higher -> wrong |
| reasoning_max_rise | shape | -1.707 | higher -> wrong |
| reasoning_max_rise_pos | timing | 1.440 | higher -> correct |
| reasoning_traj_t02 | shape | 1.438 | higher -> correct |
| reasoning_monotonicity | commitment | -1.382 | higher -> wrong |
| reasoning_max_drop | derailment | 1.315 | higher -> correct |
| reasoning_std | shape | -1.273 | higher -> wrong |
| reasoning_traj_t03 | shape | -1.270 | higher -> wrong |
| reasoning_traj_t18 | shape | 1.266 | higher -> correct |

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
| premature_exit | inverted | 1232 | 0.437 [0.405, 0.469] | 0.579 | 80 | - | - | - | - |
| rambling_overlong | confirmed | 1232 | 0.563 [0.531, 0.595] | 0.579 | 158 | 0.728 | 0.161 | 0.264 | 1.258 |
| thrashing | confirmed | 1232 | 0.568 [0.538, 0.600] | 0.579 | 179 | 0.737 | 0.185 | 0.296 | 1.274 |
| no_commitment | confirmed | 1232 | 0.534 [0.503, 0.566] | 0.579 | 132 | 0.614 | 0.114 | 0.192 | 1.060 |
| derailment_late | inconclusive | 1232 | 0.478 [0.446, 0.511] | 0.579 | 125 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 1232 | 0.594 [0.579, 0.610] | 0.579 | 165 | 0.921 | 0.213 | 0.346 | 1.592 |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, no_commitment, truncation_abort.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.659**
- `trajectory_full` ROC-AUC: **0.896**
- Above-chance discrimination preserved by the mode stack: **40.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 315 / 713 | 450 | 0.442 | 0.700 |
| any | 11 | 378 / 713 | 594 | 0.530 | 0.636 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
