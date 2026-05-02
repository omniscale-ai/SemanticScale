# SH6 swe-agent-trajectories/model-all/by-generator/swe-agent-llama-8b — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 549
- Positive class (`final_answer_correct=true`): 214
- Negative class (`final_answer_correct=false`): 335
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 549 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Status

- The `length_abort` baseline includes both chunk-count features and `truncation_abort_score` on this run.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_abort (logreg) | 2 | 0.759 +/- 0.044 | 0.607 +/- 0.079 | 0.734 +/- 0.024 | 0.705 +/- 0.021 | 0.696 +/- 0.025 |
| trajectory_shape (logreg) | 47 | 0.715 +/- 0.046 | 0.606 +/- 0.050 | 0.679 +/- 0.039 | 0.676 +/- 0.036 | 0.624 +/- 0.045 |
| trajectory_full (logreg) | 47 | 0.715 +/- 0.046 | 0.605 +/- 0.050 | 0.679 +/- 0.039 | 0.676 +/- 0.036 | 0.624 +/- 0.045 |
| reasoning_traj (MiniRocket) | 20 | 0.709 +/- 0.048 | 0.601 +/- 0.053 | 0.681 +/- 0.040 | 0.683 +/- 0.039 | 0.623 +/- 0.044 |
| trajectory_full (lightgbm) | 47 | 0.711 +/- 0.047 | 0.601 +/- 0.052 | 0.681 +/- 0.040 | 0.683 +/- 0.039 | 0.623 +/- 0.044 |
| mode_stack (logreg) | 7 | 0.751 +/- 0.037 | 0.586 +/- 0.051 | 0.723 +/- 0.028 | 0.698 +/- 0.022 | 0.683 +/- 0.031 |
| mode_stack (lightgbm) | 7 | 0.850 +/- 0.029 | 0.763 +/- 0.023 | 0.765 +/- 0.033 | 0.765 +/- 0.031 | 0.717 +/- 0.039 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t18 | shape | 0.650 | higher -> correct |
| reasoning_traj_t17 | shape | 0.648 | higher -> correct |
| reasoning_late_minus_early | transition | 0.636 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.633 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.627 | higher -> correct |
| reasoning_n_chunks | length | 0.625 | higher -> correct |
| reasoning_early_mean | shape | 0.620 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.619 | higher -> correct |
| reasoning_negative_mass | shape | 0.617 | higher -> correct |
| reasoning_late_mean | landing | 0.614 | higher -> correct |
| reasoning_total_variation | thrashing | 0.606 | higher -> correct |
| reasoning_time_positive | shape | 0.605 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_traj_t09 | shape | -0.992 | higher -> wrong |
| reasoning_traj_t05 | shape | -0.919 | higher -> wrong |
| reasoning_positive_mass | shape | -0.881 | higher -> wrong |
| reasoning_traj_t10 | shape | -0.810 | higher -> wrong |
| reasoning_trough_pos | timing | 0.758 | higher -> correct |
| reasoning_zero_crossings | thrashing | -0.747 | higher -> wrong |
| reasoning_start | shape | -0.739 | higher -> wrong |
| reasoning_max | shape | -0.681 | higher -> wrong |
| reasoning_max_drop | derailment | 0.630 | higher -> correct |
| reasoning_traj_t18 | shape | 0.602 | higher -> correct |
| reasoning_traj_t11 | shape | 0.567 | higher -> correct |
| reasoning_traj_t03 | shape | -0.492 | higher -> wrong |

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
| premature_exit | inverted | 549 | 0.375 [0.331, 0.420] | 0.610 | 27 | - | - | - | - |
| rambling_overlong | confirmed | 549 | 0.625 [0.580, 0.669] | 0.610 | 74 | 0.716 | 0.158 | 0.259 | 1.174 |
| thrashing | confirmed | 549 | 0.633 [0.587, 0.679] | 0.610 | 47 | 0.723 | 0.101 | 0.178 | 1.186 |
| no_commitment | confirmed | 549 | 0.553 [0.506, 0.599] | 0.610 | 55 | 0.782 | 0.128 | 0.221 | 1.281 |
| derailment_late | inconclusive | 549 | 0.543 [0.495, 0.591] | 0.610 | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 549 | 0.730 [0.696, 0.765] | 0.610 | 226 | 0.876 | 0.591 | 0.706 | 1.436 |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, no_commitment, truncation_abort.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.751**
- `trajectory_full` ROC-AUC: **0.715**
- Above-chance discrimination preserved by the mode stack: **116.5%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 242 / 335 | 311 | 0.722 | 0.778 |
| any | 11 | 259 / 335 | 335 | 0.773 | 0.773 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
