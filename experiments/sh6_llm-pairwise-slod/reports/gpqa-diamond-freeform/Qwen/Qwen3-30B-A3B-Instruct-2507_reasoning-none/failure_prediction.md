# SH6 gpqa-diamond-freeform/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 197
- Positive class (`is_correct=true`): 134
- Negative class (`is_correct=false`): 63

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.523 +/- 0.030 | 0.699 +/- 0.023 | 0.524 +/- 0.030 | 0.381 +/- 0.033 | 0.217 +/- 0.048 |
| trajectory_shape (logreg) | 47 | 0.488 +/- 0.062 | 0.693 +/- 0.025 | 0.480 +/- 0.055 | 0.492 +/- 0.054 | 0.578 +/- 0.054 |
| trajectory_full (logreg) | 47 | 0.488 +/- 0.062 | 0.693 +/- 0.025 | 0.480 +/- 0.055 | 0.492 +/- 0.054 | 0.578 +/- 0.054 |
| trajectory_full (lightgbm) | 47 | 0.506 +/- 0.099 | 0.702 +/- 0.067 | 0.547 +/- 0.065 | 0.619 +/- 0.062 | 0.725 +/- 0.050 |
| mode_stack (logreg) | 4 | 0.520 +/- 0.030 | 0.726 +/- 0.039 | 0.523 +/- 0.036 | 0.522 +/- 0.059 | 0.593 +/- 0.077 |
| mode_stack (lightgbm) | 4 | 0.500 +/- 0.073 | 0.737 +/- 0.050 | 0.461 +/- 0.079 | 0.533 +/- 0.068 | 0.657 +/- 0.062 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_traj_t04 | shape | 0.647 | higher -> correct |
| answer_early_mean | shape | 0.629 | higher -> correct |
| answer_std | shape | 0.613 | higher -> correct |
| answer_range | commitment | 0.608 | higher -> correct |
| answer_rebound_from_trough | shape | 0.605 | higher -> correct |
| answer_traj_t16 | shape | 0.604 | higher -> correct |
| answer_traj_t02 | shape | 0.604 | higher -> correct |
| answer_late_minus_early | transition | 0.604 | higher -> correct |
| answer_fall_from_peak | derailment | 0.596 | higher -> wrong |
| answer_min | shape | 0.596 | higher -> correct |
| answer_negative_mass | shape | 0.595 | higher -> correct |
| answer_max | shape | 0.589 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_std | shape | 0.809 | higher -> correct |
| answer_negative_mass | shape | -0.735 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | 0.526 | higher -> correct |
| answer_n_chunks | length | -0.471 | higher -> wrong |
| answer_traj_t04 | shape | 0.450 | higher -> correct |
| answer_zero_crossings | thrashing | -0.325 | higher -> wrong |
| answer_traj_t12 | shape | 0.299 | higher -> correct |
| answer_traj_t16 | shape | -0.242 | higher -> wrong |
| answer_direction_changes | thrashing | 0.234 | higher -> correct |
| answer_early_mean | shape | 0.227 | higher -> correct |
| answer_traj_t02 | shape | 0.220 | higher -> correct |
| answer_traj_t15 | shape | 0.218 | higher -> correct |

## Interpretable Failure-Mode Detectors

Each detector encodes a directional hypothesis: *higher detector score implies a higher probability of failure*.
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
| premature_exit | inconclusive | 197 | 0.500 [0.500, 0.500] | 0.320 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 197 | 0.500 [0.500, 0.500] | 0.320 | 0 | - | - | - | - |
| thrashing | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| no_commitment | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| derailment_late | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | inconclusive | 197 | 0.468 [0.384, 0.558] | 0.320 | 13 | - | - | - | - |
| answer_volatility | inconclusive | 197 | 0.462 [0.374, 0.550] | 0.320 | 20 | - | - | - | - |
| answer_uncommitted | inconclusive | 197 | 0.445 [0.355, 0.535] | 0.320 | 21 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, answer_meandering, answer_volatility, answer_uncommitted. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.520**
- `trajectory_full` ROC-AUC: **0.488**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 63 | 0 | - | - |
| any | 11 | 15 / 63 | 47 | 0.238 | 0.319 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
