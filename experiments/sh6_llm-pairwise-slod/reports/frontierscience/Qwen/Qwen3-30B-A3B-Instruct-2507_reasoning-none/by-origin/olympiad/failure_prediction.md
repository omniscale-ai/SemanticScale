# SH6 frontierscience/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 98
- Positive class (`is_correct=true`): 49
- Negative class (`is_correct=false`): 49

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.708 +/- 0.060 | 0.703 +/- 0.069 | 0.702 +/- 0.061 | 0.704 +/- 0.061 | 0.714 +/- 0.045 |
| trajectory_shape (logreg) | 47 | 0.526 +/- 0.082 | 0.536 +/- 0.027 | 0.548 +/- 0.084 | 0.551 +/- 0.083 | 0.532 +/- 0.105 |
| trajectory_full (logreg) | 47 | 0.526 +/- 0.082 | 0.536 +/- 0.027 | 0.548 +/- 0.084 | 0.551 +/- 0.083 | 0.532 +/- 0.105 |
| trajectory_full (lightgbm) | 47 | 0.509 +/- 0.144 | 0.591 +/- 0.110 | 0.520 +/- 0.086 | 0.520 +/- 0.084 | 0.544 +/- 0.077 |
| mode_stack (logreg) | 4 | 0.700 +/- 0.107 | 0.727 +/- 0.102 | 0.692 +/- 0.075 | 0.693 +/- 0.075 | 0.709 +/- 0.069 |
| mode_stack (lightgbm) | 4 | 0.570 +/- 0.076 | 0.631 +/- 0.068 | 0.550 +/- 0.092 | 0.549 +/- 0.096 | 0.525 +/- 0.148 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_n_chunks | length | 0.708 | higher -> correct |
| answer_max_rise_pos | timing | 0.648 | higher -> wrong |
| answer_traj_t04 | shape | 0.629 | higher -> correct |
| answer_positive_mass | shape | 0.629 | higher -> wrong |
| answer_traj_t17 | shape | 0.623 | higher -> wrong |
| answer_trough_pos | timing | 0.610 | higher -> correct |
| answer_traj_t16 | shape | 0.609 | higher -> wrong |
| answer_traj_t09 | shape | 0.608 | higher -> correct |
| answer_rebound_from_trough | shape | 0.603 | higher -> wrong |
| answer_traj_t18 | shape | 0.602 | higher -> wrong |
| answer_zero_crossings | thrashing | 0.601 | higher -> wrong |
| answer_range | commitment | 0.596 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | -0.900 | higher -> wrong |
| answer_traj_t12 | shape | 0.594 | higher -> correct |
| answer_traj_t04 | shape | -0.589 | higher -> wrong |
| answer_traj_t11 | shape | -0.572 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | 0.566 | higher -> correct |
| answer_traj_t10 | shape | 0.549 | higher -> correct |
| answer_min | shape | -0.516 | higher -> wrong |
| answer_traj_t09 | shape | -0.516 | higher -> wrong |
| answer_traj_t03 | shape | 0.493 | higher -> correct |
| answer_max_rise | shape | -0.465 | higher -> wrong |
| answer_peak_pos | timing | -0.464 | higher -> wrong |
| answer_traj_t17 | shape | 0.401 | higher -> correct |

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
| premature_exit | inconclusive | 98 | 0.500 [0.500, 0.500] | 0.500 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 98 | 0.500 [0.500, 0.500] | 0.500 | 0 | - | - | - | - |
| thrashing | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| no_commitment | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| derailment_late | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | inconclusive | 98 | 0.460 [0.346, 0.579] | 0.500 | 1 | - | - | - | - |
| answer_volatility | inconclusive | 98 | 0.535 [0.415, 0.645] | 0.500 | 13 | - | - | - | - |
| answer_uncommitted | inconclusive | 98 | 0.488 [0.367, 0.609] | 0.500 | 12 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, answer_meandering, answer_volatility, answer_uncommitted. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.700**
- `trajectory_full` ROC-AUC: **0.526**
- Above-chance discrimination preserved by the mode stack: **768.4%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 49 | 0 | - | - |
| any | 11 | 14 / 49 | 24 | 0.286 | 0.583 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
