# SH6 processbench/omnimath/by-generator/Qwen2.5-Math-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 55
- Positive class (`final_answer_correct=true`): 24
- Negative class (`final_answer_correct=false`): 31
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 55 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.429 +/- 0.128 | 0.457 +/- 0.084 | 0.448 +/- 0.076 | 0.436 +/- 0.068 | 0.474 +/- 0.132 |
| trajectory_shape (logreg) | 47 | 0.640 +/- 0.181 | 0.621 +/- 0.190 | 0.532 +/- 0.144 | 0.527 +/- 0.145 | 0.492 +/- 0.178 |
| trajectory_full (logreg) | 47 | 0.640 +/- 0.181 | 0.621 +/- 0.190 | 0.532 +/- 0.144 | 0.527 +/- 0.145 | 0.492 +/- 0.178 |
| reasoning_traj (MiniRocket) | 20 | 0.533 +/- 0.076 | 0.583 +/- 0.046 | 0.517 +/- 0.079 | 0.545 +/- 0.100 | 0.384 +/- 0.051 |
| trajectory_full (lightgbm) | 47 | 0.636 +/- 0.194 | 0.651 +/- 0.165 | 0.619 +/- 0.155 | 0.618 +/- 0.145 | 0.550 +/- 0.204 |
| mode_stack (logreg) | 6 | 0.574 +/- 0.139 | 0.587 +/- 0.093 | 0.540 +/- 0.120 | 0.545 +/- 0.129 | 0.495 +/- 0.099 |
| mode_stack (lightgbm) | 6 | 0.534 +/- 0.199 | 0.560 +/- 0.151 | 0.502 +/- 0.234 | 0.509 +/- 0.227 | 0.450 +/- 0.277 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t14 | shape | 0.703 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.679 | higher -> correct |
| reasoning_traj_t02 | shape | 0.670 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.665 | higher -> correct |
| reasoning_mid_mean | shape | 0.659 | higher -> correct |
| reasoning_traj_t01 | shape | 0.650 | higher -> correct |
| reasoning_traj_t13 | shape | 0.650 | higher -> correct |
| reasoning_traj_t08 | shape | 0.644 | higher -> correct |
| reasoning_traj_t15 | shape | 0.638 | higher -> correct |
| reasoning_traj_t07 | shape | 0.637 | higher -> correct |
| reasoning_max | shape | 0.631 | higher -> wrong |
| reasoning_max_rise | shape | 0.630 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_traj_t11 | shape | 0.855 | higher -> correct |
| reasoning_min | shape | -0.778 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.773 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.582 | higher -> correct |
| reasoning_traj_t15 | shape | -0.572 | higher -> wrong |
| reasoning_positive_mass | shape | -0.561 | higher -> wrong |
| reasoning_std | shape | -0.521 | higher -> wrong |
| reasoning_traj_t17 | shape | 0.515 | higher -> correct |
| reasoning_max | shape | -0.496 | higher -> wrong |
| reasoning_traj_t06 | shape | -0.488 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.476 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.471 | higher -> correct |

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
| premature_exit | inconclusive | 55 | 0.491 [0.334, 0.645] | 0.564 | 1 | - | - | - | - |
| rambling_overlong | inconclusive | 55 | 0.509 [0.355, 0.666] | 0.564 | 6 | - | - | - | - |
| thrashing | inconclusive | 55 | 0.519 [0.379, 0.676] | 0.564 | 8 | - | - | - | - |
| no_commitment | inconclusive | 55 | 0.563 [0.398, 0.716] | 0.564 | 7 | - | - | - | - |
| derailment_late | inconclusive | 55 | 0.616 [0.452, 0.769] | 0.564 | 11 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.574**
- `trajectory_full` ROC-AUC: **0.640**
- Above-chance discrimination preserved by the mode stack: **52.9%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 31 | 0 | - | - |
| any | 11 | 17 / 31 | 25 | 0.548 | 0.680 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
