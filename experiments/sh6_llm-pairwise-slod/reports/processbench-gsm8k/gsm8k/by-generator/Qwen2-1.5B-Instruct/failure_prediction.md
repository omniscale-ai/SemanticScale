# SH6 processbench/gsm8k/by-generator/Qwen2-1.5B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 41
- Positive class (`final_answer_correct=true`): 5
- Negative class (`final_answer_correct=false`): 36
- Label agreement (`is_correct` vs `final_answer_correct`): 97.6% over 41 items
- Final answer correct but reasoning wrong: 1
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.700 +/- 0.384 | 0.556 +/- 0.387 | 0.634 +/- 0.167 | 0.511 +/- 0.082 | 0.270 +/- 0.140 |
| trajectory_shape (logreg) | 29 | 0.504 +/- 0.178 | 0.240 +/- 0.077 | 0.487 +/- 0.188 | 0.706 +/- 0.067 | 0.100 +/- 0.200 |
| trajectory_full (logreg) | 29 | 0.504 +/- 0.178 | 0.240 +/- 0.077 | 0.487 +/- 0.188 | 0.706 +/- 0.067 | 0.100 +/- 0.200 |
| trajectory_full (lightgbm) | 29 | 0.393 +/- 0.172 | 0.202 +/- 0.068 | 0.445 +/- 0.028 | 0.781 +/- 0.048 | 0.000 +/- 0.000 |
| mode_stack (logreg) | 6 | 0.704 +/- 0.268 | 0.540 +/- 0.380 | 0.634 +/- 0.220 | 0.661 +/- 0.077 | 0.260 +/- 0.215 |
| mode_stack (lightgbm) | 6 | 0.557 +/- 0.357 | 0.406 +/- 0.326 | 0.505 +/- 0.186 | 0.736 +/- 0.102 | 0.100 +/- 0.200 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max_rise_pos | timing | 0.914 | higher -> wrong |
| reasoning_positive_mass | shape | 0.886 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.843 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.841 | higher -> wrong |
| reasoning_time_positive | shape | 0.821 | higher -> wrong |
| reasoning_max_drop | derailment | 0.789 | higher -> correct |
| reasoning_time_negative | shape | 0.768 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.739 | higher -> correct |
| reasoning_early_mean | shape | 0.736 | higher -> correct |
| reasoning_min | shape | 0.725 | higher -> wrong |
| reasoning_trough_pos | timing | 0.723 | higher -> correct |
| reasoning_max | shape | 0.707 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise_pos | timing | 1.325 | higher -> correct |
| reasoning_time_negative | shape | -1.002 | higher -> wrong |
| reasoning_max_drop | derailment | -0.887 | higher -> wrong |
| reasoning_time_positive | shape | 0.716 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.680 | higher -> wrong |
| reasoning_direction_changes | thrashing | -0.559 | higher -> wrong |
| reasoning_trough_pos | timing | -0.558 | higher -> wrong |
| reasoning_positive_mass | shape | 0.508 | higher -> correct |
| reasoning_start | shape | 0.494 | higher -> correct |
| reasoning_monotonicity | commitment | -0.403 | higher -> wrong |
| reasoning_fall_from_peak | derailment | -0.385 | higher -> wrong |
| reasoning_max_drop_pos | derailment | -0.357 | higher -> wrong |

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
| premature_exit | inconclusive | 41 | 0.283 [0.040, 0.585] | 0.878 | 2 | - | - | - | - |
| rambling_overlong | inconclusive | 41 | 0.717 [0.415, 0.960] | 0.878 | 10 | - | - | - | - |
| thrashing | confirmed | 41 | 0.725 [0.553, 0.913] | 0.878 | 13 | 1.000 | 0.361 | 0.531 | 1.139 |
| no_commitment | inconclusive | 41 | 0.464 [0.113, 0.843] | 0.878 | 5 | - | - | - | - |
| derailment_late | inconclusive | 41 | 0.714 [0.492, 0.888] | 0.878 | 19 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: thrashing.
- **Inconclusive**: premature_exit, rambling_overlong, no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.704**
- `trajectory_full` ROC-AUC: **0.504**
- Above-chance discrimination preserved by the mode stack: **5700.0%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 13 / 36 | 13 | 0.361 | 1.000 |
| any | 11 | 27 / 36 | 30 | 0.750 | 0.900 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
