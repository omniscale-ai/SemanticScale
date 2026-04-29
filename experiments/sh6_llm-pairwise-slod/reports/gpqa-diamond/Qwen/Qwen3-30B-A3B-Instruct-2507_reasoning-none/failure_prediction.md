# SH6 gpqa-diamond/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 197
- Positive class (`is_correct=true`): 127
- Negative class (`is_correct=false`): 70

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.702 +/- 0.077 | 0.787 +/- 0.072 | 0.685 +/- 0.032 | 0.706 +/- 0.027 | 0.768 +/- 0.023 |
| trajectory_shape | 29 | 0.618 +/- 0.058 | 0.744 +/- 0.047 | 0.646 +/- 0.044 | 0.660 +/- 0.040 | 0.723 +/- 0.037 |
| trajectory_full | 29 | 0.618 +/- 0.058 | 0.744 +/- 0.047 | 0.646 +/- 0.044 | 0.660 +/- 0.040 | 0.723 +/- 0.037 |
| mode_stack | 3 | 0.455 +/- 0.088 | 0.652 +/- 0.044 | 0.463 +/- 0.117 | 0.473 +/- 0.093 | 0.549 +/- 0.068 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_n_chunks | length | 0.702 | higher -> correct |
| answer_end | landing | 0.617 | higher -> correct |
| answer_negative_mass | shape | 0.616 | higher -> wrong |
| answer_start | shape | 0.603 | higher -> correct |
| answer_end_minus_start | landing | 0.585 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | 0.580 | higher -> wrong |
| answer_rebound_from_trough | shape | 0.577 | higher -> correct |
| answer_std | shape | 0.567 | higher -> correct |
| answer_monotonicity | commitment | 0.561 | higher -> correct |
| answer_positive_mass | shape | 0.558 | higher -> correct |
| answer_fall_from_peak | derailment | 0.557 | higher -> correct |
| answer_min | shape | 0.556 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | -0.913 | higher -> wrong |
| answer_positive_mass | shape | 0.441 | higher -> correct |
| answer_negative_mass | shape | 0.391 | higher -> correct |
| answer_max_rise_pos | timing | -0.372 | higher -> wrong |
| answer_curvature_abs_mean | thrashing | -0.369 | higher -> wrong |
| answer_max_rise | shape | -0.359 | higher -> wrong |
| answer_direction_changes | thrashing | 0.315 | higher -> correct |
| answer_total_variation | thrashing | 0.252 | higher -> correct |
| answer_max_drop_pos | derailment | 0.237 | higher -> correct |
| answer_end | landing | 0.189 | higher -> correct |
| answer_min | shape | 0.176 | higher -> correct |
| answer_start | shape | 0.152 | higher -> correct |

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
| premature_exit | inconclusive | 197 | 0.500 [0.500, 0.500] | 0.355 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 197 | 0.500 [0.500, 0.500] | 0.355 | 0 | - | - | - | - |
| thrashing | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| no_commitment | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| derailment_late | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | inconclusive | 197 | 0.483 [0.407, 0.562] | 0.355 | 12 | - | - | - | - |
| answer_volatility | inconclusive | 197 | 0.492 [0.403, 0.581] | 0.355 | 22 | - | - | - | - |
| answer_uncommitted | inconclusive | 197 | 0.559 [0.471, 0.638] | 0.355 | 24 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, answer_meandering, answer_volatility, answer_uncommitted. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.455**
- `trajectory_full` ROC-AUC: **0.618**
- Above-chance discrimination preserved by the mode stack: **-37.8%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 70 | 0 | - | - |
| any | 11 | 19 / 70 | 50 | 0.271 | 0.380 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
