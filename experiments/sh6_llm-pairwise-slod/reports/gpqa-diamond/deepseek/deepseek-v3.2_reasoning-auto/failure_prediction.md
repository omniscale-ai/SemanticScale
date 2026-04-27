# SH6 gpqa-diamond/deepseek/deepseek-v3.2_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 192
- Positive class (`is_correct=true`): 147
- Negative class (`is_correct=false`): 45

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 3 | 0.761 +/- 0.051 | 0.900 +/- 0.027 | 0.719 +/- 0.059 | 0.735 +/- 0.062 | 0.809 +/- 0.054 |
| trajectory_shape | 60 | 0.639 +/- 0.074 | 0.859 +/- 0.042 | 0.608 +/- 0.059 | 0.636 +/- 0.066 | 0.733 +/- 0.058 |
| trajectory_full | 63 | 0.702 +/- 0.108 | 0.877 +/- 0.054 | 0.651 +/- 0.078 | 0.714 +/- 0.081 | 0.802 +/- 0.061 |
| mode_stack | 10 | 0.694 +/- 0.060 | 0.878 +/- 0.030 | 0.630 +/- 0.053 | 0.646 +/- 0.050 | 0.738 +/- 0.048 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | 0.682 | higher -> correct |
| total_n_chunks | length | 0.677 | higher -> correct |
| answer_n_chunks | length | 0.656 | higher -> correct |
| answer_max_rise | shape | 0.629 | higher -> correct |
| answer_direction_changes | thrashing | 0.627 | higher -> correct |
| reasoning_mid_mean | shape | 0.608 | higher -> correct |
| answer_monotonicity | commitment | 0.608 | higher -> correct |
| answer_zero_crossings | thrashing | 0.606 | higher -> correct |
| answer_early_mean | shape | 0.603 | higher -> correct |
| reasoning_total_variation | thrashing | 0.594 | higher -> correct |
| answer_late_minus_early | transition | 0.594 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.592 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_fall_from_peak | derailment | 1.202 | higher -> correct |
| answer_n_chunks | length | 1.115 | higher -> correct |
| answer_early_mean | shape | -0.774 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.716 | higher -> correct |
| answer_std | shape | -0.690 | higher -> wrong |
| reasoning_positive_mass | shape | -0.672 | higher -> wrong |
| answer_positive_mass | shape | -0.668 | higher -> wrong |
| answer_total_variation | thrashing | 0.574 | higher -> correct |
| reasoning_n_chunks | length | -0.571 | higher -> wrong |
| reasoning_max_drop | derailment | 0.564 | higher -> correct |
| answer_end_minus_reasoning_end | landing | -0.543 | higher -> wrong |
| total_n_chunks | length | -0.540 | higher -> wrong |

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
| premature_exit | inverted | 192 | 0.316 [0.228, 0.423] | 0.234 | 16 | - | - | - | - |
| rambling_overlong | confirmed | 192 | 0.684 [0.577, 0.772] | 0.234 | 27 | 0.444 | 0.267 | 0.333 | 1.896 |
| thrashing | inconclusive | 192 | 0.524 [0.424, 0.611] | 0.234 | 6 | - | - | - | - |
| no_commitment | inconclusive | 192 | 0.535 [0.443, 0.628] | 0.234 | 17 | - | - | - | - |
| derailment_late | inconclusive | 192 | 0.419 [0.332, 0.513] | 0.234 | 20 | - | - | - | - |
| answer_drift | inconclusive | 186 | 0.411 [0.318, 0.507] | 0.210 | 18 | - | - | - | - |
| answer_meandering | inverted | 186 | 0.382 [0.329, 0.444] | 0.210 | 14 | - | - | - | - |
| answer_volatility | inverted | 186 | 0.360 [0.277, 0.453] | 0.210 | 15 | - | - | - | - |
| answer_uncommitted | inverted | 186 | 0.399 [0.332, 0.478] | 0.210 | 19 | - | - | - | - |
| answer_overrange | inconclusive | 186 | 0.428 [0.334, 0.517] | 0.210 | 15 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong.
- **Hypothesis falsified (inverted)**: premature_exit, answer_meandering, answer_volatility, answer_uncommitted. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: thrashing, no_commitment, derailment_late, answer_drift, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.694**
- `trajectory_full` ROC-AUC: **0.702**
- Above-chance discrimination preserved by the mode stack: **96.0%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 12 / 45 | 27 | 0.267 | 0.444 |
| any | 11 | 26 / 45 | 97 | 0.578 | 0.268 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
