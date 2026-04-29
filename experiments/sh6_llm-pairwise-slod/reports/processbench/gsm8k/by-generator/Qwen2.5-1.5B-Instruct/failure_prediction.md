# SH6 processbench-gsm8k/gsm8k/by-generator/Qwen2.5-1.5B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 42
- Positive class (`final_answer_correct=true`): 10
- Negative class (`final_answer_correct=false`): 32
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 42 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.782 +/- 0.200 | 0.622 +/- 0.275 | 0.736 +/- 0.216 | 0.756 +/- 0.227 | 0.630 +/- 0.260 |
| trajectory_shape (logreg) | 29 | 0.643 +/- 0.224 | 0.552 +/- 0.223 | 0.517 +/- 0.208 | 0.569 +/- 0.130 | 0.214 +/- 0.263 |
| trajectory_full (logreg) | 29 | 0.643 +/- 0.224 | 0.552 +/- 0.223 | 0.517 +/- 0.208 | 0.569 +/- 0.130 | 0.214 +/- 0.263 |
| trajectory_full (lightgbm) | 29 | 0.467 +/- 0.200 | 0.342 +/- 0.103 | 0.526 +/- 0.217 | 0.642 +/- 0.155 | 0.233 +/- 0.291 |
| mode_stack (logreg) | 6 | 0.743 +/- 0.248 | 0.677 +/- 0.293 | 0.579 +/- 0.236 | 0.617 +/- 0.215 | 0.413 +/- 0.247 |
| mode_stack (lightgbm) | 6 | 0.730 +/- 0.178 | 0.553 +/- 0.182 | 0.607 +/- 0.174 | 0.664 +/- 0.169 | 0.404 +/- 0.251 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_curvature_abs_mean | thrashing | 0.802 | higher -> correct |
| reasoning_max_rise | shape | 0.786 | higher -> correct |
| reasoning_n_chunks | length | 0.782 | higher -> correct |
| reasoning_range | commitment | 0.769 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.757 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.750 | higher -> wrong |
| reasoning_max | shape | 0.738 | higher -> wrong |
| reasoning_max_drop | derailment | 0.725 | higher -> correct |
| reasoning_total_variation | thrashing | 0.724 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.695 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.693 | higher -> correct |
| reasoning_negative_mass | shape | 0.693 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | -1.008 | higher -> wrong |
| reasoning_positive_mass | shape | -0.875 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.795 | higher -> correct |
| reasoning_max_drop | derailment | -0.645 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.616 | higher -> correct |
| reasoning_mid_mean | shape | -0.615 | higher -> wrong |
| reasoning_negative_mass | shape | 0.580 | higher -> correct |
| reasoning_min | shape | 0.484 | higher -> correct |
| reasoning_max_rise | shape | -0.470 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.456 | higher -> wrong |
| reasoning_max | shape | 0.424 | higher -> correct |
| reasoning_peak_pos | timing | 0.354 | higher -> correct |

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
| premature_exit | inverted | 42 | 0.194 [0.066, 0.340] | 0.762 | 0 | - | - | - | - |
| rambling_overlong | confirmed | 42 | 0.806 [0.660, 0.934] | 0.762 | 17 | 0.941 | 0.500 | 0.653 | 1.235 |
| thrashing | inconclusive | 42 | 0.683 [0.497, 0.843] | 0.762 | 11 | - | - | - | - |
| no_commitment | inconclusive | 42 | 0.706 [0.478, 0.886] | 0.762 | 18 | - | - | - | - |
| derailment_late | inconclusive | 42 | 0.514 [0.269, 0.776] | 0.762 | 5 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: thrashing, no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.743**
- `trajectory_full` ROC-AUC: **0.643**
- Above-chance discrimination preserved by the mode stack: **170.0%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 16 / 32 | 17 | 0.500 | 0.941 |
| any | 11 | 25 / 32 | 28 | 0.781 | 0.893 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
