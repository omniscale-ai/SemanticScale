# SH6 processbench/omnimath/by-generator/Llama-3.1-70B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 42
- Positive class (`final_answer_correct=true`): 17
- Negative class (`final_answer_correct=false`): 25
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 42 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.823 +/- 0.131 | 0.820 +/- 0.143 | 0.688 +/- 0.119 | 0.669 +/- 0.134 | 0.675 +/- 0.110 |
| trajectory_shape (logreg) | 29 | 0.697 +/- 0.227 | 0.707 +/- 0.227 | 0.637 +/- 0.219 | 0.644 +/- 0.197 | 0.559 +/- 0.309 |
| trajectory_full (logreg) | 29 | 0.697 +/- 0.227 | 0.707 +/- 0.227 | 0.637 +/- 0.219 | 0.644 +/- 0.197 | 0.559 +/- 0.309 |
| trajectory_full (lightgbm) | 29 | 0.633 +/- 0.259 | 0.662 +/- 0.202 | 0.678 +/- 0.151 | 0.667 +/- 0.179 | 0.652 +/- 0.125 |
| mode_stack (logreg) | 6 | 0.723 +/- 0.181 | 0.691 +/- 0.215 | 0.762 +/- 0.155 | 0.739 +/- 0.163 | 0.743 +/- 0.164 |
| mode_stack (lightgbm) | 6 | 0.508 +/- 0.116 | 0.559 +/- 0.103 | 0.677 +/- 0.052 | 0.689 +/- 0.064 | 0.612 +/- 0.087 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_curvature_abs_mean | thrashing | 0.860 | higher -> correct |
| reasoning_max_rise | shape | 0.830 | higher -> correct |
| reasoning_n_chunks | length | 0.823 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.813 | higher -> correct |
| reasoning_negative_mass | shape | 0.800 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.783 | higher -> correct |
| reasoning_total_variation | thrashing | 0.753 | higher -> correct |
| reasoning_std | shape | 0.743 | higher -> correct |
| reasoning_early_mean | shape | 0.740 | higher -> correct |
| reasoning_monotonicity | commitment | 0.733 | higher -> correct |
| reasoning_range | commitment | 0.730 | higher -> correct |
| reasoning_min | shape | 0.720 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise | shape | -1.459 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.768 | higher -> wrong |
| reasoning_max | shape | -0.766 | higher -> wrong |
| reasoning_peak_pos | timing | 0.710 | higher -> correct |
| reasoning_n_chunks | length | -0.605 | higher -> wrong |
| reasoning_min | shape | -0.558 | higher -> wrong |
| reasoning_negative_mass | shape | 0.547 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.485 | higher -> wrong |
| reasoning_late_mean | landing | -0.410 | higher -> wrong |
| reasoning_max_rise_pos | timing | -0.381 | higher -> wrong |
| reasoning_late_minus_early | transition | -0.352 | higher -> wrong |
| reasoning_fall_from_peak | derailment | -0.334 | higher -> wrong |

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
| premature_exit | inverted | 42 | 0.185 [0.055, 0.330] | 0.595 | 0 | - | - | - | - |
| rambling_overlong | confirmed | 42 | 0.815 [0.670, 0.945] | 0.595 | 15 | 0.867 | 0.520 | 0.650 | 1.456 |
| thrashing | confirmed | 42 | 0.814 [0.666, 0.933] | 0.595 | 13 | 0.846 | 0.440 | 0.579 | 1.422 |
| no_commitment | confirmed | 42 | 0.732 [0.570, 0.886] | 0.595 | 14 | 0.857 | 0.480 | 0.615 | 1.440 |
| derailment_late | inconclusive | 42 | 0.387 [0.202, 0.571] | 0.595 | 5 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing, no_commitment.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.723**
- `trajectory_full` ROC-AUC: **0.697**
- Above-chance discrimination preserved by the mode stack: **113.6%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 3 | 16 / 25 | 18 | 0.640 | 0.889 |
| any | 11 | 19 / 25 | 23 | 0.760 | 0.826 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
