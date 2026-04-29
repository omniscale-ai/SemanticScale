# SH6 processbench/omnimath/by-generator/Llama-3.1-8B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 65
- Positive class (`final_answer_correct=true`): 28
- Negative class (`final_answer_correct=false`): 37
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 65 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.835 +/- 0.141 | 0.790 +/- 0.149 | 0.740 +/- 0.127 | 0.738 +/- 0.125 | 0.723 +/- 0.118 |
| trajectory_shape (logreg) | 29 | 0.807 +/- 0.098 | 0.792 +/- 0.115 | 0.706 +/- 0.148 | 0.708 +/- 0.149 | 0.681 +/- 0.148 |
| trajectory_full (logreg) | 29 | 0.807 +/- 0.098 | 0.792 +/- 0.115 | 0.706 +/- 0.148 | 0.708 +/- 0.149 | 0.681 +/- 0.148 |
| trajectory_full (lightgbm) | 29 | 0.635 +/- 0.184 | 0.668 +/- 0.191 | 0.647 +/- 0.212 | 0.662 +/- 0.221 | 0.607 +/- 0.246 |
| mode_stack (logreg) | 6 | 0.799 +/- 0.153 | 0.782 +/- 0.158 | 0.740 +/- 0.127 | 0.738 +/- 0.125 | 0.723 +/- 0.118 |
| mode_stack (lightgbm) | 6 | 0.711 +/- 0.150 | 0.726 +/- 0.138 | 0.661 +/- 0.142 | 0.662 +/- 0.143 | 0.619 +/- 0.156 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | 0.835 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.784 | higher -> correct |
| reasoning_direction_changes | thrashing | 0.773 | higher -> correct |
| reasoning_max_rise | shape | 0.765 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.726 | higher -> correct |
| reasoning_max_drop | derailment | 0.710 | higher -> correct |
| reasoning_std | shape | 0.693 | higher -> correct |
| reasoning_mid_mean | shape | 0.667 | higher -> wrong |
| reasoning_rebound_from_trough | shape | 0.664 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.661 | higher -> correct |
| reasoning_negative_mass | shape | 0.660 | higher -> correct |
| reasoning_range | commitment | 0.658 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_n_chunks | length | -1.478 | higher -> wrong |
| reasoning_direction_changes | thrashing | -0.777 | higher -> wrong |
| reasoning_min | shape | 0.765 | higher -> correct |
| reasoning_max_rise | shape | -0.745 | higher -> wrong |
| reasoning_time_positive | shape | 0.738 | higher -> correct |
| reasoning_time_negative | shape | -0.738 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.717 | higher -> wrong |
| reasoning_positive_mass | shape | -0.597 | higher -> wrong |
| reasoning_late_mean | landing | -0.591 | higher -> wrong |
| reasoning_max_rise_pos | timing | -0.508 | higher -> wrong |
| reasoning_max | shape | 0.481 | higher -> correct |
| reasoning_monotonicity | commitment | -0.430 | higher -> wrong |

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
| premature_exit | inverted | 65 | 0.168 [0.081, 0.276] | 0.569 | 1 | - | - | - | - |
| rambling_overlong | confirmed | 65 | 0.832 [0.724, 0.919] | 0.569 | 19 | 0.895 | 0.459 | 0.607 | 1.572 |
| thrashing | confirmed | 65 | 0.775 [0.648, 0.872] | 0.569 | 22 | 0.864 | 0.514 | 0.644 | 1.517 |
| no_commitment | inconclusive | 65 | 0.590 [0.437, 0.727] | 0.569 | 3 | - | - | - | - |
| derailment_late | inconclusive | 65 | 0.414 [0.278, 0.566] | 0.569 | 6 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.799**
- `trajectory_full` ROC-AUC: **0.807**
- Above-chance discrimination preserved by the mode stack: **97.3%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 2 | 19 / 37 | 23 | 0.514 | 0.826 |
| any | 11 | 23 / 37 | 32 | 0.622 | 0.719 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
