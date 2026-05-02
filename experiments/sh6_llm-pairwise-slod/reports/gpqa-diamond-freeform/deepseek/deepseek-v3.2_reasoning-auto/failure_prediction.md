# SH6 gpqa-diamond-freeform/deepseek/deepseek-v3.2_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 192
- Positive class (`is_correct=true`): 148
- Negative class (`is_correct=false`): 44

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.778 +/- 0.110 | 0.898 +/- 0.053 | 0.670 +/- 0.092 | 0.614 +/- 0.062 | 0.694 +/- 0.048 |
| trajectory_shape (logreg) | 96 | 0.752 +/- 0.091 | 0.915 +/- 0.038 | 0.656 +/- 0.112 | 0.718 +/- 0.063 | 0.807 +/- 0.045 |
| trajectory_full (logreg) | 99 | 0.768 +/- 0.072 | 0.911 +/- 0.041 | 0.662 +/- 0.087 | 0.703 +/- 0.040 | 0.792 +/- 0.032 |
| reasoning_traj (MiniRocket) | 20 | 0.526 +/- 0.064 | 0.807 +/- 0.026 | 0.467 +/- 0.040 | 0.599 +/- 0.079 | 0.725 +/- 0.080 |
| trajectory_full (lightgbm) | 99 | 0.747 +/- 0.055 | 0.879 +/- 0.046 | 0.717 +/- 0.071 | 0.848 +/- 0.046 | 0.907 +/- 0.028 |
| mode_stack (logreg) | 13 | 0.758 +/- 0.119 | 0.896 +/- 0.066 | 0.723 +/- 0.110 | 0.723 +/- 0.065 | 0.801 +/- 0.047 |
| mode_stack (lightgbm) | 13 | 0.763 +/- 0.084 | 0.887 +/- 0.059 | 0.726 +/- 0.079 | 0.813 +/- 0.078 | 0.878 +/- 0.056 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_n_chunks | length | 0.750 | higher -> correct |
| reasoning_max_drop | derailment | 0.660 | higher -> wrong |
| answer_monotonicity | commitment | 0.644 | higher -> correct |
| answer_peak_pos | timing | 0.635 | higher -> correct |
| answer_rebound_from_trough | shape | 0.631 | higher -> correct |
| answer_trough_pos | timing | 0.624 | higher -> correct |
| answer_direction_changes | thrashing | 0.624 | higher -> correct |
| total_n_chunks | length | 0.623 | higher -> correct |
| reasoning_end_minus_start | landing | 0.619 | higher -> wrong |
| reasoning_traj_t16 | shape | 0.611 | higher -> correct |
| answer_range | commitment | 0.607 | higher -> correct |
| answer_zero_crossings | thrashing | 0.603 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | 2.171 | higher -> correct |
| answer_trough_pos | timing | -1.587 | higher -> wrong |
| answer_direction_changes | thrashing | -1.077 | higher -> wrong |
| answer_zero_crossings | thrashing | 0.990 | higher -> correct |
| answer_max_rise_pos | timing | -0.914 | higher -> wrong |
| reasoning_std | shape | -0.883 | higher -> wrong |
| answer_rebound_from_trough | shape | 0.846 | higher -> correct |
| reasoning_traj_t16 | shape | 0.728 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | -0.724 | higher -> wrong |
| answer_max_drop | derailment | -0.677 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.667 | higher -> wrong |
| reasoning_traj_t03 | shape | 0.644 | higher -> correct |

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
| premature_exit | inconclusive | 192 | 0.493 [0.417, 0.569] | 0.229 | 18 | - | - | - | - |
| rambling_overlong | inconclusive | 192 | 0.507 [0.431, 0.583] | 0.229 | 0 | - | - | - | - |
| thrashing | inconclusive | 192 | 0.537 [0.438, 0.636] | 0.229 | 25 | - | - | - | - |
| no_commitment | inconclusive | 192 | 0.492 [0.377, 0.607] | 0.229 | 27 | - | - | - | - |
| derailment_late | inconclusive | 192 | 0.516 [0.408, 0.619] | 0.229 | 24 | - | - | - | - |
| answer_drift | inconclusive | 175 | 0.432 [0.308, 0.565] | 0.154 | 17 | - | - | - | - |
| answer_meandering | inconclusive | 175 | 0.429 [0.335, 0.538] | 0.154 | 0 | - | - | - | - |
| answer_volatility | inverted | 175 | 0.371 [0.272, 0.487] | 0.154 | 17 | - | - | - | - |
| answer_uncommitted | inverted | 175 | 0.396 [0.312, 0.496] | 0.154 | 16 | - | - | - | - |
| answer_overrange | inconclusive | 175 | 0.445 [0.342, 0.546] | 0.154 | 16 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Hypothesis falsified (inverted)**: answer_volatility, answer_uncommitted. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.758**
- `trajectory_full` ROC-AUC: **0.768**
- Above-chance discrimination preserved by the mode stack: **96.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 44 | 0 | - | - |
| any | 11 | 25 / 44 | 102 | 0.568 | 0.245 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
