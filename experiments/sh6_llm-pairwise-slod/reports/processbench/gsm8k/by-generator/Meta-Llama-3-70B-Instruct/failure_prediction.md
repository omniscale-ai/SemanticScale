# SH6 processbench/gsm8k/by-generator/Meta-Llama-3-70B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 31
- Positive class (`final_answer_correct=true`): 15
- Negative class (`final_answer_correct=false`): 16
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 31 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.231 +/- 0.159 | 0.413 +/- 0.048 | 0.358 +/- 0.073 | 0.357 +/- 0.074 | 0.490 +/- 0.121 |
| trajectory_shape (logreg) | 47 | 0.467 +/- 0.178 | 0.568 +/- 0.082 | 0.442 +/- 0.235 | 0.443 +/- 0.236 | 0.404 +/- 0.251 |
| trajectory_full (logreg) | 47 | 0.467 +/- 0.178 | 0.568 +/- 0.082 | 0.442 +/- 0.235 | 0.443 +/- 0.236 | 0.404 +/- 0.251 |
| reasoning_traj (MiniRocket) | 20 | 0.611 +/- 0.122 | 0.681 +/- 0.114 | 0.525 +/- 0.073 | 0.519 +/- 0.079 | 0.473 +/- 0.104 |
| trajectory_full (lightgbm) | 47 | 0.572 +/- 0.211 | 0.678 +/- 0.139 | 0.600 +/- 0.249 | 0.605 +/- 0.254 | 0.546 +/- 0.304 |
| mode_stack (logreg) | 6 | 0.311 +/- 0.191 | 0.509 +/- 0.086 | 0.275 +/- 0.090 | 0.286 +/- 0.103 | 0.224 +/- 0.196 |
| mode_stack (lightgbm) | 6 | 0.544 +/- 0.184 | 0.640 +/- 0.135 | 0.600 +/- 0.133 | 0.610 +/- 0.139 | 0.622 +/- 0.104 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_curvature_abs_mean | thrashing | 0.822 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.806 | higher -> wrong |
| reasoning_max_rise | shape | 0.783 | higher -> wrong |
| reasoning_n_chunks | length | 0.769 | higher -> wrong |
| reasoning_peak_pos | timing | 0.717 | higher -> correct |
| reasoning_positive_mass | shape | 0.711 | higher -> correct |
| reasoning_late_minus_early | transition | 0.694 | higher -> wrong |
| reasoning_traj_t17 | shape | 0.694 | higher -> wrong |
| reasoning_early_mean | shape | 0.689 | higher -> wrong |
| reasoning_traj_t02 | shape | 0.689 | higher -> wrong |
| reasoning_range | commitment | 0.683 | higher -> wrong |
| reasoning_trough_pos | timing | 0.678 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_peak_pos | timing | 1.194 | higher -> correct |
| reasoning_monotonicity | commitment | 1.172 | higher -> correct |
| reasoning_max_drop | derailment | 1.032 | higher -> correct |
| reasoning_positive_mass | shape | -0.854 | higher -> wrong |
| reasoning_min | shape | -0.641 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.425 | higher -> wrong |
| reasoning_traj_t15 | shape | -0.413 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.364 | higher -> correct |
| reasoning_std | shape | -0.302 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -0.262 | higher -> wrong |
| reasoning_negative_mass | shape | -0.260 | higher -> wrong |
| reasoning_n_chunks | length | -0.250 | higher -> wrong |

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
| premature_exit | inconclusive | 31 | 0.571 [0.367, 0.757] | 0.516 | 3 | - | - | - | - |
| rambling_overlong | inconclusive | 31 | 0.429 [0.243, 0.633] | 0.516 | 4 | - | - | - | - |
| thrashing | inconclusive | 31 | 0.394 [0.205, 0.600] | 0.516 | 3 | - | - | - | - |
| no_commitment | inconclusive | 31 | 0.585 [0.360, 0.800] | 0.516 | 2 | - | - | - | - |
| derailment_late | inconclusive | 31 | 0.631 [0.409, 0.827] | 0.516 | 6 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.311**
- `trajectory_full` ROC-AUC: **0.467**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 16 | 0 | - | - |
| any | 11 | 8 / 16 | 15 | 0.500 | 0.533 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
