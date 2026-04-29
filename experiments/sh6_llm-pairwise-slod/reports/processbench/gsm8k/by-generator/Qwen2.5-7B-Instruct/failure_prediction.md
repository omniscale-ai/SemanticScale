# SH6 processbench/gsm8k/by-generator/Qwen2.5-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 27
- Positive class (`final_answer_correct=true`): 15
- Negative class (`final_answer_correct=false`): 12
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 27 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.444 +/- 0.238 | 0.589 +/- 0.126 | 0.517 +/- 0.178 | 0.513 +/- 0.208 | 0.516 +/- 0.301 |
| trajectory_shape (logreg) | 47 | 0.522 +/- 0.301 | 0.701 +/- 0.208 | 0.533 +/- 0.201 | 0.507 +/- 0.210 | 0.485 +/- 0.288 |
| trajectory_full (logreg) | 47 | 0.522 +/- 0.301 | 0.701 +/- 0.208 | 0.533 +/- 0.201 | 0.507 +/- 0.210 | 0.485 +/- 0.288 |
| reasoning_traj (MiniRocket) | 20 | 0.511 +/- 0.259 | 0.702 +/- 0.136 | 0.533 +/- 0.201 | 0.507 +/- 0.210 | 0.466 +/- 0.278 |
| trajectory_full (lightgbm) | 47 | 0.300 +/- 0.306 | 0.603 +/- 0.180 | 0.417 +/- 0.307 | 0.407 +/- 0.298 | 0.430 +/- 0.283 |
| mode_stack (logreg) | 6 | 0.467 +/- 0.240 | 0.662 +/- 0.153 | 0.533 +/- 0.227 | 0.527 +/- 0.243 | 0.511 +/- 0.309 |
| mode_stack (lightgbm) | 6 | 0.222 +/- 0.176 | 0.506 +/- 0.061 | 0.300 +/- 0.194 | 0.287 +/- 0.173 | 0.181 +/- 0.234 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_min | shape | 0.889 | higher -> wrong |
| reasoning_traj_t05 | shape | 0.789 | higher -> correct |
| reasoning_std | shape | 0.778 | higher -> wrong |
| reasoning_traj_t15 | shape | 0.778 | higher -> wrong |
| reasoning_traj_t16 | shape | 0.756 | higher -> wrong |
| reasoning_traj_t01 | shape | 0.744 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.717 | higher -> correct |
| reasoning_range | commitment | 0.711 | higher -> wrong |
| reasoning_traj_t04 | shape | 0.711 | higher -> correct |
| reasoning_traj_t13 | shape | 0.700 | higher -> wrong |
| reasoning_negative_mass | shape | 0.700 | higher -> wrong |
| reasoning_traj_t06 | shape | 0.700 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | -1.029 | higher -> wrong |
| reasoning_max_rise_pos | timing | -0.615 | higher -> wrong |
| reasoning_range | commitment | 0.608 | higher -> correct |
| reasoning_max_drop | derailment | 0.529 | higher -> correct |
| reasoning_max | shape | 0.519 | higher -> correct |
| reasoning_monotonicity | commitment | -0.501 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -0.487 | higher -> wrong |
| reasoning_positive_mass | shape | -0.467 | higher -> wrong |
| reasoning_trough_pos | timing | -0.462 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.438 | higher -> wrong |
| reasoning_min | shape | -0.373 | higher -> wrong |
| reasoning_peak_pos | timing | 0.370 | higher -> correct |

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
| premature_exit | inconclusive | 27 | 0.494 [0.278, 0.728] | 0.444 | 1 | - | - | - | - |
| rambling_overlong | inconclusive | 27 | 0.506 [0.272, 0.722] | 0.444 | 3 | - | - | - | - |
| thrashing | inconclusive | 27 | 0.478 [0.239, 0.711] | 0.444 | 5 | - | - | - | - |
| no_commitment | inconclusive | 27 | 0.389 [0.181, 0.645] | 0.444 | 2 | - | - | - | - |
| derailment_late | inconclusive | 27 | 0.589 [0.353, 0.807] | 0.444 | 7 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.467**
- `trajectory_full` ROC-AUC: **0.522**
- Above-chance discrimination preserved by the mode stack: **-150.0%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 12 | 0 | - | - |
| any | 11 | 9 / 12 | 14 | 0.750 | 0.643 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
