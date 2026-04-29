# SH6 processbench/gsm8k/by-generator/Llama-3.1-8B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 61
- Positive class (`final_answer_correct=true`): 24
- Negative class (`final_answer_correct=false`): 37
- Label agreement (`is_correct` vs `final_answer_correct`): 98.4% over 61 items
- Final answer correct but reasoning wrong: 1
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.579 +/- 0.174 | 0.485 +/- 0.101 | 0.550 +/- 0.176 | 0.558 +/- 0.170 | 0.472 +/- 0.182 |
| trajectory_shape (logreg) | 47 | 0.503 +/- 0.124 | 0.465 +/- 0.097 | 0.525 +/- 0.097 | 0.540 +/- 0.072 | 0.416 +/- 0.156 |
| trajectory_full (logreg) | 47 | 0.503 +/- 0.124 | 0.465 +/- 0.097 | 0.525 +/- 0.097 | 0.540 +/- 0.072 | 0.416 +/- 0.156 |
| reasoning_traj (MiniRocket) | 20 | 0.341 +/- 0.216 | 0.423 +/- 0.147 | 0.341 +/- 0.109 | 0.374 +/- 0.098 | 0.193 +/- 0.171 |
| trajectory_full (lightgbm) | 47 | 0.449 +/- 0.158 | 0.512 +/- 0.129 | 0.526 +/- 0.120 | 0.573 +/- 0.099 | 0.356 +/- 0.198 |
| mode_stack (logreg) | 6 | 0.611 +/- 0.210 | 0.589 +/- 0.178 | 0.556 +/- 0.176 | 0.554 +/- 0.176 | 0.511 +/- 0.181 |
| mode_stack (lightgbm) | 6 | 0.467 +/- 0.192 | 0.499 +/- 0.150 | 0.482 +/- 0.159 | 0.488 +/- 0.168 | 0.421 +/- 0.150 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_end_minus_start | landing | 0.729 | higher -> wrong |
| reasoning_start | shape | 0.691 | higher -> wrong |
| reasoning_rebound_from_trough | shape | 0.679 | higher -> wrong |
| reasoning_end | landing | 0.679 | higher -> wrong |
| reasoning_max_drop | derailment | 0.668 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.666 | higher -> correct |
| reasoning_positive_mass | shape | 0.666 | higher -> wrong |
| reasoning_traj_t18 | shape | 0.661 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.657 | higher -> correct |
| reasoning_max_rise | shape | 0.654 | higher -> wrong |
| reasoning_traj_t01 | shape | 0.651 | higher -> wrong |
| reasoning_time_positive | shape | 0.648 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise_pos | timing | -0.993 | higher -> wrong |
| reasoning_n_chunks | length | -0.854 | higher -> wrong |
| reasoning_max_rise | shape | 0.685 | higher -> correct |
| reasoning_max_drop | derailment | -0.631 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.555 | higher -> correct |
| reasoning_monotonicity | commitment | 0.497 | higher -> correct |
| reasoning_traj_t04 | shape | -0.492 | higher -> wrong |
| reasoning_range | commitment | 0.391 | higher -> correct |
| reasoning_max | shape | 0.381 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.377 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.367 | higher -> wrong |
| reasoning_positive_mass | shape | -0.308 | higher -> wrong |

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
| premature_exit | inconclusive | 61 | 0.443 [0.312, 0.588] | 0.607 | 7 | - | - | - | - |
| rambling_overlong | inconclusive | 61 | 0.557 [0.412, 0.688] | 0.607 | 7 | - | - | - | - |
| thrashing | inconclusive | 61 | 0.409 [0.262, 0.553] | 0.607 | 2 | - | - | - | - |
| no_commitment | inconclusive | 61 | 0.624 [0.461, 0.761] | 0.607 | 10 | - | - | - | - |
| derailment_late | inconclusive | 61 | 0.446 [0.296, 0.599] | 0.607 | 6 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.611**
- `trajectory_full` ROC-AUC: **0.503**
- Above-chance discrimination preserved by the mode stack: **3458.3%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 37 | 0 | - | - |
| any | 11 | 17 / 37 | 27 | 0.459 | 0.630 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
