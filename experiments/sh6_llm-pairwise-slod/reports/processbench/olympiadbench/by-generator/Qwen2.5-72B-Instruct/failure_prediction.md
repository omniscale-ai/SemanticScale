# SH6 processbench/olympiadbench/by-generator/Qwen2.5-72B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 39
- Positive class (`final_answer_correct=true`): 19
- Negative class (`final_answer_correct=false`): 20
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 39 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.383 +/- 0.129 | 0.531 +/- 0.121 | 0.383 +/- 0.067 | 0.382 +/- 0.068 | 0.355 +/- 0.203 |
| trajectory_shape (logreg) | 47 | 0.396 +/- 0.160 | 0.571 +/- 0.105 | 0.383 +/- 0.159 | 0.386 +/- 0.160 | 0.314 +/- 0.195 |
| trajectory_full (logreg) | 47 | 0.396 +/- 0.160 | 0.571 +/- 0.105 | 0.383 +/- 0.159 | 0.386 +/- 0.160 | 0.314 +/- 0.195 |
| reasoning_traj (MiniRocket) | 20 | 0.408 +/- 0.204 | 0.601 +/- 0.158 | 0.367 +/- 0.215 | 0.364 +/- 0.212 | 0.372 +/- 0.135 |
| trajectory_full (lightgbm) | 47 | 0.346 +/- 0.182 | 0.526 +/- 0.118 | 0.442 +/- 0.168 | 0.436 +/- 0.168 | 0.444 +/- 0.128 |
| mode_stack (logreg) | 6 | 0.379 +/- 0.198 | 0.582 +/- 0.133 | 0.492 +/- 0.145 | 0.493 +/- 0.147 | 0.447 +/- 0.126 |
| mode_stack (lightgbm) | 6 | 0.471 +/- 0.142 | 0.561 +/- 0.159 | 0.467 +/- 0.113 | 0.464 +/- 0.111 | 0.471 +/- 0.149 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_trough_pos | timing | 0.769 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.762 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.704 | higher -> wrong |
| reasoning_traj_t12 | shape | 0.700 | higher -> wrong |
| reasoning_traj_t02 | shape | 0.696 | higher -> correct |
| reasoning_min | shape | 0.692 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.692 | higher -> wrong |
| reasoning_traj_t10 | shape | 0.692 | higher -> wrong |
| reasoning_end | landing | 0.688 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.679 | higher -> wrong |
| reasoning_time_positive | shape | 0.675 | higher -> correct |
| reasoning_time_negative | shape | 0.675 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_traj_t15 | shape | -0.900 | higher -> wrong |
| reasoning_positive_mass | shape | -0.848 | higher -> wrong |
| reasoning_traj_t09 | shape | -0.764 | higher -> wrong |
| reasoning_traj_t05 | shape | -0.718 | higher -> wrong |
| reasoning_peak_pos | timing | 0.689 | higher -> correct |
| reasoning_traj_t03 | shape | 0.674 | higher -> correct |
| reasoning_trough_pos | timing | -0.639 | higher -> wrong |
| reasoning_traj_t07 | shape | 0.629 | higher -> correct |
| reasoning_traj_t02 | shape | 0.543 | higher -> correct |
| reasoning_traj_t06 | shape | -0.531 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.462 | higher -> wrong |
| reasoning_max_rise | shape | -0.461 | higher -> wrong |

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
| premature_exit | inconclusive | 39 | 0.413 [0.217, 0.610] | 0.513 | 2 | - | - | - | - |
| rambling_overlong | inconclusive | 39 | 0.587 [0.390, 0.783] | 0.513 | 4 | - | - | - | - |
| thrashing | inconclusive | 39 | 0.537 [0.354, 0.730] | 0.513 | 4 | - | - | - | - |
| no_commitment | inconclusive | 39 | 0.579 [0.387, 0.758] | 0.513 | 4 | - | - | - | - |
| derailment_late | inconclusive | 39 | 0.428 [0.254, 0.630] | 0.513 | 4 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.379**
- `trajectory_full` ROC-AUC: **0.396**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 20 | 0 | - | - |
| any | 11 | 7 / 20 | 14 | 0.350 | 0.500 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
