# SH6 processbench/gsm8k/by-generator/Qwen2-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 52
- Positive class (`final_answer_correct=true`): 21
- Negative class (`final_answer_correct=false`): 31
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 52 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.629 +/- 0.155 | 0.587 +/- 0.111 | 0.601 +/- 0.095 | 0.576 +/- 0.094 | 0.583 +/- 0.118 |
| trajectory_shape (logreg) | 47 | 0.588 +/- 0.165 | 0.612 +/- 0.145 | 0.579 +/- 0.143 | 0.578 +/- 0.141 | 0.534 +/- 0.164 |
| trajectory_full (logreg) | 47 | 0.588 +/- 0.165 | 0.612 +/- 0.145 | 0.579 +/- 0.143 | 0.578 +/- 0.141 | 0.534 +/- 0.164 |
| reasoning_traj (MiniRocket) | 20 | 0.512 +/- 0.093 | 0.500 +/- 0.124 | 0.534 +/- 0.074 | 0.538 +/- 0.068 | 0.448 +/- 0.146 |
| trajectory_full (lightgbm) | 47 | 0.778 +/- 0.138 | 0.764 +/- 0.147 | 0.631 +/- 0.140 | 0.656 +/- 0.140 | 0.552 +/- 0.183 |
| mode_stack (logreg) | 6 | 0.666 +/- 0.275 | 0.688 +/- 0.232 | 0.649 +/- 0.165 | 0.655 +/- 0.144 | 0.578 +/- 0.220 |
| mode_stack (lightgbm) | 6 | 0.698 +/- 0.175 | 0.732 +/- 0.150 | 0.674 +/- 0.117 | 0.673 +/- 0.113 | 0.626 +/- 0.147 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_direction_changes | thrashing | 0.717 | higher -> correct |
| reasoning_positive_mass | shape | 0.698 | higher -> correct |
| reasoning_traj_t04 | shape | 0.678 | higher -> wrong |
| reasoning_mid_mean | shape | 0.675 | higher -> wrong |
| reasoning_traj_t11 | shape | 0.670 | higher -> wrong |
| reasoning_traj_t10 | shape | 0.669 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.664 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.643 | higher -> correct |
| reasoning_traj_t09 | shape | 0.641 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.639 | higher -> correct |
| reasoning_traj_t12 | shape | 0.639 | higher -> wrong |
| reasoning_traj_t03 | shape | 0.634 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise_pos | timing | 1.190 | higher -> correct |
| reasoning_direction_changes | thrashing | -1.120 | higher -> wrong |
| reasoning_max_rise | shape | 0.816 | higher -> correct |
| reasoning_monotonicity | commitment | 0.735 | higher -> correct |
| reasoning_start | shape | 0.698 | higher -> correct |
| reasoning_traj_t08 | shape | -0.528 | higher -> wrong |
| reasoning_traj_t11 | shape | 0.509 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.503 | higher -> correct |
| reasoning_end_minus_start | landing | -0.429 | higher -> wrong |
| reasoning_n_chunks | length | -0.428 | higher -> wrong |
| reasoning_traj_t01 | shape | 0.417 | higher -> correct |
| reasoning_traj_t03 | shape | -0.380 | higher -> wrong |

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
| premature_exit | inconclusive | 52 | 0.356 [0.208, 0.512] | 0.596 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 52 | 0.644 [0.488, 0.792] | 0.596 | 6 | - | - | - | - |
| thrashing | confirmed | 52 | 0.732 [0.579, 0.868] | 0.596 | 5 | 0.600 | 0.097 | 0.167 | 1.006 |
| no_commitment | inconclusive | 52 | 0.582 [0.388, 0.760] | 0.596 | 0 | - | - | - | - |
| derailment_late | inconclusive | 52 | 0.482 [0.333, 0.642] | 0.596 | 5 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: thrashing.
- **Inconclusive**: premature_exit, rambling_overlong, no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.666**
- `trajectory_full` ROC-AUC: **0.588**
- Above-chance discrimination preserved by the mode stack: **188.4%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 3 / 31 | 5 | 0.097 | 0.600 |
| any | 11 | 8 / 31 | 13 | 0.258 | 0.615 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
