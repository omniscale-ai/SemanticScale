# SH6 agenthallu/framework-all/by-framework/OpenDeepSearch — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 100
- Positive class (`final_answer_correct=true`): 42
- Negative class (`final_answer_correct=false`): 58
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 100 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.564 +/- 0.076 | 0.484 +/- 0.067 | 0.557 +/- 0.082 | 0.550 +/- 0.063 | 0.517 +/- 0.140 |
| trajectory_shape (logreg) | 96 | 0.451 +/- 0.079 | 0.434 +/- 0.058 | 0.520 +/- 0.073 | 0.510 +/- 0.049 | 0.454 +/- 0.129 |
| trajectory_full (logreg) | 99 | 0.435 +/- 0.066 | 0.424 +/- 0.044 | 0.499 +/- 0.101 | 0.490 +/- 0.080 | 0.431 +/- 0.154 |
| reasoning_traj (MiniRocket) | 20 | 0.489 +/- 0.118 | 0.440 +/- 0.058 | 0.473 +/- 0.102 | 0.480 +/- 0.103 | 0.411 +/- 0.099 |
| trajectory_full (lightgbm) | 99 | 0.433 +/- 0.137 | 0.406 +/- 0.059 | 0.433 +/- 0.104 | 0.460 +/- 0.102 | 0.264 +/- 0.152 |
| mode_stack (logreg) | 13 | 0.618 +/- 0.088 | 0.558 +/- 0.077 | 0.565 +/- 0.100 | 0.560 +/- 0.116 | 0.537 +/- 0.074 |
| mode_stack (lightgbm) | 13 | 0.394 +/- 0.114 | 0.438 +/- 0.086 | 0.458 +/- 0.095 | 0.480 +/- 0.098 | 0.321 +/- 0.152 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max_rise | shape | 0.634 | higher -> correct |
| answer_end_minus_reasoning_end | landing | 0.629 | higher -> correct |
| answer_start_minus_reasoning_end | landing | 0.626 | higher -> correct |
| reasoning_end | landing | 0.599 | higher -> correct |
| reasoning_start | shape | 0.599 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.597 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.596 | higher -> correct |
| reasoning_end_minus_start | landing | 0.591 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.589 | higher -> correct |
| reasoning_traj_t04 | shape | 0.584 | higher -> correct |
| reasoning_traj_t05 | shape | 0.582 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.580 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise | shape | -0.808 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.753 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.678 | higher -> correct |
| answer_peak_pos | timing | -0.583 | higher -> wrong |
| answer_trough_pos | timing | -0.478 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.430 | higher -> wrong |
| reasoning_start | shape | 0.426 | higher -> correct |
| reasoning_min | shape | -0.415 | higher -> wrong |
| reasoning_peak_pos | timing | 0.414 | higher -> correct |
| answer_n_chunks | length | 0.401 | higher -> correct |
| answer_monotonicity | commitment | -0.384 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.378 | higher -> correct |

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
| premature_exit | inconclusive | 100 | 0.448 [0.332, 0.558] | 0.580 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 100 | 0.552 [0.442, 0.668] | 0.580 | 12 | - | - | - | - |
| thrashing | inconclusive | 100 | 0.570 [0.472, 0.667] | 0.580 | 7 | - | - | - | - |
| no_commitment | inconclusive | 100 | 0.508 [0.396, 0.626] | 0.580 | 8 | - | - | - | - |
| derailment_late | inconclusive | 100 | 0.443 [0.332, 0.559] | 0.580 | 9 | - | - | - | - |
| answer_drift | inconclusive | 88 | 0.408 [0.285, 0.535] | 0.580 | 8 | - | - | - | - |
| answer_meandering | inconclusive | 88 | 0.513 [0.460, 0.562] | 0.580 | 6 | - | - | - | - |
| answer_volatility | inconclusive | 88 | 0.516 [0.444, 0.576] | 0.580 | 7 | - | - | - | - |
| answer_uncommitted | inconclusive | 88 | 0.471 [0.406, 0.536] | 0.580 | 0 | - | - | - | - |
| answer_overrange | inconclusive | 88 | 0.472 [0.337, 0.604] | 0.580 | 9 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.618**
- `trajectory_full` ROC-AUC: **0.435**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 58 | 0 | - | - |
| any | 11 | 24 / 58 | 43 | 0.414 | 0.558 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
