# SH6 agenthallu/framework-all/by-framework/Camel — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 93
- Positive class (`final_answer_correct=true`): 36
- Negative class (`final_answer_correct=false`): 57
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 93 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.677 +/- 0.135 | 0.663 +/- 0.170 | 0.589 +/- 0.115 | 0.559 +/- 0.107 | 0.557 +/- 0.115 |
| trajectory_shape (logreg) | 60 | 0.461 +/- 0.100 | 0.497 +/- 0.110 | 0.479 +/- 0.054 | 0.493 +/- 0.082 | 0.383 +/- 0.055 |
| trajectory_full (logreg) | 63 | 0.474 +/- 0.105 | 0.506 +/- 0.103 | 0.461 +/- 0.081 | 0.471 +/- 0.104 | 0.376 +/- 0.068 |
| trajectory_full (lightgbm) | 63 | 0.478 +/- 0.148 | 0.499 +/- 0.133 | 0.508 +/- 0.089 | 0.557 +/- 0.089 | 0.334 +/- 0.137 |
| mode_stack (logreg) | 10 | 0.605 +/- 0.105 | 0.601 +/- 0.060 | 0.529 +/- 0.109 | 0.517 +/- 0.104 | 0.476 +/- 0.128 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | 0.687 | higher -> correct |
| answer_time_negative | shape | 0.661 | higher -> correct |
| total_n_chunks | length | 0.658 | higher -> correct |
| answer_negative_mass | shape | 0.647 | higher -> correct |
| answer_min | shape | 0.642 | higher -> correct |
| answer_std | shape | 0.642 | higher -> correct |
| answer_rebound_from_trough | shape | 0.637 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.635 | higher -> correct |
| answer_max_drop | derailment | 0.634 | higher -> correct |
| answer_total_variation | thrashing | 0.632 | higher -> correct |
| answer_max | shape | 0.629 | higher -> correct |
| answer_n_chunks | length | 0.627 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | -0.881 | higher -> wrong |
| answer_max_rise_pos | timing | 0.820 | higher -> correct |
| answer_max_drop_pos | derailment | -0.708 | higher -> wrong |
| answer_peak_pos | timing | -0.640 | higher -> wrong |
| reasoning_peak_pos | timing | 0.614 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.604 | higher -> wrong |
| answer_direction_changes | thrashing | -0.565 | higher -> wrong |
| answer_n_chunks | length | -0.558 | higher -> wrong |
| answer_start | shape | -0.473 | higher -> wrong |
| answer_negative_mass | shape | -0.445 | higher -> wrong |
| answer_zero_crossings | thrashing | 0.409 | higher -> correct |
| reasoning_max_rise_pos | timing | -0.360 | higher -> wrong |

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
| premature_exit | inconclusive | 93 | 0.437 [0.319, 0.564] | 0.613 | 6 | - | - | - | - |
| rambling_overlong | inconclusive | 93 | 0.563 [0.436, 0.681] | 0.613 | 11 | - | - | - | - |
| thrashing | inconclusive | 93 | 0.566 [0.445, 0.676] | 0.613 | 12 | - | - | - | - |
| no_commitment | inconclusive | 93 | 0.592 [0.468, 0.716] | 0.613 | 16 | - | - | - | - |
| derailment_late | inconclusive | 93 | 0.549 [0.428, 0.670] | 0.613 | 13 | - | - | - | - |
| answer_drift | confirmed | 93 | 0.621 [0.503, 0.734] | 0.613 | 13 | 0.692 | 0.158 | 0.257 | 1.130 |
| answer_meandering | confirmed | 93 | 0.626 [0.520, 0.728] | 0.613 | 12 | 0.667 | 0.140 | 0.232 | 1.088 |
| answer_volatility | inconclusive | 93 | 0.597 [0.489, 0.704] | 0.613 | 14 | - | - | - | - |
| answer_uncommitted | inverted | 93 | 0.374 [0.263, 0.490] | 0.613 | 0 | - | - | - | - |
| answer_overrange | inconclusive | 93 | 0.563 [0.450, 0.673] | 0.613 | 19 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_drift, answer_meandering.
- **Hypothesis falsified (inverted)**: answer_uncommitted. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_volatility, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.605**
- `trajectory_full` ROC-AUC: **0.474**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 2 | 15 / 57 | 22 | 0.263 | 0.682 |
| any | 11 | 40 / 57 | 57 | 0.702 | 0.702 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
