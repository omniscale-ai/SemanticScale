# SH6 processbench/olympiadbench/by-generator/Llama-3.1-70B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 51
- Positive class (`final_answer_correct=true`): 21
- Negative class (`final_answer_correct=false`): 30
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 51 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.623 +/- 0.114 | 0.534 +/- 0.109 | 0.642 +/- 0.133 | 0.644 +/- 0.108 | 0.534 +/- 0.282 |
| trajectory_shape (logreg) | 29 | 0.562 +/- 0.186 | 0.552 +/- 0.168 | 0.507 +/- 0.104 | 0.511 +/- 0.101 | 0.425 +/- 0.147 |
| trajectory_full (logreg) | 29 | 0.562 +/- 0.186 | 0.552 +/- 0.168 | 0.507 +/- 0.104 | 0.511 +/- 0.101 | 0.425 +/- 0.147 |
| trajectory_full (lightgbm) | 29 | 0.445 +/- 0.097 | 0.517 +/- 0.095 | 0.403 +/- 0.096 | 0.415 +/- 0.123 | 0.317 +/- 0.069 |
| mode_stack (logreg) | 5 | 0.387 +/- 0.094 | 0.438 +/- 0.088 | 0.482 +/- 0.113 | 0.489 +/- 0.079 | 0.359 +/- 0.218 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_end | landing | 0.697 | higher -> wrong |
| reasoning_max_drop | derailment | 0.660 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.650 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.643 | higher -> correct |
| reasoning_trough_pos | timing | 0.636 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.631 | higher -> wrong |
| reasoning_n_chunks | length | 0.623 | higher -> correct |
| reasoning_end_minus_start | landing | 0.620 | higher -> wrong |
| reasoning_early_mean | shape | 0.618 | higher -> correct |
| reasoning_time_positive | shape | 0.613 | higher -> correct |
| reasoning_time_negative | shape | 0.613 | higher -> correct |
| reasoning_max_rise | shape | 0.607 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | 1.059 | higher -> correct |
| reasoning_max_drop | derailment | -0.837 | higher -> wrong |
| reasoning_peak_pos | timing | -0.715 | higher -> wrong |
| reasoning_start | shape | -0.652 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.631 | higher -> correct |
| reasoning_time_positive | shape | -0.592 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.589 | higher -> correct |
| reasoning_time_negative | shape | 0.563 | higher -> correct |
| reasoning_end_minus_start | landing | 0.497 | higher -> correct |
| reasoning_trough_pos | timing | 0.483 | higher -> correct |
| reasoning_max | shape | 0.435 | higher -> correct |
| reasoning_range | commitment | 0.411 | higher -> correct |

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
| premature_exit | inconclusive | 51 | 0.367 [0.215, 0.532] | 0.588 | 1 | - | - | - | - |
| rambling_overlong | inconclusive | 51 | 0.633 [0.468, 0.785] | 0.588 | 10 | - | - | - | - |
| thrashing | inconclusive | 51 | 0.593 [0.427, 0.736] | 0.588 | 6 | - | - | - | - |
| no_commitment | inconclusive | 51 | 0.547 [0.375, 0.718] | 0.588 | 5 | - | - | - | - |
| derailment_late | inconclusive | 51 | 0.498 [0.327, 0.662] | 0.588 | 6 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.387**
- `trajectory_full` ROC-AUC: **0.562**
- Above-chance discrimination preserved by the mode stack: **-183.8%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 30 | 0 | - | - |
| any | 11 | 12 / 30 | 19 | 0.400 | 0.632 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
