# SH6 swe-agent-trajectories/model-all_steps-50 — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 200
- Positive class (`final_answer_correct=true`): 17
- Negative class (`final_answer_correct=false`): 183
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 200 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 1 | 0.770 +/- 0.047 | 0.233 +/- 0.021 | 0.814 +/- 0.032 | 0.660 +/- 0.056 | 0.335 +/- 0.033 |
| trajectory_shape | 29 | 0.924 +/- 0.020 | 0.389 +/- 0.095 | 0.924 +/- 0.020 | 0.860 +/- 0.037 | 0.554 +/- 0.092 |
| trajectory_full | 29 | 0.924 +/- 0.020 | 0.389 +/- 0.095 | 0.924 +/- 0.020 | 0.860 +/- 0.037 | 0.554 +/- 0.092 |
| mode_stack | 6 | 0.933 +/- 0.029 | 0.434 +/- 0.124 | 0.924 +/- 0.020 | 0.860 +/- 0.037 | 0.554 +/- 0.092 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_end_minus_start | landing | 0.924 | higher -> correct |
| reasoning_total_variation | thrashing | 0.924 | higher -> correct |
| reasoning_peak_pos | timing | 0.924 | higher -> correct |
| reasoning_fall_from_peak | derailment | 0.924 | higher -> correct |
| reasoning_max_rise | shape | 0.924 | higher -> correct |
| reasoning_time_positive | shape | 0.924 | higher -> correct |
| reasoning_time_negative | shape | 0.924 | higher -> correct |
| reasoning_monotonicity | commitment | 0.896 | higher -> correct |
| reasoning_min | shape | 0.858 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.839 | higher -> correct |
| reasoning_positive_mass | shape | 0.836 | higher -> correct |
| reasoning_late_mean | landing | 0.815 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_monotonicity | commitment | 0.606 | higher -> correct |
| reasoning_time_negative | shape | 0.563 | higher -> correct |
| reasoning_time_positive | shape | -0.563 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.493 | higher -> correct |
| reasoning_max_rise | shape | -0.463 | higher -> wrong |
| reasoning_start | shape | 0.435 | higher -> correct |
| reasoning_peak_pos | timing | -0.377 | higher -> wrong |
| reasoning_end_minus_start | landing | -0.347 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.331 | higher -> wrong |
| reasoning_max_drop_pos | derailment | -0.319 | higher -> wrong |
| reasoning_n_chunks | length | 0.279 | higher -> correct |
| reasoning_positive_mass | shape | -0.277 | higher -> wrong |

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
| premature_exit | confirmed | 200 | 0.770 [0.715, 0.824] | 0.915 | 127 | 1.000 | 0.694 | 0.819 | 1.093 |
| rambling_overlong | inverted | 200 | 0.230 [0.176, 0.285] | 0.915 | 28 | - | - | - | - |
| thrashing | inconclusive | 200 | 0.459 [0.396, 0.522] | 0.915 | 64 | - | - | - | - |
| no_commitment | confirmed | 200 | 0.896 [0.860, 0.931] | 0.915 | 150 | 1.000 | 0.820 | 0.901 | 1.093 |
| derailment_late | inverted | 200 | 0.077 [0.050, 0.104] | 0.915 | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | confirmed | 200 | 0.619 [0.522, 0.696] | 0.915 | 67 | 0.970 | 0.355 | 0.520 | 1.060 |

### Verdict summary

- **Confirmed on this run**: premature_exit, no_commitment, truncation_abort.
- **Hypothesis falsified (inverted)**: rambling_overlong, derailment_late. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: thrashing. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.933**
- `trajectory_full` ROC-AUC: **0.924**
- Above-chance discrimination preserved by the mode stack: **102.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 3 | 161 / 183 | 163 | 0.880 | 0.988 |
| any | 11 | 161 / 183 | 163 | 0.880 | 0.988 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
