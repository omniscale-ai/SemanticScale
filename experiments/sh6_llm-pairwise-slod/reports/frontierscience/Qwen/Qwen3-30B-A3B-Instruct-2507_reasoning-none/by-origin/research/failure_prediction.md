# SH6 frontierscience/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none/by-origin/research — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 60
- Positive class (`is_correct=true`): 33
- Negative class (`is_correct=false`): 27

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.750 +/- 0.099 | 0.820 +/- 0.109 | 0.679 +/- 0.131 | 0.683 +/- 0.122 | 0.720 +/- 0.124 |
| trajectory_shape (logreg) | 29 | 0.694 +/- 0.141 | 0.767 +/- 0.113 | 0.610 +/- 0.133 | 0.617 +/- 0.113 | 0.658 +/- 0.093 |
| trajectory_full (logreg) | 29 | 0.694 +/- 0.141 | 0.767 +/- 0.113 | 0.610 +/- 0.133 | 0.617 +/- 0.113 | 0.658 +/- 0.093 |
| trajectory_full (lightgbm) | 29 | 0.700 +/- 0.090 | 0.772 +/- 0.066 | 0.643 +/- 0.120 | 0.650 +/- 0.122 | 0.688 +/- 0.125 |
| mode_stack (logreg) | 4 | 0.707 +/- 0.112 | 0.786 +/- 0.098 | 0.604 +/- 0.066 | 0.617 +/- 0.067 | 0.673 +/- 0.068 |
| mode_stack (lightgbm) | 4 | 0.717 +/- 0.112 | 0.783 +/- 0.116 | 0.593 +/- 0.095 | 0.600 +/- 0.082 | 0.624 +/- 0.117 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_n_chunks | length | 0.750 | higher -> correct |
| answer_early_mean | shape | 0.679 | higher -> correct |
| answer_min | shape | 0.656 | higher -> wrong |
| answer_zero_crossings | thrashing | 0.655 | higher -> correct |
| answer_max | shape | 0.650 | higher -> correct |
| answer_max_drop_pos | derailment | 0.646 | higher -> correct |
| answer_late_minus_early | transition | 0.644 | higher -> correct |
| answer_max_rise_pos | timing | 0.638 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.629 | higher -> wrong |
| answer_negative_mass | shape | 0.622 | higher -> wrong |
| answer_std | shape | 0.606 | higher -> wrong |
| answer_late_mean | landing | 0.599 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | -0.872 | higher -> wrong |
| answer_max | shape | 0.787 | higher -> correct |
| answer_max_drop | derailment | 0.630 | higher -> correct |
| answer_total_variation | thrashing | -0.623 | higher -> wrong |
| answer_peak_pos | timing | 0.585 | higher -> correct |
| answer_std | shape | -0.579 | higher -> wrong |
| answer_range | commitment | 0.480 | higher -> correct |
| answer_max_drop_pos | derailment | -0.473 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.375 | higher -> correct |
| answer_max_rise | shape | 0.374 | higher -> correct |
| answer_time_negative | shape | 0.342 | higher -> correct |
| answer_time_positive | shape | -0.342 | higher -> wrong |

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
| premature_exit | inconclusive | 60 | 0.500 [0.500, 0.500] | 0.450 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 60 | 0.500 [0.500, 0.500] | 0.450 | 0 | - | - | - | - |
| thrashing | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| no_commitment | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| derailment_late | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | inconclusive | 60 | 0.554 [0.405, 0.700] | 0.450 | 1 | - | - | - | - |
| answer_volatility | inconclusive | 60 | 0.459 [0.305, 0.607] | 0.450 | 4 | - | - | - | - |
| answer_uncommitted | inconclusive | 60 | 0.562 [0.411, 0.715] | 0.450 | 6 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, answer_meandering, answer_volatility, answer_uncommitted. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.707**
- `trajectory_full` ROC-AUC: **0.694**
- Above-chance discrimination preserved by the mode stack: **106.4%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 27 | 0 | - | - |
| any | 11 | 3 / 27 | 11 | 0.111 | 0.273 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
