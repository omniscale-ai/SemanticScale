# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 95
- Positive class (`is_correct=true`): 61
- Negative class (`is_correct=false`): 34

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.663 +/- 0.196 | 0.791 +/- 0.130 | 0.670 +/- 0.157 | 0.705 +/- 0.136 | 0.776 +/- 0.104 |
| trajectory_shape (logreg) | 60 | 0.596 +/- 0.155 | 0.721 +/- 0.109 | 0.617 +/- 0.105 | 0.642 +/- 0.102 | 0.714 +/- 0.084 |
| trajectory_full (logreg) | 63 | 0.680 +/- 0.133 | 0.790 +/- 0.091 | 0.703 +/- 0.078 | 0.726 +/- 0.070 | 0.786 +/- 0.059 |
| trajectory_full (lightgbm) | 63 | 0.733 +/- 0.090 | 0.784 +/- 0.091 | 0.746 +/- 0.074 | 0.789 +/- 0.074 | 0.845 +/- 0.060 |
| mode_stack (logreg) | 10 | 0.637 +/- 0.104 | 0.752 +/- 0.053 | 0.633 +/- 0.131 | 0.642 +/- 0.150 | 0.679 +/- 0.185 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_positive_mass | shape | 0.733 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.697 | higher -> wrong |
| answer_total_variation | thrashing | 0.660 | higher -> wrong |
| reasoning_max_drop | derailment | 0.659 | higher -> wrong |
| answer_max_drop | derailment | 0.650 | higher -> wrong |
| total_n_chunks | length | 0.649 | higher -> correct |
| reasoning_n_chunks | length | 0.646 | higher -> correct |
| answer_std | shape | 0.638 | higher -> wrong |
| answer_min | shape | 0.637 | higher -> wrong |
| answer_end_minus_reasoning_end | landing | 0.625 | higher -> correct |
| reasoning_mid_mean | shape | 0.623 | higher -> wrong |
| answer_start | shape | 0.621 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | 1.761 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | -0.984 | higher -> wrong |
| answer_monotonicity | commitment | 0.915 | higher -> correct |
| answer_zero_crossings | thrashing | -0.770 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | 0.724 | higher -> correct |
| answer_peak_pos | timing | -0.682 | higher -> wrong |
| answer_time_positive | shape | -0.630 | higher -> wrong |
| answer_time_negative | shape | -0.630 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.602 | higher -> wrong |
| answer_rebound_from_trough | shape | 0.582 | higher -> correct |
| answer_end | landing | -0.577 | higher -> wrong |
| answer_max | shape | 0.565 | higher -> correct |

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
| premature_exit | inverted | 95 | 0.351 [0.238, 0.473] | 0.358 | 7 | - | - | - | - |
| rambling_overlong | confirmed | 95 | 0.649 [0.527, 0.762] | 0.358 | 18 | 0.667 | 0.353 | 0.462 | 1.863 |
| thrashing | inconclusive | 95 | 0.489 [0.372, 0.606] | 0.358 | 3 | - | - | - | - |
| no_commitment | inconclusive | 95 | 0.419 [0.285, 0.553] | 0.358 | 10 | - | - | - | - |
| derailment_late | inconclusive | 95 | 0.574 [0.454, 0.696] | 0.358 | 12 | - | - | - | - |
| answer_drift | inconclusive | 85 | 0.475 [0.335, 0.618] | 0.282 | 9 | - | - | - | - |
| answer_meandering | inconclusive | 85 | 0.582 [0.481, 0.700] | 0.282 | 10 | - | - | - | - |
| answer_volatility | inconclusive | 85 | 0.582 [0.443, 0.720] | 0.282 | 11 | - | - | - | - |
| answer_uncommitted | inconclusive | 85 | 0.559 [0.453, 0.676] | 0.282 | 9 | - | - | - | - |
| answer_overrange | inconclusive | 85 | 0.542 [0.406, 0.691] | 0.282 | 7 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.637**
- `trajectory_full` ROC-AUC: **0.680**
- Above-chance discrimination preserved by the mode stack: **76.1%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 12 / 34 | 18 | 0.353 | 0.667 |
| any | 11 | 22 / 34 | 51 | 0.647 | 0.431 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
