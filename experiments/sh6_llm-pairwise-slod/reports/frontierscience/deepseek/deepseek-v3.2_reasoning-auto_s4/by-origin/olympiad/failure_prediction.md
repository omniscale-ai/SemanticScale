# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s4/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 97
- Positive class (`is_correct=true`): 58
- Negative class (`is_correct=false`): 39

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.367 +/- 0.155 | 0.556 +/- 0.092 | 0.496 +/- 0.088 | 0.556 +/- 0.093 | 0.681 +/- 0.080 |
| trajectory_shape (logreg) | 60 | 0.482 +/- 0.061 | 0.634 +/- 0.030 | 0.507 +/- 0.057 | 0.505 +/- 0.045 | 0.546 +/- 0.040 |
| trajectory_full (logreg) | 63 | 0.481 +/- 0.054 | 0.625 +/- 0.033 | 0.537 +/- 0.079 | 0.535 +/- 0.065 | 0.578 +/- 0.053 |
| trajectory_full (lightgbm) | 63 | 0.708 +/- 0.076 | 0.745 +/- 0.045 | 0.709 +/- 0.066 | 0.723 +/- 0.081 | 0.761 +/- 0.092 |
| mode_stack (logreg) | 13 | 0.405 +/- 0.131 | 0.598 +/- 0.113 | 0.413 +/- 0.115 | 0.431 +/- 0.122 | 0.494 +/- 0.159 |
| mode_stack (lightgbm) | 13 | 0.711 +/- 0.061 | 0.759 +/- 0.059 | 0.672 +/- 0.044 | 0.691 +/- 0.034 | 0.747 +/- 0.039 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_rebound_from_trough | shape | 0.645 | higher -> wrong |
| reasoning_positive_mass | shape | 0.644 | higher -> wrong |
| answer_n_chunks | length | 0.636 | higher -> wrong |
| reasoning_late_mean | landing | 0.636 | higher -> wrong |
| answer_end_minus_reasoning_end | landing | 0.634 | higher -> wrong |
| reasoning_end | landing | 0.625 | higher -> wrong |
| reasoning_std | shape | 0.623 | higher -> wrong |
| reasoning_max_rise | shape | 0.621 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.620 | higher -> wrong |
| reasoning_max | shape | 0.620 | higher -> wrong |
| reasoning_range | commitment | 0.616 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.614 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_zero_crossings | thrashing | 1.114 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | -0.950 | higher -> wrong |
| answer_peak_pos | timing | 0.885 | higher -> correct |
| answer_rebound_from_trough | shape | 0.860 | higher -> correct |
| answer_end_minus_start | landing | -0.767 | higher -> wrong |
| answer_time_positive | shape | 0.671 | higher -> correct |
| answer_late_minus_early | transition | -0.662 | higher -> wrong |
| answer_late_mean | landing | -0.630 | higher -> wrong |
| answer_zero_crossings | thrashing | 0.615 | higher -> correct |
| answer_early_mean | shape | 0.596 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.590 | higher -> wrong |
| answer_max_rise_pos | timing | -0.580 | higher -> wrong |

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
| premature_exit | inconclusive | 97 | 0.482 [0.354, 0.601] | 0.402 | 12 | - | - | - | - |
| rambling_overlong | inconclusive | 97 | 0.518 [0.399, 0.646] | 0.402 | 13 | - | - | - | - |
| thrashing | inconclusive | 97 | 0.466 [0.358, 0.589] | 0.402 | 0 | - | - | - | - |
| no_commitment | inconclusive | 97 | 0.519 [0.400, 0.643] | 0.402 | 12 | - | - | - | - |
| derailment_late | inconclusive | 97 | 0.494 [0.380, 0.615] | 0.402 | 9 | - | - | - | - |
| answer_drift | inconclusive | 83 | 0.459 [0.334, 0.582] | 0.301 | 6 | - | - | - | - |
| answer_meandering | inconclusive | 83 | 0.511 [0.429, 0.598] | 0.301 | 7 | - | - | - | - |
| answer_volatility | inconclusive | 83 | 0.578 [0.442, 0.708] | 0.301 | 8 | - | - | - | - |
| answer_uncommitted | inconclusive | 83 | 0.516 [0.423, 0.614] | 0.301 | 9 | - | - | - | - |
| answer_overrange | inconclusive | 83 | 0.420 [0.279, 0.573] | 0.301 | 8 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.405**
- `trajectory_full` ROC-AUC: **0.481**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 39 | 0 | - | - |
| any | 11 | 22 / 39 | 48 | 0.564 | 0.458 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
