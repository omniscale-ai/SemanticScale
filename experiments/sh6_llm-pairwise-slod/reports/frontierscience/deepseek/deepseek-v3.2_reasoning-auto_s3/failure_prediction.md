# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s3 — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 153
- Positive class (`is_correct=true`): 58
- Negative class (`is_correct=false`): 95

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.734 +/- 0.052 | 0.533 +/- 0.058 | 0.753 +/- 0.052 | 0.705 +/- 0.061 | 0.711 +/- 0.049 |
| trajectory_shape (logreg) | 60 | 0.751 +/- 0.052 | 0.631 +/- 0.089 | 0.667 +/- 0.052 | 0.660 +/- 0.041 | 0.599 +/- 0.093 |
| trajectory_full (logreg) | 63 | 0.772 +/- 0.063 | 0.671 +/- 0.091 | 0.669 +/- 0.053 | 0.673 +/- 0.049 | 0.598 +/- 0.075 |
| trajectory_full (lightgbm) | 63 | 0.786 +/- 0.092 | 0.685 +/- 0.129 | 0.696 +/- 0.101 | 0.712 +/- 0.101 | 0.628 +/- 0.120 |
| mode_stack (logreg) | 13 | 0.830 +/- 0.081 | 0.742 +/- 0.127 | 0.738 +/- 0.066 | 0.725 +/- 0.047 | 0.679 +/- 0.084 |
| mode_stack (lightgbm) | 13 | 0.802 +/- 0.048 | 0.675 +/- 0.085 | 0.703 +/- 0.059 | 0.712 +/- 0.048 | 0.628 +/- 0.090 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_total_variation | thrashing | 0.817 | higher -> correct |
| answer_max_drop | derailment | 0.816 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.810 | higher -> correct |
| answer_max_rise | shape | 0.796 | higher -> correct |
| answer_direction_changes | thrashing | 0.755 | higher -> correct |
| answer_monotonicity | commitment | 0.747 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.746 | higher -> correct |
| answer_rebound_from_trough | shape | 0.745 | higher -> correct |
| answer_zero_crossings | thrashing | 0.744 | higher -> correct |
| answer_range | commitment | 0.726 | higher -> correct |
| answer_min | shape | 0.726 | higher -> correct |
| answer_max | shape | 0.721 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_fall_from_peak | derailment | 1.310 | higher -> correct |
| answer_rebound_from_trough | shape | 0.992 | higher -> correct |
| answer_n_chunks | length | -0.835 | higher -> wrong |
| answer_peak_pos | timing | 0.791 | higher -> correct |
| answer_max_drop | derailment | -0.767 | higher -> wrong |
| answer_max_rise | shape | -0.712 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.708 | higher -> correct |
| answer_negative_mass | shape | -0.675 | higher -> wrong |
| answer_time_positive | shape | -0.611 | higher -> wrong |
| reasoning_std | shape | 0.530 | higher -> correct |
| reasoning_max | shape | -0.501 | higher -> wrong |
| answer_std | shape | -0.491 | higher -> wrong |

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
| premature_exit | inconclusive | 153 | 0.505 [0.415, 0.594] | 0.621 | 30 | - | - | - | - |
| rambling_overlong | inconclusive | 153 | 0.495 [0.406, 0.585] | 0.621 | 20 | - | - | - | - |
| thrashing | inconclusive | 152 | 0.415 [0.327, 0.503] | 0.618 | 6 | - | - | - | - |
| no_commitment | inconclusive | 152 | 0.569 [0.472, 0.657] | 0.618 | 28 | - | - | - | - |
| derailment_late | inconclusive | 152 | 0.548 [0.449, 0.646] | 0.618 | 16 | - | - | - | - |
| answer_drift | inconclusive | 138 | 0.545 [0.445, 0.645] | 0.580 | 15 | - | - | - | - |
| answer_meandering | confirmed | 139 | 0.798 [0.734, 0.857] | 0.583 | 54 | 0.926 | 0.617 | 0.741 | 1.589 |
| answer_volatility | confirmed | 139 | 0.795 [0.720, 0.862] | 0.583 | 57 | 0.895 | 0.630 | 0.739 | 1.535 |
| answer_uncommitted | confirmed | 139 | 0.793 [0.724, 0.853] | 0.583 | 57 | 0.895 | 0.630 | 0.739 | 1.535 |
| answer_overrange | confirmed | 138 | 0.751 [0.668, 0.828] | 0.580 | 47 | 0.872 | 0.512 | 0.646 | 1.505 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_meandering, answer_volatility, answer_uncommitted, answer_overrange.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.830**
- `trajectory_full` ROC-AUC: **0.772**
- Above-chance discrimination preserved by the mode stack: **121.2%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 52 / 95 | 61 | 0.547 | 0.852 |
| any | 11 | 78 / 95 | 109 | 0.821 | 0.716 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
