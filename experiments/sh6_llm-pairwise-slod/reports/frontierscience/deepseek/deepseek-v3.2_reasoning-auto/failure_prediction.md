# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 153
- Positive class (`is_correct=true`): 61
- Negative class (`is_correct=false`): 92

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only | 3 | 0.771 +/- 0.065 | 0.602 +/- 0.095 | 0.801 +/- 0.036 | 0.771 +/- 0.048 | 0.770 +/- 0.037 |
| trajectory_shape | 60 | 0.884 +/- 0.062 | 0.806 +/- 0.123 | 0.774 +/- 0.065 | 0.778 +/- 0.052 | 0.724 +/- 0.085 |
| trajectory_full | 63 | 0.891 +/- 0.052 | 0.817 +/- 0.111 | 0.781 +/- 0.084 | 0.784 +/- 0.069 | 0.728 +/- 0.112 |
| mode_stack | 10 | 0.870 +/- 0.038 | 0.805 +/- 0.064 | 0.757 +/- 0.063 | 0.738 +/- 0.075 | 0.724 +/- 0.063 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_max_rise | shape | 0.849 | higher -> correct |
| answer_direction_changes | thrashing | 0.837 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.837 | higher -> correct |
| answer_max_drop | derailment | 0.826 | higher -> correct |
| answer_monotonicity | commitment | 0.815 | higher -> correct |
| answer_total_variation | thrashing | 0.814 | higher -> correct |
| answer_zero_crossings | thrashing | 0.787 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.776 | higher -> correct |
| answer_rebound_from_trough | shape | 0.765 | higher -> correct |
| answer_end | landing | 0.726 | higher -> correct |
| answer_n_chunks | length | 0.700 | higher -> correct |
| answer_late_mean | landing | 0.696 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_fall_from_peak | derailment | 1.471 | higher -> correct |
| answer_monotonicity | commitment | 1.421 | higher -> correct |
| answer_early_mean | shape | -1.150 | higher -> wrong |
| reasoning_total_variation | thrashing | -1.068 | higher -> wrong |
| answer_time_positive | shape | -0.930 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.828 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | 0.745 | higher -> correct |
| answer_end | landing | -0.713 | higher -> wrong |
| answer_mid_mean | shape | -0.697 | higher -> wrong |
| answer_negative_mass | shape | 0.683 | higher -> correct |
| answer_late_minus_early | transition | 0.671 | higher -> correct |
| answer_trough_pos | timing | 0.670 | higher -> correct |

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
| premature_exit | inconclusive | 153 | 0.478 [0.385, 0.576] | 0.601 | 17 | - | - | - | - |
| rambling_overlong | inconclusive | 153 | 0.522 [0.424, 0.615] | 0.601 | 25 | - | - | - | - |
| thrashing | inconclusive | 153 | 0.431 [0.342, 0.528] | 0.601 | 7 | - | - | - | - |
| no_commitment | inverted | 153 | 0.403 [0.320, 0.493] | 0.601 | 12 | - | - | - | - |
| derailment_late | inconclusive | 153 | 0.542 [0.450, 0.644] | 0.601 | 23 | - | - | - | - |
| answer_drift | inconclusive | 137 | 0.494 [0.398, 0.588] | 0.555 | 20 | - | - | - | - |
| answer_meandering | confirmed | 137 | 0.864 [0.806, 0.919] | 0.555 | 62 | 0.903 | 0.737 | 0.812 | 1.628 |
| answer_volatility | confirmed | 137 | 0.868 [0.804, 0.927] | 0.555 | 63 | 0.905 | 0.750 | 0.820 | 1.631 |
| answer_uncommitted | confirmed | 137 | 0.812 [0.738, 0.886] | 0.555 | 58 | 0.897 | 0.684 | 0.776 | 1.616 |
| answer_overrange | confirmed | 137 | 0.782 [0.696, 0.861] | 0.555 | 55 | 0.891 | 0.645 | 0.748 | 1.606 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_meandering, answer_volatility, answer_uncommitted, answer_overrange.
- **Hypothesis falsified (inverted)**: no_commitment. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, derailment_late, answer_drift. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.870**
- `trajectory_full` ROC-AUC: **0.891**
- Above-chance discrimination preserved by the mode stack: **94.6%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 58 / 92 | 69 | 0.630 | 0.841 |
| any | 11 | 79 / 92 | 109 | 0.859 | 0.725 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
