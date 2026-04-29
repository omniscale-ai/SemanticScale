# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 153
- Positive class (`is_correct=true`): 61
- Negative class (`is_correct=false`): 92

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.765 +/- 0.065 | 0.593 +/- 0.097 | 0.785 +/- 0.048 | 0.751 +/- 0.064 | 0.756 +/- 0.044 |
| trajectory_shape (logreg) | 60 | 0.837 +/- 0.085 | 0.748 +/- 0.133 | 0.741 +/- 0.088 | 0.738 +/- 0.081 | 0.692 +/- 0.108 |
| trajectory_full (logreg) | 63 | 0.834 +/- 0.085 | 0.740 +/- 0.122 | 0.754 +/- 0.086 | 0.751 +/- 0.073 | 0.704 +/- 0.109 |
| trajectory_full (lightgbm) | 63 | 0.886 +/- 0.058 | 0.791 +/- 0.102 | 0.828 +/- 0.087 | 0.830 +/- 0.079 | 0.787 +/- 0.111 |
| mode_stack (logreg) | 13 | 0.866 +/- 0.049 | 0.794 +/- 0.091 | 0.754 +/- 0.059 | 0.738 +/- 0.063 | 0.717 +/- 0.064 |
| mode_stack (lightgbm) | 13 | 0.867 +/- 0.055 | 0.783 +/- 0.090 | 0.793 +/- 0.091 | 0.790 +/- 0.087 | 0.752 +/- 0.103 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_max_rise | shape | 0.851 | higher -> correct |
| answer_direction_changes | thrashing | 0.840 | higher -> correct |
| answer_max_drop | derailment | 0.832 | higher -> correct |
| answer_curvature_abs_mean | thrashing | 0.829 | higher -> correct |
| answer_total_variation | thrashing | 0.827 | higher -> correct |
| answer_monotonicity | commitment | 0.815 | higher -> correct |
| answer_zero_crossings | thrashing | 0.786 | higher -> correct |
| answer_rebound_from_trough | shape | 0.776 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | 0.774 | higher -> correct |
| answer_end | landing | 0.713 | higher -> correct |
| answer_n_chunks | length | 0.700 | higher -> correct |
| answer_end_minus_reasoning_end | landing | 0.685 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_fall_from_peak | derailment | 1.589 | higher -> correct |
| answer_monotonicity | commitment | 1.502 | higher -> correct |
| answer_early_mean | shape | -0.999 | higher -> wrong |
| answer_time_positive | shape | -0.909 | higher -> wrong |
| answer_mid_mean | shape | -0.767 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | -0.693 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | 0.685 | higher -> correct |
| answer_direction_changes | thrashing | -0.676 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | -0.670 | higher -> wrong |
| answer_end | landing | -0.669 | higher -> wrong |
| answer_late_minus_early | transition | 0.661 | higher -> correct |
| reasoning_total_variation | thrashing | -0.660 | higher -> wrong |

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
| premature_exit | inconclusive | 153 | 0.479 [0.387, 0.576] | 0.601 | 17 | - | - | - | - |
| rambling_overlong | inconclusive | 153 | 0.521 [0.424, 0.613] | 0.601 | 23 | - | - | - | - |
| thrashing | inconclusive | 153 | 0.479 [0.390, 0.573] | 0.601 | 5 | - | - | - | - |
| no_commitment | inconclusive | 153 | 0.415 [0.329, 0.515] | 0.601 | 11 | - | - | - | - |
| derailment_late | inconclusive | 153 | 0.526 [0.429, 0.626] | 0.601 | 22 | - | - | - | - |
| answer_drift | inconclusive | 137 | 0.498 [0.406, 0.592] | 0.555 | 17 | - | - | - | - |
| answer_meandering | confirmed | 137 | 0.866 [0.809, 0.920] | 0.555 | 62 | 0.903 | 0.737 | 0.812 | 1.628 |
| answer_volatility | confirmed | 137 | 0.864 [0.799, 0.925] | 0.555 | 63 | 0.905 | 0.750 | 0.820 | 1.631 |
| answer_uncommitted | confirmed | 137 | 0.812 [0.739, 0.887] | 0.555 | 57 | 0.895 | 0.671 | 0.767 | 1.613 |
| answer_overrange | confirmed | 137 | 0.795 [0.712, 0.873] | 0.555 | 48 | 0.875 | 0.553 | 0.677 | 1.577 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_meandering, answer_volatility, answer_uncommitted, answer_overrange.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.866**
- `trajectory_full` ROC-AUC: **0.834**
- Above-chance discrimination preserved by the mode stack: **109.6%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 4 | 58 / 92 | 68 | 0.630 | 0.853 |
| any | 11 | 76 / 92 | 105 | 0.826 | 0.724 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
