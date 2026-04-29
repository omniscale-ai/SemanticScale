# SH6 frontierscience/R1-Distill-32B-cloudjudge_reasoning-auto — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 147
- Positive class (`is_correct=true`): 27
- Negative class (`is_correct=false`): 120

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.663 +/- 0.082 | 0.396 +/- 0.140 | 0.569 +/- 0.082 | 0.564 +/- 0.047 | 0.314 +/- 0.105 |
| trajectory_shape (logreg) | 96 | 0.523 +/- 0.072 | 0.260 +/- 0.056 | 0.534 +/- 0.099 | 0.666 +/- 0.048 | 0.257 +/- 0.135 |
| trajectory_full (logreg) | 99 | 0.525 +/- 0.087 | 0.260 +/- 0.075 | 0.535 +/- 0.116 | 0.694 +/- 0.050 | 0.242 +/- 0.173 |
| reasoning_traj (MiniRocket) | 20 | 0.572 +/- 0.092 | 0.277 +/- 0.075 | 0.529 +/- 0.103 | 0.673 +/- 0.049 | 0.215 +/- 0.151 |
| trajectory_full (lightgbm) | 99 | 0.521 +/- 0.133 | 0.253 +/- 0.069 | 0.463 +/- 0.031 | 0.735 +/- 0.039 | 0.036 +/- 0.073 |
| mode_stack (logreg) | 13 | 0.655 +/- 0.039 | 0.405 +/- 0.115 | 0.561 +/- 0.045 | 0.633 +/- 0.078 | 0.310 +/- 0.046 |
| mode_stack (lightgbm) | 13 | 0.662 +/- 0.102 | 0.387 +/- 0.123 | 0.581 +/- 0.046 | 0.789 +/- 0.028 | 0.308 +/- 0.095 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t03 | shape | 0.708 | higher -> correct |
| answer_rebound_from_trough | shape | 0.701 | higher -> correct |
| answer_end | landing | 0.692 | higher -> correct |
| answer_end_minus_start | landing | 0.689 | higher -> correct |
| reasoning_n_chunks | length | 0.680 | higher -> correct |
| reasoning_traj_t08 | shape | 0.668 | higher -> wrong |
| total_n_chunks | length | 0.668 | higher -> correct |
| answer_trough_pos | timing | 0.657 | higher -> correct |
| answer_n_chunks | length | 0.652 | higher -> correct |
| answer_max_rise | shape | 0.650 | higher -> correct |
| answer_traj_t09 | shape | 0.649 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.647 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_monotonicity | commitment | 1.368 | higher -> correct |
| answer_monotonicity | commitment | 0.968 | higher -> correct |
| answer_traj_t11 | shape | 0.937 | higher -> correct |
| reasoning_traj_t14 | shape | -0.726 | higher -> wrong |
| answer_total_variation | thrashing | 0.706 | higher -> correct |
| reasoning_traj_t13 | shape | 0.691 | higher -> correct |
| reasoning_traj_t03 | shape | 0.688 | higher -> correct |
| reasoning_traj_t15 | shape | 0.647 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.627 | higher -> correct |
| answer_std | shape | -0.619 | higher -> wrong |
| answer_max_rise | shape | -0.618 | higher -> wrong |
| reasoning_zero_crossings | thrashing | -0.609 | higher -> wrong |

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
| premature_exit | confirmed | 147 | 0.661 [0.559, 0.766] | 0.816 | 51 | 0.941 | 0.400 | 0.561 | 1.153 |
| rambling_overlong | inverted | 147 | 0.339 [0.234, 0.441] | 0.816 | 9 | - | - | - | - |
| thrashing | inconclusive | 147 | 0.435 [0.332, 0.534] | 0.816 | 25 | - | - | - | - |
| no_commitment | confirmed | 147 | 0.641 [0.533, 0.744] | 0.816 | 46 | 0.935 | 0.358 | 0.518 | 1.145 |
| derailment_late | inconclusive | 147 | 0.397 [0.269, 0.511] | 0.816 | 5 | - | - | - | - |
| answer_drift | inconclusive | 144 | 0.414 [0.280, 0.544] | 0.812 | 13 | - | - | - | - |
| answer_meandering | confirmed | 144 | 0.644 [0.525, 0.758] | 0.812 | 37 | 0.919 | 0.291 | 0.442 | 1.131 |
| answer_volatility | confirmed | 144 | 0.655 [0.533, 0.765] | 0.812 | 51 | 0.941 | 0.410 | 0.571 | 1.158 |
| answer_uncommitted | confirmed | 144 | 0.642 [0.524, 0.755] | 0.812 | 40 | 0.925 | 0.316 | 0.471 | 1.138 |
| answer_overrange | inconclusive | 144 | 0.599 [0.474, 0.721] | 0.812 | 30 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: premature_exit, no_commitment, answer_meandering, answer_volatility, answer_uncommitted.
- **Hypothesis falsified (inverted)**: rambling_overlong. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: thrashing, derailment_late, answer_drift, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.655**
- `trajectory_full` ROC-AUC: **0.525**
- Above-chance discrimination preserved by the mode stack: **613.2%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 5 | 88 / 120 | 98 | 0.733 | 0.898 |
| any | 11 | 105 / 120 | 124 | 0.875 | 0.847 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
