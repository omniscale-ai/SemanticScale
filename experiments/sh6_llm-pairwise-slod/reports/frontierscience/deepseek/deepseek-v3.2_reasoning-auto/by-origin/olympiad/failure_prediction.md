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
| trajectory_shape (logreg) | 96 | 0.511 +/- 0.151 | 0.688 +/- 0.077 | 0.483 +/- 0.125 | 0.505 +/- 0.108 | 0.592 +/- 0.085 |
| trajectory_full (logreg) | 99 | 0.610 +/- 0.119 | 0.749 +/- 0.083 | 0.570 +/- 0.084 | 0.600 +/- 0.098 | 0.672 +/- 0.108 |
| reasoning_traj (MiniRocket) | 20 | 0.627 +/- 0.077 | 0.759 +/- 0.068 | 0.588 +/- 0.074 | 0.642 +/- 0.039 | 0.734 +/- 0.020 |
| trajectory_full (lightgbm) | 99 | 0.699 +/- 0.069 | 0.770 +/- 0.047 | 0.678 +/- 0.111 | 0.726 +/- 0.102 | 0.797 +/- 0.083 |
| mode_stack (logreg) | 13 | 0.733 +/- 0.131 | 0.822 +/- 0.088 | 0.717 +/- 0.109 | 0.747 +/- 0.107 | 0.802 +/- 0.094 |
| mode_stack (lightgbm) | 13 | 0.740 +/- 0.080 | 0.818 +/- 0.053 | 0.648 +/- 0.148 | 0.695 +/- 0.117 | 0.773 +/- 0.083 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_positive_mass | shape | 0.733 | higher -> wrong |
| answer_fall_from_peak | derailment | 0.697 | higher -> wrong |
| reasoning_traj_t07 | shape | 0.666 | higher -> wrong |
| answer_traj_t01 | shape | 0.663 | higher -> wrong |
| answer_total_variation | thrashing | 0.660 | higher -> wrong |
| reasoning_max_drop | derailment | 0.659 | higher -> wrong |
| answer_max_drop | derailment | 0.650 | higher -> wrong |
| total_n_chunks | length | 0.649 | higher -> correct |
| answer_traj_t04 | shape | 0.648 | higher -> wrong |
| reasoning_traj_t13 | shape | 0.647 | higher -> correct |
| reasoning_n_chunks | length | 0.646 | higher -> correct |
| answer_traj_t02 | shape | 0.645 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_n_chunks | length | 1.573 | higher -> correct |
| answer_monotonicity | commitment | 0.983 | higher -> correct |
| answer_range_minus_reasoning_range | commitment | -0.968 | higher -> wrong |
| answer_time_positive | shape | -0.914 | higher -> wrong |
| answer_zero_crossings | thrashing | -0.737 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.658 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | 0.648 | higher -> correct |
| reasoning_max_rise_pos | timing | -0.641 | higher -> wrong |
| answer_max | shape | 0.580 | higher -> correct |
| answer_peak_pos | timing | -0.569 | higher -> wrong |
| reasoning_n_chunks | length | -0.528 | higher -> wrong |
| answer_direction_changes | thrashing | -0.514 | higher -> wrong |

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

- `mode_stack` ROC-AUC: **0.733**
- `trajectory_full` ROC-AUC: **0.610**
- Above-chance discrimination preserved by the mode stack: **212.5%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 12 / 34 | 18 | 0.353 | 0.667 |
| any | 11 | 22 / 34 | 51 | 0.647 | 0.431 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
