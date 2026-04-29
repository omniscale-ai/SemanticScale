# SH6 frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s2/by-origin/olympiad — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 96
- Positive class (`is_correct=true`): 58
- Negative class (`is_correct=false`): 38

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.604 +/- 0.105 | 0.697 +/- 0.099 | 0.548 +/- 0.089 | 0.572 +/- 0.095 | 0.652 +/- 0.101 |
| trajectory_shape (logreg) | 96 | 0.630 +/- 0.101 | 0.730 +/- 0.095 | 0.635 +/- 0.038 | 0.646 +/- 0.047 | 0.700 +/- 0.053 |
| trajectory_full (logreg) | 99 | 0.631 +/- 0.090 | 0.719 +/- 0.085 | 0.632 +/- 0.056 | 0.646 +/- 0.070 | 0.703 +/- 0.076 |
| reasoning_traj (MiniRocket) | 20 | 0.453 +/- 0.066 | 0.646 +/- 0.065 | 0.462 +/- 0.068 | 0.469 +/- 0.062 | 0.536 +/- 0.081 |
| trajectory_full (lightgbm) | 99 | 0.587 +/- 0.108 | 0.713 +/- 0.076 | 0.559 +/- 0.080 | 0.594 +/- 0.061 | 0.684 +/- 0.036 |
| mode_stack (logreg) | 13 | 0.611 +/- 0.056 | 0.711 +/- 0.076 | 0.554 +/- 0.079 | 0.552 +/- 0.105 | 0.576 +/- 0.170 |
| mode_stack (lightgbm) | 13 | 0.700 +/- 0.099 | 0.776 +/- 0.074 | 0.648 +/- 0.065 | 0.666 +/- 0.074 | 0.714 +/- 0.093 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_start | shape | 0.662 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.652 | higher -> wrong |
| reasoning_n_chunks | length | 0.646 | higher -> correct |
| reasoning_traj_t10 | shape | 0.643 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.643 | higher -> wrong |
| reasoning_min | shape | 0.643 | higher -> correct |
| reasoning_range | commitment | 0.639 | higher -> correct |
| reasoning_std | shape | 0.637 | higher -> correct |
| total_n_chunks | length | 0.637 | higher -> correct |
| reasoning_start | shape | 0.632 | higher -> wrong |
| reasoning_max_rise | shape | 0.631 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.629 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_rebound_from_trough | shape | 0.803 | higher -> correct |
| reasoning_traj_t16 | shape | -0.764 | higher -> wrong |
| reasoning_time_negative | shape | -0.714 | higher -> wrong |
| reasoning_time_positive | shape | 0.714 | higher -> correct |
| answer_trough_pos | timing | 0.680 | higher -> correct |
| reasoning_positive_mass | shape | -0.675 | higher -> wrong |
| answer_peak_pos | timing | -0.662 | higher -> wrong |
| reasoning_max_rise | shape | 0.644 | higher -> correct |
| answer_traj_t05 | shape | 0.628 | higher -> correct |
| answer_monotonicity | commitment | -0.598 | higher -> wrong |
| reasoning_trough_pos | timing | 0.594 | higher -> correct |
| answer_end_minus_reasoning_end | landing | -0.592 | higher -> wrong |

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
| premature_exit | inverted | 96 | 0.372 [0.259, 0.488] | 0.396 | 9 | - | - | - | - |
| rambling_overlong | confirmed | 96 | 0.628 [0.512, 0.741] | 0.396 | 16 | 0.625 | 0.263 | 0.370 | 1.579 |
| thrashing | inconclusive | 96 | 0.496 [0.374, 0.617] | 0.396 | 6 | - | - | - | - |
| no_commitment | inconclusive | 96 | 0.510 [0.386, 0.644] | 0.396 | 13 | - | - | - | - |
| derailment_late | inconclusive | 96 | 0.464 [0.348, 0.584] | 0.396 | 8 | - | - | - | - |
| answer_drift | inconclusive | 87 | 0.523 [0.398, 0.651] | 0.333 | 9 | - | - | - | - |
| answer_meandering | inconclusive | 87 | 0.490 [0.399, 0.587] | 0.333 | 8 | - | - | - | - |
| answer_volatility | inconclusive | 87 | 0.464 [0.325, 0.609] | 0.333 | 10 | - | - | - | - |
| answer_uncommitted | inconclusive | 87 | 0.453 [0.362, 0.551] | 0.333 | 8 | - | - | - | - |
| answer_overrange | inconclusive | 87 | 0.588 [0.450, 0.712] | 0.333 | 10 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.611**
- `trajectory_full` ROC-AUC: **0.631**
- Above-chance discrimination preserved by the mode stack: **84.4%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 10 / 38 | 16 | 0.263 | 0.625 |
| any | 11 | 26 / 38 | 62 | 0.684 | 0.419 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
