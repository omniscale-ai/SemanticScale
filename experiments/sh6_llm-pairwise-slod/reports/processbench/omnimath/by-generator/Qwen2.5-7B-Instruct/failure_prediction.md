# SH6 processbench/omnimath/by-generator/Qwen2.5-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 50
- Positive class (`final_answer_correct=true`): 27
- Negative class (`final_answer_correct=false`): 23
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 50 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.553 +/- 0.274 | 0.667 +/- 0.151 | 0.567 +/- 0.165 | 0.580 +/- 0.147 | 0.638 +/- 0.106 |
| trajectory_shape (logreg) | 47 | 0.478 +/- 0.171 | 0.652 +/- 0.136 | 0.537 +/- 0.112 | 0.540 +/- 0.102 | 0.597 +/- 0.085 |
| trajectory_full (logreg) | 47 | 0.478 +/- 0.171 | 0.652 +/- 0.136 | 0.537 +/- 0.112 | 0.540 +/- 0.102 | 0.597 +/- 0.085 |
| reasoning_traj (MiniRocket) | 20 | 0.488 +/- 0.069 | 0.606 +/- 0.066 | 0.523 +/- 0.120 | 0.540 +/- 0.102 | 0.611 +/- 0.073 |
| trajectory_full (lightgbm) | 47 | 0.562 +/- 0.129 | 0.674 +/- 0.121 | 0.500 +/- 0.053 | 0.500 +/- 0.063 | 0.520 +/- 0.093 |
| mode_stack (logreg) | 6 | 0.577 +/- 0.124 | 0.692 +/- 0.091 | 0.520 +/- 0.040 | 0.520 +/- 0.040 | 0.504 +/- 0.111 |
| mode_stack (lightgbm) | 6 | 0.521 +/- 0.182 | 0.666 +/- 0.124 | 0.517 +/- 0.177 | 0.520 +/- 0.172 | 0.510 +/- 0.256 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t17 | shape | 0.731 | higher -> wrong |
| reasoning_traj_t11 | shape | 0.724 | higher -> wrong |
| reasoning_early_mean | shape | 0.715 | higher -> wrong |
| reasoning_time_positive | shape | 0.703 | higher -> correct |
| reasoning_time_negative | shape | 0.703 | higher -> correct |
| reasoning_late_minus_early | transition | 0.690 | higher -> wrong |
| reasoning_traj_t07 | shape | 0.689 | higher -> correct |
| reasoning_late_mean | landing | 0.683 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.676 | higher -> wrong |
| reasoning_traj_t08 | shape | 0.657 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.654 | higher -> wrong |
| reasoning_std | shape | 0.651 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_monotonicity | commitment | 1.046 | higher -> correct |
| reasoning_traj_t12 | shape | 0.829 | higher -> correct |
| reasoning_min | shape | -0.818 | higher -> wrong |
| reasoning_total_variation | thrashing | -0.789 | higher -> wrong |
| reasoning_n_chunks | length | -0.628 | higher -> wrong |
| reasoning_max_rise | shape | 0.580 | higher -> correct |
| reasoning_time_negative | shape | 0.496 | higher -> correct |
| reasoning_time_positive | shape | -0.496 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.482 | higher -> wrong |
| reasoning_traj_t05 | shape | -0.478 | higher -> wrong |
| reasoning_traj_t16 | shape | -0.472 | higher -> wrong |
| reasoning_max_drop | derailment | 0.464 | higher -> correct |

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
| premature_exit | inconclusive | 50 | 0.399 [0.238, 0.560] | 0.460 | 3 | - | - | - | - |
| rambling_overlong | inconclusive | 50 | 0.601 [0.440, 0.762] | 0.460 | 6 | - | - | - | - |
| thrashing | inconclusive | 50 | 0.618 [0.463, 0.768] | 0.460 | 3 | - | - | - | - |
| no_commitment | inconclusive | 50 | 0.575 [0.404, 0.720] | 0.460 | 7 | - | - | - | - |
| derailment_late | inconclusive | 50 | 0.571 [0.397, 0.735] | 0.460 | 8 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.577**
- `trajectory_full` ROC-AUC: **0.478**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 23 | 0 | - | - |
| any | 11 | 11 / 23 | 19 | 0.478 | 0.579 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
