# SH6 processbench/olympiadbench/by-generator/Qwen2.5-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 46
- Positive class (`final_answer_correct=true`): 19
- Negative class (`final_answer_correct=false`): 27
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 46 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.604 +/- 0.170 | 0.576 +/- 0.191 | 0.527 +/- 0.175 | 0.522 +/- 0.178 | 0.502 +/- 0.182 |
| trajectory_shape (logreg) | 47 | 0.657 +/- 0.209 | 0.736 +/- 0.139 | 0.560 +/- 0.142 | 0.540 +/- 0.183 | 0.543 +/- 0.121 |
| trajectory_full (logreg) | 47 | 0.657 +/- 0.209 | 0.736 +/- 0.139 | 0.560 +/- 0.142 | 0.540 +/- 0.183 | 0.543 +/- 0.121 |
| reasoning_traj (MiniRocket) | 20 | 0.487 +/- 0.141 | 0.527 +/- 0.076 | 0.505 +/- 0.133 | 0.504 +/- 0.124 | 0.435 +/- 0.161 |
| trajectory_full (lightgbm) | 47 | 0.539 +/- 0.191 | 0.597 +/- 0.183 | 0.548 +/- 0.120 | 0.540 +/- 0.137 | 0.517 +/- 0.094 |
| mode_stack (logreg) | 6 | 0.503 +/- 0.211 | 0.545 +/- 0.205 | 0.510 +/- 0.178 | 0.518 +/- 0.187 | 0.451 +/- 0.187 |
| mode_stack (lightgbm) | 6 | 0.648 +/- 0.184 | 0.682 +/- 0.156 | 0.552 +/- 0.196 | 0.536 +/- 0.208 | 0.504 +/- 0.199 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t14 | shape | 0.745 | higher -> correct |
| reasoning_peak_pos | timing | 0.731 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.731 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.713 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.708 | higher -> correct |
| reasoning_traj_t01 | shape | 0.694 | higher -> correct |
| reasoning_traj_t17 | shape | 0.690 | higher -> wrong |
| reasoning_traj_t13 | shape | 0.687 | higher -> correct |
| reasoning_min | shape | 0.680 | higher -> correct |
| reasoning_mid_mean | shape | 0.665 | higher -> correct |
| reasoning_start | shape | 0.662 | higher -> correct |
| reasoning_traj_t08 | shape | 0.656 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_traj_t13 | shape | -0.804 | higher -> wrong |
| reasoning_peak_pos | timing | 0.778 | higher -> correct |
| reasoning_trough_pos | timing | 0.642 | higher -> correct |
| reasoning_positive_mass | shape | -0.641 | higher -> wrong |
| reasoning_traj_t11 | shape | 0.611 | higher -> correct |
| reasoning_max_drop | derailment | 0.522 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.496 | higher -> correct |
| reasoning_traj_t08 | shape | -0.489 | higher -> wrong |
| reasoning_traj_t18 | shape | 0.489 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.457 | higher -> correct |
| reasoning_start | shape | 0.425 | higher -> correct |
| reasoning_monotonicity | commitment | 0.411 | higher -> correct |

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
| premature_exit | inconclusive | 46 | 0.367 [0.221, 0.549] | 0.587 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 46 | 0.633 [0.451, 0.779] | 0.587 | 11 | - | - | - | - |
| thrashing | inconclusive | 46 | 0.633 [0.467, 0.778] | 0.587 | 9 | - | - | - | - |
| no_commitment | inconclusive | 46 | 0.610 [0.446, 0.779] | 0.587 | 12 | - | - | - | - |
| derailment_late | inconclusive | 46 | 0.491 [0.310, 0.658] | 0.587 | 10 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.503**
- `trajectory_full` ROC-AUC: **0.657**
- Above-chance discrimination preserved by the mode stack: **1.8%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 27 | 0 | - | - |
| any | 11 | 20 / 27 | 25 | 0.741 | 0.800 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
