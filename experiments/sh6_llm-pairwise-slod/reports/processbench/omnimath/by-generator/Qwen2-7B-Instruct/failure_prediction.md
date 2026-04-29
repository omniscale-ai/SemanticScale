# SH6 processbench/omnimath/by-generator/Qwen2-7B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 46
- Positive class (`final_answer_correct=true`): 27
- Negative class (`final_answer_correct=false`): 19
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 46 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.685 +/- 0.173 | 0.766 +/- 0.138 | 0.647 +/- 0.124 | 0.631 +/- 0.109 | 0.651 +/- 0.108 |
| trajectory_shape (logreg) | 47 | 0.593 +/- 0.109 | 0.776 +/- 0.076 | 0.593 +/- 0.166 | 0.587 +/- 0.147 | 0.627 +/- 0.131 |
| trajectory_full (logreg) | 47 | 0.593 +/- 0.109 | 0.776 +/- 0.076 | 0.593 +/- 0.166 | 0.587 +/- 0.147 | 0.627 +/- 0.131 |
| reasoning_traj (MiniRocket) | 20 | 0.586 +/- 0.144 | 0.719 +/- 0.138 | 0.547 +/- 0.115 | 0.584 +/- 0.091 | 0.629 +/- 0.160 |
| trajectory_full (lightgbm) | 47 | 0.688 +/- 0.135 | 0.807 +/- 0.127 | 0.665 +/- 0.126 | 0.673 +/- 0.100 | 0.741 +/- 0.061 |
| mode_stack (logreg) | 6 | 0.646 +/- 0.130 | 0.815 +/- 0.063 | 0.588 +/- 0.093 | 0.587 +/- 0.083 | 0.628 +/- 0.063 |
| mode_stack (lightgbm) | 6 | 0.603 +/- 0.151 | 0.760 +/- 0.129 | 0.563 +/- 0.110 | 0.562 +/- 0.107 | 0.609 +/- 0.109 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_traj_t08 | shape | 0.792 | higher -> correct |
| reasoning_traj_t04 | shape | 0.766 | higher -> wrong |
| reasoning_traj_t07 | shape | 0.757 | higher -> correct |
| reasoning_traj_t05 | shape | 0.754 | higher -> wrong |
| reasoning_time_negative | shape | 0.733 | higher -> wrong |
| reasoning_start | shape | 0.730 | higher -> wrong |
| reasoning_late_minus_early | transition | 0.729 | higher -> wrong |
| reasoning_positive_mass | shape | 0.727 | higher -> correct |
| reasoning_traj_t02 | shape | 0.726 | higher -> wrong |
| reasoning_early_mean | shape | 0.726 | higher -> wrong |
| reasoning_traj_t03 | shape | 0.725 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.708 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_min | shape | 0.919 | higher -> correct |
| reasoning_max_drop_pos | derailment | -0.793 | higher -> wrong |
| reasoning_traj_t08 | shape | -0.712 | higher -> wrong |
| reasoning_max_rise_pos | timing | 0.642 | higher -> correct |
| reasoning_traj_t07 | shape | -0.635 | higher -> wrong |
| reasoning_monotonicity | commitment | -0.611 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.602 | higher -> wrong |
| reasoning_n_chunks | length | -0.573 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | -0.539 | higher -> wrong |
| reasoning_max_drop | derailment | 0.530 | higher -> correct |
| reasoning_positive_mass | shape | -0.447 | higher -> wrong |
| reasoning_traj_t05 | shape | 0.439 | higher -> correct |

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
| premature_exit | inverted | 46 | 0.328 [0.174, 0.499] | 0.413 | 3 | - | - | - | - |
| rambling_overlong | confirmed | 46 | 0.672 [0.501, 0.826] | 0.413 | 6 | 0.667 | 0.211 | 0.320 | 1.614 |
| thrashing | confirmed | 46 | 0.666 [0.508, 0.820] | 0.413 | 6 | 0.500 | 0.158 | 0.240 | 1.211 |
| no_commitment | inconclusive | 46 | 0.456 [0.286, 0.640] | 0.413 | 0 | - | - | - | - |
| derailment_late | inconclusive | 46 | 0.646 [0.467, 0.808] | 0.413 | 6 | - | - | - | - |
| answer_drift | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_meandering | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_volatility | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_uncommitted | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| answer_overrange | insufficient_data | 0 | - | - | 0 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: rambling_overlong, thrashing.
- **Hypothesis falsified (inverted)**: premature_exit. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: no_commitment, derailment_late. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.646**
- `trajectory_full` ROC-AUC: **0.593**
- Above-chance discrimination preserved by the mode stack: **156.5%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 2 | 5 / 19 | 10 | 0.263 | 0.500 |
| any | 11 | 8 / 19 | 18 | 0.421 | 0.444 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
