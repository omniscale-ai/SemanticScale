# SH6 agenthallu/framework-all/by-framework/Octotools — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 47
- Positive class (`final_answer_correct=true`): 17
- Negative class (`final_answer_correct=false`): 30
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 47 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.303 +/- 0.215 | 0.367 +/- 0.119 | 0.292 +/- 0.144 | 0.342 +/- 0.154 | 0.114 +/- 0.140 |
| trajectory_shape (logreg) | 60 | 0.550 +/- 0.198 | 0.513 +/- 0.207 | 0.467 +/- 0.180 | 0.502 +/- 0.178 | 0.340 +/- 0.228 |
| trajectory_full (logreg) | 63 | 0.539 +/- 0.211 | 0.509 +/- 0.210 | 0.500 +/- 0.167 | 0.524 +/- 0.162 | 0.379 +/- 0.225 |
| trajectory_full (lightgbm) | 63 | 0.406 +/- 0.213 | 0.440 +/- 0.166 | 0.350 +/- 0.164 | 0.360 +/- 0.148 | 0.267 +/- 0.194 |
| mode_stack (logreg) | 10 | 0.592 +/- 0.134 | 0.552 +/- 0.171 | 0.633 +/- 0.103 | 0.611 +/- 0.125 | 0.572 +/- 0.116 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| answer_start | shape | 0.703 | higher -> wrong |
| answer_max_rise | shape | 0.697 | higher -> wrong |
| answer_mid_mean | shape | 0.694 | higher -> wrong |
| answer_start_minus_reasoning_end | landing | 0.692 | higher -> wrong |
| answer_late_mean | landing | 0.683 | higher -> wrong |
| answer_minus_reasoning_mean | answer_alignment | 0.681 | higher -> wrong |
| answer_max_rise_pos | timing | 0.665 | higher -> wrong |
| reasoning_direction_changes | thrashing | 0.658 | higher -> correct |
| answer_n_chunks | length | 0.653 | higher -> wrong |
| answer_end_minus_start | landing | 0.650 | higher -> wrong |
| total_n_chunks | length | 0.649 | higher -> wrong |
| reasoning_mid_mean | shape | 0.647 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_max_rise_pos | timing | 0.793 | higher -> correct |
| reasoning_monotonicity | commitment | -0.782 | higher -> wrong |
| answer_min | shape | 0.772 | higher -> correct |
| answer_fall_from_peak | derailment | -0.708 | higher -> wrong |
| answer_range | commitment | -0.684 | higher -> wrong |
| answer_monotonicity | commitment | -0.681 | higher -> wrong |
| answer_positive_mass | shape | 0.669 | higher -> correct |
| answer_max_rise_pos | timing | -0.656 | higher -> wrong |
| answer_direction_changes | thrashing | 0.578 | higher -> correct |
| answer_negative_mass | shape | 0.566 | higher -> correct |
| answer_early_mean | shape | -0.562 | higher -> wrong |
| reasoning_early_mean | shape | -0.524 | higher -> wrong |

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
| premature_exit | inconclusive | 47 | 0.552 [0.396, 0.719] | 0.638 | 0 | - | - | - | - |
| rambling_overlong | inconclusive | 47 | 0.448 [0.281, 0.604] | 0.638 | 3 | - | - | - | - |
| thrashing | inconclusive | 47 | 0.364 [0.219, 0.517] | 0.638 | 4 | - | - | - | - |
| no_commitment | inverted | 47 | 0.333 [0.186, 0.486] | 0.638 | 6 | - | - | - | - |
| derailment_late | inconclusive | 47 | 0.614 [0.443, 0.773] | 0.638 | 10 | - | - | - | - |
| answer_drift | inconclusive | 47 | 0.425 [0.262, 0.598] | 0.638 | 7 | - | - | - | - |
| answer_meandering | inconclusive | 47 | 0.421 [0.245, 0.606] | 0.638 | 4 | - | - | - | - |
| answer_volatility | inconclusive | 47 | 0.494 [0.315, 0.671] | 0.638 | 6 | - | - | - | - |
| answer_uncommitted | inconclusive | 47 | 0.447 [0.241, 0.639] | 0.638 | 3 | - | - | - | - |
| answer_overrange | inconclusive | 47 | 0.533 [0.359, 0.708] | 0.638 | 6 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Hypothesis falsified (inverted)**: no_commitment. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.592**
- `trajectory_full` ROC-AUC: **0.539**
- Above-chance discrimination preserved by the mode stack: **235.7%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 30 | 0 | - | - |
| any | 11 | 20 / 30 | 30 | 0.667 | 0.667 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
