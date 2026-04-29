# SH6 processbench/olympiadbench/by-generator/Llama-3.1-8B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 60
- Positive class (`final_answer_correct=true`): 34
- Negative class (`final_answer_correct=false`): 26
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 60 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.539 +/- 0.103 | 0.648 +/- 0.086 | 0.579 +/- 0.069 | 0.600 +/- 0.082 | 0.663 +/- 0.108 |
| trajectory_shape (logreg) | 29 | 0.574 +/- 0.107 | 0.692 +/- 0.104 | 0.543 +/- 0.144 | 0.550 +/- 0.145 | 0.563 +/- 0.177 |
| trajectory_full (logreg) | 29 | 0.574 +/- 0.107 | 0.692 +/- 0.104 | 0.543 +/- 0.144 | 0.550 +/- 0.145 | 0.563 +/- 0.177 |
| trajectory_full (lightgbm) | 29 | 0.573 +/- 0.099 | 0.686 +/- 0.070 | 0.534 +/- 0.089 | 0.550 +/- 0.085 | 0.624 +/- 0.090 |
| mode_stack (logreg) | 6 | 0.597 +/- 0.090 | 0.684 +/- 0.058 | 0.543 +/- 0.103 | 0.550 +/- 0.113 | 0.588 +/- 0.149 |
| mode_stack (lightgbm) | 6 | 0.558 +/- 0.174 | 0.704 +/- 0.119 | 0.559 +/- 0.117 | 0.550 +/- 0.125 | 0.555 +/- 0.160 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max_rise_pos | timing | 0.612 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.602 | higher -> correct |
| reasoning_peak_pos | timing | 0.599 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.596 | higher -> correct |
| reasoning_start | shape | 0.591 | higher -> correct |
| reasoning_mid_mean | shape | 0.590 | higher -> correct |
| reasoning_trough_pos | timing | 0.582 | higher -> correct |
| reasoning_monotonicity | commitment | 0.580 | higher -> correct |
| reasoning_positive_mass | shape | 0.580 | higher -> wrong |
| reasoning_max_drop | derailment | 0.573 | higher -> correct |
| reasoning_min | shape | 0.567 | higher -> wrong |
| reasoning_max | shape | 0.567 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_zero_crossings | thrashing | 0.656 | higher -> correct |
| reasoning_n_chunks | length | -0.622 | higher -> wrong |
| reasoning_end_minus_start | landing | 0.512 | higher -> correct |
| reasoning_max_drop | derailment | -0.481 | higher -> wrong |
| reasoning_min | shape | 0.459 | higher -> correct |
| reasoning_start | shape | -0.457 | higher -> wrong |
| reasoning_peak_pos | timing | -0.426 | higher -> wrong |
| reasoning_max | shape | 0.418 | higher -> correct |
| reasoning_trough_pos | timing | 0.408 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | -0.407 | higher -> wrong |
| reasoning_mid_mean | shape | 0.396 | higher -> correct |
| reasoning_monotonicity | commitment | -0.344 | higher -> wrong |

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
| premature_exit | inconclusive | 60 | 0.443 [0.290, 0.600] | 0.433 | 7 | - | - | - | - |
| rambling_overlong | inconclusive | 60 | 0.557 [0.400, 0.710] | 0.433 | 10 | - | - | - | - |
| thrashing | inconclusive | 60 | 0.564 [0.420, 0.713] | 0.433 | 9 | - | - | - | - |
| no_commitment | inconclusive | 60 | 0.455 [0.307, 0.598] | 0.433 | 5 | - | - | - | - |
| derailment_late | inconclusive | 60 | 0.561 [0.421, 0.708] | 0.433 | 7 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.597**
- `trajectory_full` ROC-AUC: **0.574**
- Above-chance discrimination preserved by the mode stack: **131.8%** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 26 | 0 | - | - |
| any | 11 | 15 / 26 | 29 | 0.577 | 0.517 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
