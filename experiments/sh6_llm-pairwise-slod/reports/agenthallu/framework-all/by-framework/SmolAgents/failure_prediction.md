# SH6 agenthallu/framework-all/by-framework/SmolAgents — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 91
- Positive class (`final_answer_correct=true`): 34
- Negative class (`final_answer_correct=false`): 57
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 91 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.581 +/- 0.060 | 0.499 +/- 0.056 | 0.596 +/- 0.037 | 0.638 +/- 0.062 | 0.475 +/- 0.032 |
| trajectory_shape (logreg) | 60 | 0.424 +/- 0.110 | 0.389 +/- 0.058 | 0.480 +/- 0.091 | 0.506 +/- 0.090 | 0.360 +/- 0.129 |
| trajectory_full (logreg) | 63 | 0.451 +/- 0.122 | 0.420 +/- 0.091 | 0.474 +/- 0.085 | 0.506 +/- 0.075 | 0.341 +/- 0.128 |
| trajectory_full (lightgbm) | 63 | 0.633 +/- 0.158 | 0.598 +/- 0.173 | 0.637 +/- 0.089 | 0.639 +/- 0.066 | 0.550 +/- 0.128 |
| mode_stack (logreg) | 10 | 0.589 +/- 0.087 | 0.504 +/- 0.117 | 0.555 +/- 0.100 | 0.573 +/- 0.089 | 0.443 +/- 0.154 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_max_drop_pos | derailment | 0.621 | higher -> correct |
| reasoning_late_mean | landing | 0.613 | higher -> correct |
| reasoning_std | shape | 0.607 | higher -> wrong |
| reasoning_min | shape | 0.607 | higher -> wrong |
| total_n_chunks | length | 0.604 | higher -> correct |
| reasoning_late_minus_early | transition | 0.600 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.599 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.595 | higher -> correct |
| answer_end_minus_reasoning_end | landing | 0.589 | higher -> wrong |
| reasoning_mid_mean | shape | 0.585 | higher -> correct |
| answer_start_minus_reasoning_end | landing | 0.581 | higher -> correct |
| answer_mid_mean | shape | 0.579 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| answer_max_rise_pos | timing | 0.901 | higher -> correct |
| reasoning_peak_pos | timing | -0.774 | higher -> wrong |
| answer_end_minus_reasoning_end | landing | 0.717 | higher -> correct |
| reasoning_max | shape | 0.641 | higher -> correct |
| answer_n_chunks | length | 0.618 | higher -> correct |
| answer_late_mean | landing | -0.541 | higher -> wrong |
| reasoning_start | shape | -0.541 | higher -> wrong |
| reasoning_negative_mass | shape | -0.533 | higher -> wrong |
| answer_early_mean | shape | -0.529 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.526 | higher -> correct |
| answer_max_drop | derailment | 0.480 | higher -> correct |
| reasoning_end_minus_start | landing | 0.466 | higher -> correct |

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
| premature_exit | inconclusive | 91 | 0.567 [0.448, 0.681] | 0.626 | 16 | - | - | - | - |
| rambling_overlong | inconclusive | 91 | 0.433 [0.319, 0.552] | 0.626 | 4 | - | - | - | - |
| thrashing | inconclusive | 91 | 0.444 [0.331, 0.555] | 0.626 | 3 | - | - | - | - |
| no_commitment | inconclusive | 91 | 0.597 [0.479, 0.719] | 0.626 | 16 | - | - | - | - |
| derailment_late | inconclusive | 91 | 0.431 [0.308, 0.547] | 0.626 | 7 | - | - | - | - |
| answer_drift | inconclusive | 78 | 0.385 [0.266, 0.521] | 0.603 | 5 | - | - | - | - |
| answer_meandering | inconclusive | 78 | 0.434 [0.346, 0.517] | 0.603 | 7 | - | - | - | - |
| answer_volatility | inconclusive | 78 | 0.427 [0.339, 0.517] | 0.603 | 6 | - | - | - | - |
| answer_uncommitted | inconclusive | 78 | 0.507 [0.416, 0.605] | 0.603 | 0 | - | - | - | - |
| answer_overrange | inconclusive | 78 | 0.461 [0.324, 0.585] | 0.603 | 8 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.589**
- `trajectory_full` ROC-AUC: **0.451**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 57 | 0 | - | - |
| any | 11 | 29 / 57 | 44 | 0.509 | 0.659 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
