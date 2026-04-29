# SH6 processbench/gsm8k/by-generator/Meta-Llama-3-8B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 24
- Positive class (`final_answer_correct=true`): 13
- Negative class (`final_answer_correct=false`): 11
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 24 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.683 +/- 0.200 | 0.743 +/- 0.188 | 0.650 +/- 0.249 | 0.640 +/- 0.233 | 0.654 +/- 0.217 |
| trajectory_shape (logreg) | 47 | 0.500 +/- 0.211 | 0.691 +/- 0.155 | 0.383 +/- 0.145 | 0.410 +/- 0.111 | 0.409 +/- 0.214 |
| trajectory_full (logreg) | 47 | 0.500 +/- 0.211 | 0.691 +/- 0.155 | 0.383 +/- 0.145 | 0.410 +/- 0.111 | 0.409 +/- 0.214 |
| reasoning_traj (MiniRocket) | 20 | 0.650 +/- 0.271 | 0.808 +/- 0.153 | 0.633 +/- 0.245 | 0.620 +/- 0.240 | 0.574 +/- 0.337 |
| trajectory_full (lightgbm) | 47 | 0.500 +/- 0.000 | 0.590 +/- 0.111 | 0.450 +/- 0.100 | 0.370 +/- 0.060 | 0.194 +/- 0.244 |
| mode_stack (logreg) | 6 | 0.467 +/- 0.194 | 0.653 +/- 0.182 | 0.567 +/- 0.133 | 0.550 +/- 0.134 | 0.553 +/- 0.157 |
| mode_stack (lightgbm) | 6 | 0.450 +/- 0.100 | 0.540 +/- 0.080 | 0.450 +/- 0.100 | 0.370 +/- 0.060 | 0.114 +/- 0.229 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_positive_mass | shape | 0.817 | higher -> wrong |
| reasoning_traj_t12 | shape | 0.783 | higher -> wrong |
| reasoning_traj_t13 | shape | 0.750 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.717 | higher -> correct |
| reasoning_max_drop | derailment | 0.700 | higher -> correct |
| reasoning_curvature_abs_mean | thrashing | 0.700 | higher -> correct |
| reasoning_n_chunks | length | 0.683 | higher -> correct |
| reasoning_std | shape | 0.667 | higher -> wrong |
| reasoning_max_rise | shape | 0.667 | higher -> correct |
| reasoning_total_variation | thrashing | 0.650 | higher -> correct |
| reasoning_min | shape | 0.633 | higher -> correct |
| reasoning_rebound_from_trough | shape | 0.633 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_monotonicity | commitment | -0.823 | higher -> wrong |
| reasoning_negative_mass | shape | 0.711 | higher -> correct |
| reasoning_zero_crossings | thrashing | 0.553 | higher -> correct |
| reasoning_max_rise_pos | timing | 0.468 | higher -> correct |
| reasoning_min | shape | 0.417 | higher -> correct |
| reasoning_std | shape | 0.414 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.386 | higher -> correct |
| reasoning_range | commitment | -0.381 | higher -> wrong |
| reasoning_max_rise | shape | 0.356 | higher -> correct |
| reasoning_time_negative | shape | 0.348 | higher -> correct |
| reasoning_time_positive | shape | -0.348 | higher -> wrong |
| reasoning_rebound_from_trough | shape | -0.286 | higher -> wrong |

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
| premature_exit | inconclusive | 24 | 0.587 [0.339, 0.818] | 0.458 | 2 | - | - | - | - |
| rambling_overlong | inconclusive | 24 | 0.413 [0.182, 0.661] | 0.458 | 2 | - | - | - | - |
| thrashing | inconclusive | 24 | 0.458 [0.231, 0.689] | 0.458 | 3 | - | - | - | - |
| no_commitment | inconclusive | 24 | 0.294 [0.086, 0.519] | 0.458 | 3 | - | - | - | - |
| derailment_late | inconclusive | 24 | 0.545 [0.294, 0.781] | 0.458 | 6 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.467**
- `trajectory_full` ROC-AUC: **0.500**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 11 | 0 | - | - |
| any | 11 | 7 / 11 | 14 | 0.636 | 0.500 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
