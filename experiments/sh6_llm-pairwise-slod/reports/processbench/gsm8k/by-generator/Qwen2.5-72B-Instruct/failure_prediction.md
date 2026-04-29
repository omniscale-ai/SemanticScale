# SH6 processbench/gsm8k/by-generator/Qwen2.5-72B-Instruct — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 23
- Positive class (`final_answer_correct=true`): 21
- Negative class (`final_answer_correct=false`): 2
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 23 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Status

- Reduced cross-validation folds from 5 to 2 due to class counts.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 1 | 0.743 +/- 0.107 | 0.964 +/- 0.017 | 0.577 +/- 0.123 | 0.644 +/- 0.189 | 0.740 +/- 0.169 |
| trajectory_shape (logreg) | 29 | 0.186 +/- 0.086 | 0.878 +/- 0.030 | 0.455 +/- 0.045 | 0.830 +/- 0.080 | 0.905 +/- 0.048 |
| trajectory_full (logreg) | 29 | 0.186 +/- 0.086 | 0.878 +/- 0.030 | 0.455 +/- 0.045 | 0.830 +/- 0.080 | 0.905 +/- 0.048 |
| trajectory_full (lightgbm) | 29 | 0.500 +/- 0.000 | 0.913 +/- 0.004 | 0.500 +/- 0.000 | 0.913 +/- 0.004 | 0.954 +/- 0.002 |
| mode_stack (logreg) | 5 | 0.482 +/- 0.118 | 0.941 +/- 0.016 | 0.532 +/- 0.168 | 0.561 +/- 0.106 | 0.686 +/- 0.114 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_mid_mean | shape | 1.000 | higher -> correct |
| reasoning_end_minus_start | landing | 0.950 | higher -> wrong |
| reasoning_trough_pos | timing | 0.900 | higher -> wrong |
| reasoning_early_mean | shape | 0.850 | higher -> wrong |
| reasoning_end | landing | 0.818 | higher -> wrong |
| reasoning_monotonicity | commitment | 0.814 | higher -> wrong |
| reasoning_fall_from_peak | derailment | 0.814 | higher -> wrong |
| reasoning_late_minus_early | transition | 0.809 | higher -> wrong |
| reasoning_max_drop_pos | derailment | 0.809 | higher -> wrong |
| reasoning_positive_mass | shape | 0.805 | higher -> wrong |
| reasoning_total_variation | thrashing | 0.764 | higher -> wrong |
| reasoning_start | shape | 0.750 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_mid_mean | shape | -0.911 | higher -> wrong |
| reasoning_trough_pos | timing | 0.906 | higher -> correct |
| reasoning_n_chunks | length | -0.716 | higher -> wrong |
| reasoning_max_rise_pos | timing | -0.659 | higher -> wrong |
| reasoning_late_mean | landing | 0.556 | higher -> correct |
| reasoning_max_drop_pos | derailment | 0.469 | higher -> correct |
| reasoning_early_mean | shape | 0.417 | higher -> correct |
| reasoning_negative_mass | shape | 0.366 | higher -> correct |
| reasoning_start | shape | 0.345 | higher -> correct |
| reasoning_direction_changes | thrashing | -0.316 | higher -> wrong |
| reasoning_time_positive | shape | -0.293 | higher -> wrong |
| reasoning_time_negative | shape | 0.293 | higher -> correct |

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
| premature_exit | inconclusive | 23 | 0.274 [0.024, 0.568] | 0.087 | 1 | - | - | - | - |
| rambling_overlong | inconclusive | 23 | 0.726 [0.432, 0.976] | 0.087 | 1 | - | - | - | - |
| thrashing | inconclusive | 23 | 0.595 [0.341, 0.817] | 0.087 | 2 | - | - | - | - |
| no_commitment | inconclusive | 23 | 0.429 [0.048, 0.857] | 0.087 | 2 | - | - | - | - |
| derailment_late | inconclusive | 23 | 0.571 [0.227, 0.864] | 0.087 | 2 | - | - | - | - |
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

- `mode_stack` ROC-AUC: **0.482**
- `trajectory_full` ROC-AUC: **0.186**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 0 | 0 / 2 | 0 | - | - |
| any | 11 | 0 / 2 | 8 | 0.000 | 0.000 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over `length_only` is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
