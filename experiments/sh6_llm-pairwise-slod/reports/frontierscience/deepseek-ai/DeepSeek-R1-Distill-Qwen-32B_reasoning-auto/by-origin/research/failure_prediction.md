# SH6 frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto/by-origin/research — Failure Prediction

## Setup

- Target label: `is_correct`
- Items analysed: 59
- Positive class (`is_correct=true`): 3
- Negative class (`is_correct=false`): 56

## Status

- Reduced cross-validation folds from 5 to 3 due to class counts.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.535 +/- 0.238 | 0.139 +/- 0.080 | 0.565 +/- 0.311 | 0.475 +/- 0.184 | 0.139 +/- 0.104 |
| trajectory_shape (logreg) | 96 | 0.351 +/- 0.460 | 0.368 +/- 0.447 | 0.658 +/- 0.242 | 0.950 +/- 0.041 | 0.333 +/- 0.471 |
| trajectory_full (logreg) | 99 | 0.368 +/- 0.447 | 0.368 +/- 0.447 | 0.658 +/- 0.242 | 0.950 +/- 0.041 | 0.333 +/- 0.471 |
| reasoning_traj (MiniRocket) | 20 | 0.754 +/- 0.216 | 0.430 +/- 0.405 | 0.500 +/- 0.000 | 0.949 +/- 0.001 | 0.000 +/- 0.000 |
| trajectory_full (lightgbm) | 99 | 0.716 +/- 0.105 | 0.176 +/- 0.057 | 0.500 +/- 0.000 | 0.949 +/- 0.001 | 0.000 +/- 0.000 |
| mode_stack (logreg) | 13 | 0.772 +/- 0.174 | 0.426 +/- 0.407 | 0.621 +/- 0.210 | 0.881 +/- 0.049 | 0.133 +/- 0.189 |
| mode_stack (lightgbm) | 13 | 0.521 +/- 0.223 | 0.118 +/- 0.043 | 0.463 +/- 0.052 | 0.879 +/- 0.100 | 0.000 +/- 0.000 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_late_mean | landing | 0.982 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | 0.946 | higher -> correct |
| answer_traj_t18 | shape | 0.929 | higher -> correct |
| reasoning_traj_t09 | shape | 0.929 | higher -> wrong |
| answer_std | shape | 0.927 | higher -> wrong |
| reasoning_traj_t02 | shape | 0.912 | higher -> correct |
| reasoning_traj_t10 | shape | 0.908 | higher -> wrong |
| reasoning_traj_t15 | shape | 0.908 | higher -> wrong |
| answer_traj_t08 | shape | 0.877 | higher -> wrong |
| reasoning_zero_crossings | thrashing | 0.872 | higher -> wrong |
| reasoning_traj_t13 | shape | 0.858 | higher -> wrong |
| reasoning_curvature_abs_mean | thrashing | 0.854 | higher -> wrong |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_traj_t02 | shape | 0.447 | higher -> correct |
| reasoning_traj_t07 | shape | -0.379 | higher -> wrong |
| reasoning_traj_t14 | shape | -0.321 | higher -> wrong |
| answer_traj_t17 | shape | -0.297 | higher -> wrong |
| answer_range_minus_reasoning_range | commitment | -0.286 | higher -> wrong |
| answer_traj_t10 | shape | 0.280 | higher -> correct |
| reasoning_traj_t17 | shape | 0.277 | higher -> correct |
| answer_peak_pos | timing | 0.273 | higher -> correct |
| answer_minus_reasoning_mean | answer_alignment | 0.261 | higher -> correct |
| answer_traj_t18 | shape | -0.258 | higher -> wrong |
| reasoning_std | shape | 0.257 | higher -> correct |
| reasoning_range | commitment | 0.246 | higher -> correct |

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
| premature_exit | inconclusive | 59 | 0.220 [0.031, 0.552] | 0.949 | 5 | - | - | - | - |
| rambling_overlong | inconclusive | 59 | 0.780 [0.448, 0.969] | 0.949 | 29 | - | - | - | - |
| thrashing | inconclusive | 59 | 0.586 [0.034, 0.931] | 0.949 | 7 | - | - | - | - |
| no_commitment | inconclusive | 59 | 0.601 [0.224, 0.895] | 0.949 | 18 | - | - | - | - |
| derailment_late | inconclusive | 59 | 0.521 [0.268, 0.862] | 0.949 | 21 | - | - | - | - |
| answer_drift | inconclusive | 58 | 0.479 [0.140, 0.893] | 0.948 | 10 | - | - | - | - |
| answer_meandering | inconclusive | 58 | 0.697 [0.263, 1.000] | 0.948 | 18 | - | - | - | - |
| answer_volatility | inconclusive | 58 | 0.697 [0.456, 1.000] | 0.948 | 30 | - | - | - | - |
| answer_uncommitted | inconclusive | 58 | 0.333 [0.107, 0.661] | 0.948 | 11 | - | - | - | - |
| answer_overrange | confirmed | 58 | 0.927 [0.842, 1.000] | 0.948 | 51 | 0.980 | 0.909 | 0.943 | 1.034 |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: answer_overrange.
- **Inconclusive**: premature_exit, rambling_overlong, thrashing, no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.772**
- `trajectory_full` ROC-AUC: **0.368**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

> Caveat: the FrontierScience capture number is **inflated** because the answer-side detectors (`answer_meandering`, `answer_volatility`, `answer_uncommitted`, `answer_overrange`) were selected post-hoc by ranking univariate AUCs on this dataset. Treat this number as a descriptive upper bound. The SWE-agent capture number, where the reasoning-side detectors were pre-registered, is the unbiased estimate.

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 50 / 56 | 51 | 0.893 | 0.980 |
| any | 11 | 56 / 56 | 59 | 1.000 | 0.949 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
