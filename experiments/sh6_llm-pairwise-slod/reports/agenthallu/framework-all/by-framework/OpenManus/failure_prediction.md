# SH6 agenthallu/framework-all/by-framework/OpenManus — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 104
- Positive class (`final_answer_correct=true`): 20
- Negative class (`final_answer_correct=false`): 84
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 104 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| length_only (logreg) | 3 | 0.638 +/- 0.181 | 0.425 +/- 0.173 | 0.546 +/- 0.157 | 0.635 +/- 0.128 | 0.302 +/- 0.213 |
| trajectory_shape (logreg) | 96 | 0.477 +/- 0.081 | 0.246 +/- 0.054 | 0.525 +/- 0.074 | 0.664 +/- 0.066 | 0.239 +/- 0.135 |
| trajectory_full (logreg) | 99 | 0.472 +/- 0.072 | 0.242 +/- 0.041 | 0.500 +/- 0.083 | 0.654 +/- 0.080 | 0.210 +/- 0.133 |
| reasoning_traj (MiniRocket) | 20 | 0.473 +/- 0.072 | 0.275 +/- 0.093 | 0.463 +/- 0.088 | 0.655 +/- 0.112 | 0.145 +/- 0.128 |
| trajectory_full (lightgbm) | 99 | 0.545 +/- 0.206 | 0.327 +/- 0.116 | 0.534 +/- 0.045 | 0.740 +/- 0.054 | 0.218 +/- 0.115 |
| mode_stack (logreg) | 13 | 0.569 +/- 0.173 | 0.371 +/- 0.160 | 0.574 +/- 0.127 | 0.588 +/- 0.089 | 0.325 +/- 0.165 |
| mode_stack (lightgbm) | 13 | 0.572 +/- 0.252 | 0.409 +/- 0.202 | 0.623 +/- 0.164 | 0.760 +/- 0.095 | 0.367 +/- 0.272 |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| reasoning_start | shape | 0.684 | higher -> wrong |
| reasoning_traj_t16 | shape | 0.678 | higher -> wrong |
| answer_traj_t17 | shape | 0.672 | higher -> wrong |
| reasoning_max_rise | shape | 0.656 | higher -> correct |
| reasoning_late_mean | landing | 0.651 | higher -> wrong |
| reasoning_late_minus_early | transition | 0.650 | higher -> wrong |
| reasoning_min | shape | 0.647 | higher -> correct |
| answer_end | landing | 0.641 | higher -> correct |
| reasoning_traj_t03 | shape | 0.641 | higher -> correct |
| total_n_chunks | length | 0.639 | higher -> correct |
| reasoning_traj_t07 | shape | 0.638 | higher -> correct |
| reasoning_n_chunks | length | 0.636 | higher -> correct |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| reasoning_traj_t03 | shape | -1.163 | higher -> wrong |
| reasoning_traj_t01 | shape | 0.949 | higher -> correct |
| answer_max_drop_pos | derailment | -0.922 | higher -> wrong |
| reasoning_peak_pos | timing | -0.819 | higher -> wrong |
| reasoning_max_rise | shape | 0.792 | higher -> correct |
| reasoning_traj_t07 | shape | 0.744 | higher -> correct |
| reasoning_monotonicity | commitment | -0.719 | higher -> wrong |
| reasoning_traj_t14 | shape | 0.711 | higher -> correct |
| answer_max_rise_pos | timing | 0.697 | higher -> correct |
| reasoning_max_drop | derailment | -0.691 | higher -> wrong |
| answer_traj_t02 | shape | 0.679 | higher -> correct |
| answer_min | shape | -0.670 | higher -> wrong |

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
| premature_exit | confirmed | 104 | 0.650 [0.514, 0.765] | 0.808 | 20 | 0.900 | 0.214 | 0.346 | 1.114 |
| rambling_overlong | inverted | 104 | 0.350 [0.235, 0.486] | 0.808 | 4 | - | - | - | - |
| thrashing | inverted | 104 | 0.352 [0.230, 0.481] | 0.808 | 5 | - | - | - | - |
| no_commitment | inconclusive | 104 | 0.439 [0.314, 0.575] | 0.808 | 11 | - | - | - | - |
| derailment_late | inconclusive | 104 | 0.578 [0.450, 0.691] | 0.808 | 19 | - | - | - | - |
| answer_drift | inconclusive | 104 | 0.637 [0.489, 0.764] | 0.808 | 25 | - | - | - | - |
| answer_meandering | inconclusive | 104 | 0.399 [0.247, 0.550] | 0.808 | 8 | - | - | - | - |
| answer_volatility | inconclusive | 104 | 0.386 [0.229, 0.540] | 0.808 | 7 | - | - | - | - |
| answer_uncommitted | inconclusive | 104 | 0.444 [0.317, 0.573] | 0.808 | 0 | - | - | - | - |
| answer_overrange | inconclusive | 104 | 0.399 [0.266, 0.543] | 0.808 | 11 | - | - | - | - |
| truncation_abort | insufficient_data | 0 | - | - | 0 | - | - | - | - |

### Verdict summary

- **Confirmed on this run**: premature_exit.
- **Hypothesis falsified (inverted)**: rambling_overlong, thrashing. The score direction flipped — higher values predict success, not failure, on this dataset. This is a real negative result, not a detector failure.
- **Inconclusive**: no_commitment, derailment_late, answer_drift, answer_meandering, answer_volatility, answer_uncommitted, answer_overrange. The bootstrap CI spans 0.5, so we cannot reject the null on this run.

### Signal capture

How much of the failure-prediction signal does the named-mode taxonomy actually carry? The `mode_stack` model is a logistic regression on the detector scores only; the comparison set is the `trajectory_full` model fit on all trajectory features.

- `mode_stack` ROC-AUC: **0.569**
- `trajectory_full` ROC-AUC: **0.472**
- Above-chance discrimination preserved by the mode stack: **n/a** ((AUC_modes − 0.5) ÷ (AUC_full − 0.5))

### Failure coverage

What fraction of failures get flagged by at least one detector? `any` uses the union of every detector's flag; `confirmed` restricts to detectors whose directional hypothesis was confirmed on this run, which is the principled coverage number.

| Variant | # Modes | Failures caught | Items flagged | Recall | Precision |
|---|---|---|---|---|---|
| confirmed | 1 | 18 / 84 | 20 | 0.214 | 0.900 |
| any | 11 | 56 / 84 | 67 | 0.667 | 0.836 |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
