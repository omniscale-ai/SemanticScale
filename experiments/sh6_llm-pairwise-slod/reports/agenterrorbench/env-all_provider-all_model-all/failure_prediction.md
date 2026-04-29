# SH6 agenterrorbench/env-all_provider-all_model-all — Failure Prediction

## Setup

- Target label: `final_answer_correct`
- Items analysed: 199
- Positive class (`final_answer_correct=true`): 0
- Negative class (`final_answer_correct=false`): 199
- Label agreement (`is_correct` vs `final_answer_correct`): 100.0% over 199 items
- Final answer correct but reasoning wrong: 0
- Final answer wrong but reasoning clean: 0

## Status

- Prediction is not identifiable because the run contains only one target class.

## Cross-Validated Prediction

| Model | # Features | ROC-AUC | Avg Precision | Balanced Acc. | Accuracy | F1 |
|---|---|---|---|---|---|---|
| Not run | - | - | - | - | - | - |

## Top Single Features

| Feature | Family | Signal ROC-AUC | Direction |
|---|---|---|---|
| Not available | - | - | - |

## Strongest Multivariate Coefficients

| Feature | Family | Coefficient | Direction |
|---|---|---|---|
| Not available | - | - | - |

## Interpretation Notes

- `trajectory_shape` excludes chunk-count features, so any lift over the structural baseline is genuine trajectory signal.
- Pair-density columns are saved in the feature CSV for diagnostics, but excluded from the prediction models because they reflect ranking coverage rather than reasoning behavior.
- Positive coefficients mean higher feature values predict final-answer success; negative coefficients predict failure.
- `signal ROC-AUC` treats both directions symmetrically, so values closer to 1.0 indicate stronger standalone predictive signal.
- Failure-mode detectors are calibrated against the success distribution (90th percentile by default), and their directional claim is tested with a bootstrap AUC CI against 0.5. Precision/recall are only shown for `confirmed` detectors; `inverted` detectors are reported as honest negative results.
