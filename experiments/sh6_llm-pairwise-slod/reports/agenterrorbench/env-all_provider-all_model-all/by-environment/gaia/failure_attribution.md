# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenterrorbench`  
**Run:** `env-all_provider-all_model-all/by-environment/gaia`  
**Type label field:** `critical_error_module`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 22.4% | 49 | 713 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| action | 16.7% | 6 |
| memory | 0.0% | 5 |
| planning | 30.8% | 26 |
| reflection | 12.5% | 8 |
| system | 25.0% | 4 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| gaia | 22.4% | 49 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 37.4% | 36.8% | 53.3% | 45 | 4 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| action | 16.7% | 16.7% | 16.7% | 6 |
| memory | 33.3% | 20.0% | 25.0% | 5 |
| planning | 67.9% | 73.1% | 70.4% | 26 |
| reflection | 37.5% | 37.5% | 37.5% | 8 |
