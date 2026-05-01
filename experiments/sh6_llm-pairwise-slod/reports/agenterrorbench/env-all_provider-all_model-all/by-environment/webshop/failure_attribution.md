# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenterrorbench`  
**Run:** `env-all_provider-all_model-all/by-environment/webshop`  
**Type label field:** `critical_error_module`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 18.0% | 50 | 1296 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| action | 50.0% | 2 |
| memory | 20.0% | 15 |
| plan | 0.0% | 19 |
| reflection | 33.3% | 9 |
| system | 40.0% | 5 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| webshop | 18.0% | 50 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 28.6% | 29.0% | 39.6% | 48 | 4 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| memory | 25.0% | 20.0% | 22.2% | 15 |
| plan | 56.0% | 73.7% | 63.6% | 19 |
| reflection | 40.0% | 22.2% | 28.6% | 9 |
| system | 0.0% | 0.0% | 0.0% | 5 |
