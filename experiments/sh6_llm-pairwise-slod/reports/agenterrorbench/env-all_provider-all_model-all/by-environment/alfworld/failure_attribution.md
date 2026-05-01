# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenterrorbench`  
**Run:** `env-all_provider-all_model-all/by-environment/alfworld`  
**Type label field:** `critical_error_module`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 6.0% | 100 | 3000 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| action | 7.1% | 14 |
| memory | 5.6% | 18 |
| plan | 6.7% | 30 |
| planning | 0.0% | 3 |
| reflection | 4.5% | 22 |
| system | 7.7% | 13 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| alfworld | 6.0% | 100 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 18.0% | 18.7% | 19.6% | 97 | 5 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| action | 0.0% | 0.0% | 0.0% | 14 |
| memory | 20.7% | 33.3% | 25.5% | 18 |
| plan | 41.2% | 23.3% | 29.8% | 30 |
| reflection | 12.5% | 13.6% | 13.0% | 22 |
| system | 20.0% | 23.1% | 21.4% | 13 |
