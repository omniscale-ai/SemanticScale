# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenterrorbench`  
**Run:** `env-all_provider-all_model-all`  
**Type label field:** `critical_error_module`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 12.6% | 199 | 5009 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| action | 22.7% | 22 |
| memory | 5.3% | 38 |
| plan | 4.1% | 49 |
| planning | 31.0% | 29 |
| reflection | 12.8% | 39 |
| system | 9.1% | 22 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| alfworld | 7.0% | 100 |
| gaia | 26.5% | 49 |
| webshop | 10.0% | 50 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 27.8% | 27.6% | 28.6% | 199 | 6 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| action | 12.0% | 13.6% | 12.8% | 22 |
| memory | 33.3% | 31.6% | 32.4% | 38 |
| plan | 38.5% | 30.6% | 34.1% | 49 |
| planning | 44.4% | 41.4% | 42.9% | 29 |
| reflection | 21.3% | 25.6% | 23.3% | 39 |
| system | 20.0% | 22.7% | 21.3% | 22 |
