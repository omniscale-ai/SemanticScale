# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/OpenDeepSearch`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 20.7% | 58 | 288 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Planning Hallucination | 25.0% | 16 |
| Reasoning Hallucination | 11.8% | 17 |
| Retrieval Hallucination | 24.0% | 25 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 12.5% | 8 |
| Math | 23.1% | 13 |
| Science | 25.0% | 12 |
| World Knowledge | 20.0% | 25 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 23.7% | 23.7% | 24.1% | 58 | 3 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Planning Hallucination | 29.4% | 31.2% | 30.3% | 16 |
| Reasoning Hallucination | 10.5% | 11.8% | 11.1% | 17 |
| Retrieval Hallucination | 31.8% | 28.0% | 29.8% | 25 |
