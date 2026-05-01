# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/Camel`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 35.1% | 57 | 523 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Human-Interaction Hallucination | 45.2% | 31 |
| Planning Hallucination | 0.0% | 5 |
| Reasoning Hallucination | 21.4% | 14 |
| Retrieval Hallucination | 42.9% | 7 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 12.5% | 8 |
| Math | 41.7% | 12 |
| Science | 39.3% | 28 |
| World Knowledge | 33.3% | 9 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 18.7% | 17.5% | 31.6% | 57 | 4 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Human-Interaction Hallucination | 55.6% | 48.4% | 51.7% | 31 |
| Planning Hallucination | 0.0% | 0.0% | 0.0% | 5 |
| Reasoning Hallucination | 25.0% | 21.4% | 23.1% | 14 |
| Retrieval Hallucination | 0.0% | 0.0% | 0.0% | 7 |
