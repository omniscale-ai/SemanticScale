# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/SmolAgents`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 36.8% | 57 | 310 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Planning Hallucination | 72.0% | 25 |
| Reasoning Hallucination | 6.2% | 16 |
| Retrieval Hallucination | 12.5% | 16 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 12.5% | 8 |
| Math | 40.0% | 20 |
| Science | 38.5% | 13 |
| World Knowledge | 43.8% | 16 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 31.4% | 31.3% | 33.3% | 57 | 3 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Planning Hallucination | 45.8% | 44.0% | 44.9% | 25 |
| Reasoning Hallucination | 25.0% | 25.0% | 25.0% | 16 |
| Retrieval Hallucination | 23.5% | 25.0% | 24.2% | 16 |
