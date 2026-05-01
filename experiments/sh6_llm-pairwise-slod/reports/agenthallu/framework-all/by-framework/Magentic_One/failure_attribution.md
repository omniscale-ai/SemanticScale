# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/Magentic_One`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 27.8% | 54 | 560 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Planning Hallucination | 70.0% | 20 |
| Reasoning Hallucination | 0.0% | 26 |
| Retrieval Hallucination | 12.5% | 8 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 10.0% | 10 |
| Math | 23.1% | 13 |
| Science | 56.2% | 16 |
| World Knowledge | 13.3% | 15 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 46.1% | 47.8% | 46.3% | 54 | 3 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Planning Hallucination | 47.8% | 55.0% | 51.2% | 20 |
| Reasoning Hallucination | 47.6% | 38.5% | 42.6% | 26 |
| Retrieval Hallucination | 40.0% | 50.0% | 44.4% | 8 |
