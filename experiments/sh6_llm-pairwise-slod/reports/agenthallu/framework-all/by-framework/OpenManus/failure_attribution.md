# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/OpenManus`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 46.4% | 84 | 467 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Human-Interaction Hallucination | 59.5% | 42 |
| Reasoning Hallucination | 60.0% | 20 |
| Retrieval Hallucination | 9.1% | 22 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 0.0% | 2 |
| Math | 64.7% | 17 |
| Science | 64.5% | 31 |
| World Knowledge | 23.5% | 34 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 52.3% | 52.2% | 54.8% | 84 | 3 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Human-Interaction Hallucination | 59.1% | 61.9% | 60.5% | 42 |
| Reasoning Hallucination | 47.1% | 40.0% | 43.2% | 20 |
| Retrieval Hallucination | 52.2% | 54.5% | 53.3% | 22 |
