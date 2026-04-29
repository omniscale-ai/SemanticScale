# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 23.0% | 443 | 3315 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Human-Interaction Hallucination | 34.2% | 73 |
| Planning Hallucination | 58.2% | 67 |
| Reasoning Hallucination | 5.9% | 118 |
| Retrieval Hallucination | 13.4% | 82 |
| Tool-Use Hallucination | 19.4% | 103 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 15.4% | 39 |
| Math | 28.2% | 85 |
| Science | 28.4% | 109 |
| Tool Use | 19.4% | 103 |
| World Knowledge | 19.6% | 107 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 44.4% | 44.5% | 46.7% | 443 | 5 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Human-Interaction Hallucination | 29.3% | 30.1% | 29.7% | 73 |
| Planning Hallucination | 22.7% | 29.9% | 25.8% | 67 |
| Reasoning Hallucination | 41.5% | 33.1% | 36.8% | 118 |
| Retrieval Hallucination | 32.9% | 34.1% | 33.5% | 82 |
| Tool-Use Hallucination | 97.0% | 95.1% | 96.1% | 103 |
