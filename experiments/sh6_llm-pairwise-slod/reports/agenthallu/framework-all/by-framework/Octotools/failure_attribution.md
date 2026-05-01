# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/Octotools`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 40.0% | 30 | 145 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Planning Hallucination | 0.0% | 1 |
| Reasoning Hallucination | 44.0% | 25 |
| Retrieval Hallucination | 25.0% | 4 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| General Assistant | 33.3% | 3 |
| Math | 40.0% | 10 |
| Science | 55.6% | 9 |
| World Knowledge | 25.0% | 8 |

## Failure Type

_Skipped: Fewer than 2 failure-type classes have >= 5 samples (counts: {'Reasoning Hallucination': 25, 'Retrieval Hallucination': 4, 'Planning Hallucination': 1})._
