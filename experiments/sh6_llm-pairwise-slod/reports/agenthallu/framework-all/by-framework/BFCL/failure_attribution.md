# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenthallu`  
**Run:** `framework-all/by-framework/BFCL`  
**Type label field:** `hallucination_category`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 28.2% | 103 | 1022 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Tool-Use Hallucination | 28.2% | 103 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| Tool Use | 28.2% | 103 |

## Failure Type

_Skipped: Fewer than 2 failure-type classes have >= 5 samples (counts: {'Tool-Use Hallucination': 103})._
