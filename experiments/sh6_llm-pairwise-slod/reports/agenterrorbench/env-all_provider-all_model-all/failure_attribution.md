# SH6 Failure Attribution (SLoD-only)

**Dataset:** `agenterrorbench`  
**Run:** `env-all_provider-all_model-all`  
**Type label field:** `critical_error_type`

## Failure Location

| Model | Step localization accuracy | Failed traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 12.6% | 199 | 5009 | 5 |

### By failure type

| Failure type | Step localization accuracy | Failed traces |
|---|---|---|
| Parameter_error | 0.0% | 1 |
| causal_misattribution | 0.0% | 5 |
| constraint_ignorance | 30.8% | 13 |
| environment_error | 14.3% | 7 |
| hallucination | 15.4% | 13 |
| impossible_action | 10.0% | 10 |
| inefficient_plan | 11.6% | 43 |
| invalid_action | 0.0% | 1 |
| llm_limit | 100.0% | 1 |
| memory_retrieval_failure | 0.0% | 2 |
| misalignment | 14.3% | 7 |
| outcome_misinterpretation | 8.3% | 12 |
| over_simplification | 0.0% | 20 |
| parameter_error | 33.3% | 3 |
| plan_inefficient | 0.0% | 3 |
| progress_misjudge | 21.1% | 19 |
| step_limit | 0.0% | 8 |
| tool_execution_error | 0.0% | 3 |
| tool_execution_error  | 0.0% | 1 |
| unknown | 14.8% | 27 |

### By subject

| Subject | Step localization accuracy | Failed traces |
|---|---|---|
| alfworld | 7.0% | 100 |
| gaia | 26.5% | 49 |
| webshop | 10.0% | 50 |

## Failure Type

| Model | Macro-F1 | Macro-recall | Accuracy | Items | Classes | CV folds |
|---|---|---|---|---|---|---|
| logreg | 13.1% | 12.8% | 15.9% | 157 | 11 | 5 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| causal_misattribution | 0.0% | 0.0% | 0.0% | 5 |
| constraint_ignorance | 25.0% | 23.1% | 24.0% | 13 |
| environment_error | 0.0% | 0.0% | 0.0% | 7 |
| hallucination | 9.1% | 7.7% | 8.3% | 13 |
| impossible_action | 5.9% | 10.0% | 7.4% | 10 |
| inefficient_plan | 27.0% | 23.3% | 25.0% | 43 |
| misalignment | 20.0% | 14.3% | 16.7% | 7 |
| outcome_misinterpretation | 25.0% | 25.0% | 25.0% | 12 |
| over_simplification | 19.0% | 20.0% | 19.5% | 20 |
| progress_misjudge | 5.0% | 5.3% | 5.1% | 19 |
| step_limit | 14.3% | 12.5% | 13.3% | 8 |
