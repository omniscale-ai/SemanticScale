# AgentHallu SLoD Trajectory Baseline

**Run:** `framework-all`

## Judgment

| Model | Feature set | Macro-F1 | Macro-recall | Accuracy | n hallucinated | n clean |
|---|---|---|---|---|---|---|
| logreg | trajectory_full | 49.0% | 50.3% | 49.6% | 443 | 250 |

## Attribution

| Model | Step localization accuracy | Hallucinated traces | Step rows | CV folds |
|---|---|---|---|---|
| logreg | 23.0% | 443 | 3315 | 5 |

## Attribution by Hallucination Category

| Category | Step localization accuracy | Hallucinated traces |
|---|---|---|
| Human-Interaction Hallucination | 34.2% | 73 |
| Planning Hallucination | 58.2% | 67 |
| Reasoning Hallucination | 5.9% | 118 |
| Retrieval Hallucination | 13.4% | 82 |
| Tool-Use Hallucination | 19.4% | 103 |
