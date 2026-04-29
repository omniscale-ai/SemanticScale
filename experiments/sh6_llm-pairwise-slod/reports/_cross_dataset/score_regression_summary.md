# FrontierScience Score Regression — Cross-Run Summary

Headline metric: pooled OOF Spearman ρ between predicted and actual `rubric_score`, with 95% percentile bootstrap CI on the rubric subset of each run.

| Run | n | ridge ρ [CI] | lightgbm_reg ρ [CI] | length_only ρ [CI] | ridge R² | lightgbm_reg R² |
|---|---:|---|---|---|---:|---:|
| `deepseek/deepseek-v3.2_reasoning-auto` | 58 | +0.151 [-0.122, +0.408] | -0.039 [-0.302, +0.232] | -0.272 [-0.495, -0.022] | -1.816 | -0.427 |
| `deepseek/deepseek-v3.2_reasoning-auto_s1` | 52 | +0.089 [-0.188, +0.357] | -0.184 [-0.433, +0.067] | +0.055 [-0.225, +0.331] | -1.615 | -0.760 |
| `Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none` | 60 | +0.259 [+0.003, +0.470] | +0.291 [+0.045, +0.508] | +0.394 [+0.104, +0.638] | -0.322 | -0.099 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto` | 59 | +0.122 [-0.142, +0.378] | +0.007 [-0.226, +0.246] | +0.059 [-0.201, +0.276] | -1.929 | -0.410 |
| `R1-Distill-32B-cloudjudge_reasoning-auto` | 59 | +0.145 [-0.138, +0.406] | -0.180 [-0.435, +0.116] | -0.215 [-0.428, +0.027] | -1.116 | -0.664 |

## How to read this

- `ridge` and `lightgbm_reg` use the full `trajectory_full` feature set (same predictors as the classifier path, just regressing rubric_score instead of classifying is_correct).
- `length_only` is the ridge baseline on `{reasoning,answer,total}_n_chunks` only. If the trajectory ρ is not materially above the length-only ρ, the score signal is mostly answer-length, not SLoD shape.
- ρ-CIs are wide because per-run n ≈ 55. Read the table by direction of effect across runs, not by any single CI.
- `_s1` / `_types-research` seeds and the s2/s3/s4 reseeds will be added once `02_slod.py` is run on those configs (currently only s1 has features extracted).

Cross-run CSV: `score_regression_summary.csv`. JSON dump of pooled metrics: `score_regression_summary.json`. Predicted-vs-actual scatter for the canonical run (`deepseek/deepseek-v3.2_reasoning-auto`): `score_regression_canonical_scatter.png`.
