# Score Regression — `frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto`

**Target:** `rubric_score` ∈ [0, 1] (rubric items only, n = 59).

**CV:** KFold(5, shuffle=True, random_state=42). Bootstrap: 1000 draws, percentile CI.

## Pooled OOF metrics

| Model | n | R² [95% CI] | Spearman ρ [95% CI] | MAE [95% CI] |
|---|---:|---|---|---|
| `ridge` (trajectory_full) | 59 | -1.929 [-3.607, -0.864] | +0.122 [-0.142, +0.378] | +0.251 [+0.200, +0.314] |
| `lightgbm_reg` (trajectory_full) | 59 | -0.410 [-0.873, -0.128] | +0.007 [-0.226, +0.246] | +0.186 [+0.152, +0.223] |
| `ridge` (length_only) | 59 | +0.009 [-0.176, +0.082] | +0.059 [-0.201, +0.276] | +0.157 [+0.128, +0.192] |

## Per-fold metrics (mean ± std)

| Model | R² | MAE | Spearman ρ |
|---|---|---|---|
| `ridge` (trajectory_full) | -2.837 ± 2.395 | — | +0.092 ± 0.372 |
| `lightgbm_reg` (trajectory_full) | -0.505 ± 0.347 | — | -0.030 ± 0.304 |
| `ridge` (length_only) | -0.081 ± 0.231 | — | +0.270 ± 0.196 |

## Reading this report

- Headline metric is Spearman ρ. With n = 59, bootstrap CIs are wide; treat any model whose ρ-CI overlaps 0 as null on this run.
- The length_only ridge baseline says how much of the signal is just "longer answers tend to score higher / lower". Trajectory shape only matters if ρ_trajectory > ρ_length_only (with non-overlapping CI).
- R² is shown for completeness but is unstable on small left-skewed distributions; a near-zero or negative R² with a positive Spearman ρ means the model recovers ranks but not magnitudes.

Artifacts in `artifacts/`: `oof_regression_{ridge,lightgbm_reg}.parquet`, `regression_metadata_*.json`, `regression_importance_*.csv`, `length_only/oof_regression_ridge.parquet`.
