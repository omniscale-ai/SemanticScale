# Score Regression — `frontierscience/deepseek/deepseek-v3.2_reasoning-auto`

**Target:** `rubric_score` ∈ [0, 1] (rubric items only, n = 58).

**CV:** KFold(5, shuffle=True, random_state=42). Bootstrap: 1000 draws, percentile CI.

## Pooled OOF metrics

| Model | n | R² [95% CI] | Spearman ρ [95% CI] | MAE [95% CI] |
|---|---:|---|---|---|
| `ridge` (trajectory_full) | 58 | -1.816 [-3.758, -0.623] | +0.151 [-0.122, +0.408] | +0.226 [+0.176, +0.282] |
| `lightgbm_reg` (trajectory_full) | 58 | -0.427 [-0.872, -0.128] | -0.039 [-0.302, +0.232] | +0.177 [+0.148, +0.206] |
| `ridge` (length_only) | 58 | -0.189 [-0.393, -0.063] | -0.272 [-0.495, -0.022] | +0.168 [+0.141, +0.194] |

## Per-fold metrics (mean ± std)

| Model | R² | MAE | Spearman ρ |
|---|---|---|---|
| `ridge` (trajectory_full) | -2.311 ± 1.929 | — | +0.288 ± 0.344 |
| `lightgbm_reg` (trajectory_full) | -0.562 ± 0.202 | — | +0.057 ± 0.247 |
| `ridge` (length_only) | -0.303 ± 0.207 | — | -0.038 ± 0.197 |

## Reading this report

- Headline metric is Spearman ρ. With n = 58, bootstrap CIs are wide; treat any model whose ρ-CI overlaps 0 as null on this run.
- The length_only ridge baseline says how much of the signal is just "longer answers tend to score higher / lower". Trajectory shape only matters if ρ_trajectory > ρ_length_only (with non-overlapping CI).
- R² is shown for completeness but is unstable on small left-skewed distributions; a near-zero or negative R² with a positive Spearman ρ means the model recovers ranks but not magnitudes.

Artifacts in `artifacts/`: `oof_regression_{ridge,lightgbm_reg}.parquet`, `regression_metadata_*.json`, `regression_importance_*.csv`, `length_only/oof_regression_ridge.parquet`.
