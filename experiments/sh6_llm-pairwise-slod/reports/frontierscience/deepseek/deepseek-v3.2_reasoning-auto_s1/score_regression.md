# Score Regression — `frontierscience/deepseek/deepseek-v3.2_reasoning-auto_s1`

**Target:** `rubric_score` ∈ [0, 1] (rubric items only, n = 52).

**CV:** KFold(5, shuffle=True, random_state=42). Bootstrap: 1000 draws, percentile CI.

## Pooled OOF metrics

| Model | n | R² [95% CI] | Spearman ρ [95% CI] | MAE [95% CI] |
|---|---:|---|---|---|
| `ridge` (trajectory_full) | 52 | -1.615 [-3.008, -0.666] | +0.089 [-0.188, +0.357] | +0.210 [+0.167, +0.255] |
| `lightgbm_reg` (trajectory_full) | 52 | -0.760 [-1.380, -0.369] | -0.184 [-0.433, +0.067] | +0.176 [+0.145, +0.210] |
| `ridge` (length_only) | 52 | -0.015 [-0.223, +0.122] | +0.055 [-0.225, +0.331] | +0.130 [+0.107, +0.157] |

## Per-fold metrics (mean ± std)

| Model | R² | MAE | Spearman ρ |
|---|---|---|---|
| `ridge` (trajectory_full) | -1.754 ± 0.735 | — | +0.111 ± 0.191 |
| `lightgbm_reg` (trajectory_full) | -0.884 ± 0.471 | — | -0.144 ± 0.320 |
| `ridge` (length_only) | -0.080 ± 0.112 | — | +0.166 ± 0.185 |

## Reading this report

- Headline metric is Spearman ρ. With n = 52, bootstrap CIs are wide; treat any model whose ρ-CI overlaps 0 as null on this run.
- The length_only ridge baseline says how much of the signal is just "longer answers tend to score higher / lower". Trajectory shape only matters if ρ_trajectory > ρ_length_only (with non-overlapping CI).
- R² is shown for completeness but is unstable on small left-skewed distributions; a near-zero or negative R² with a positive Spearman ρ means the model recovers ranks but not magnitudes.

Artifacts in `artifacts/`: `oof_regression_{ridge,lightgbm_reg}.parquet`, `regression_metadata_*.json`, `regression_importance_*.csv`, `length_only/oof_regression_ridge.parquet`.
