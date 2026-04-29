# Score Regression — `frontierscience/R1-Distill-32B-cloudjudge_reasoning-auto`

**Target:** `rubric_score` ∈ [0, 1] (rubric items only, n = 59).

**CV:** KFold(5, shuffle=True, random_state=42). Bootstrap: 1000 draws, percentile CI.

## Pooled OOF metrics

| Model | n | R² [95% CI] | Spearman ρ [95% CI] | MAE [95% CI] |
|---|---:|---|---|---|
| `ridge` (trajectory_full) | 59 | -1.116 [-2.282, -0.529] | +0.145 [-0.138, +0.406] | +0.110 [+0.088, +0.132] |
| `lightgbm_reg` (trajectory_full) | 59 | -0.664 [-1.051, -0.424] | -0.180 [-0.435, +0.116] | +0.096 [+0.078, +0.117] |
| `ridge` (length_only) | 59 | -0.187 [-0.396, -0.081] | -0.215 [-0.428, +0.027] | +0.084 [+0.069, +0.100] |

## Per-fold metrics (mean ± std)

| Model | R² | MAE | Spearman ρ |
|---|---|---|---|
| `ridge` (trajectory_full) | -1.795 ± 0.995 | — | +0.177 ± 0.248 |
| `lightgbm_reg` (trajectory_full) | -1.024 ± 0.434 | — | -0.189 ± 0.253 |
| `ridge` (length_only) | -0.389 ± 0.404 | — | -0.138 ± 0.178 |

## Reading this report

- Headline metric is Spearman ρ. With n = 59, bootstrap CIs are wide; treat any model whose ρ-CI overlaps 0 as null on this run.
- The length_only ridge baseline says how much of the signal is just "longer answers tend to score higher / lower". Trajectory shape only matters if ρ_trajectory > ρ_length_only (with non-overlapping CI).
- R² is shown for completeness but is unstable on small left-skewed distributions; a near-zero or negative R² with a positive Spearman ρ means the model recovers ranks but not magnitudes.

Artifacts in `artifacts/`: `oof_regression_{ridge,lightgbm_reg}.parquet`, `regression_metadata_*.json`, `regression_importance_*.csv`, `length_only/oof_regression_ridge.parquet`.
