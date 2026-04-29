# Score Regression — `frontierscience/Qwen/Qwen3-30B-A3B-Instruct-2507_reasoning-none`

**Target:** `rubric_score` ∈ [0, 1] (rubric items only, n = 60).

**CV:** KFold(5, shuffle=True, random_state=42). Bootstrap: 1000 draws, percentile CI.

## Pooled OOF metrics

| Model | n | R² [95% CI] | Spearman ρ [95% CI] | MAE [95% CI] |
|---|---:|---|---|---|
| `ridge` (trajectory_full) | 60 | -0.322 [-0.902, +0.019] | +0.259 [+0.003, +0.470] | +0.247 [+0.203, +0.288] |
| `lightgbm_reg` (trajectory_full) | 60 | -0.099 [-0.567, +0.178] | +0.291 [+0.045, +0.508] | +0.223 [+0.187, +0.259] |
| `ridge` (length_only) | 60 | +0.222 [-0.175, +0.465] | +0.394 [+0.104, +0.638] | +0.185 [+0.154, +0.219] |

## Per-fold metrics (mean ± std)

| Model | R² | MAE | Spearman ρ |
|---|---|---|---|
| `ridge` (trajectory_full) | -0.725 ± 0.863 | — | +0.258 ± 0.373 |
| `lightgbm_reg` (trajectory_full) | -0.293 ± 0.422 | — | +0.334 ± 0.199 |
| `ridge` (length_only) | +0.091 ± 0.450 | — | +0.483 ± 0.220 |

## Reading this report

- Headline metric is Spearman ρ. With n = 60, bootstrap CIs are wide; treat any model whose ρ-CI overlaps 0 as null on this run.
- The length_only ridge baseline says how much of the signal is just "longer answers tend to score higher / lower". Trajectory shape only matters if ρ_trajectory > ρ_length_only (with non-overlapping CI).
- R² is shown for completeness but is unstable on small left-skewed distributions; a near-zero or negative R² with a positive Spearman ρ means the model recovers ranks but not magnitudes.

Artifacts in `artifacts/`: `oof_regression_{ridge,lightgbm_reg}.parquet`, `regression_metadata_*.json`, `regression_importance_*.csv`, `length_only/oof_regression_ridge.parquet`.
