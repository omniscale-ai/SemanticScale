# SH6 Stage-5 — Model Comparison Protocol (Pre-Registered)

**Frozen:** 2026-04-29
**Owner:** stage-5 model-comparison track

This document fixes the experimental protocol for comparing alternative
classifiers on the existing Stage-5 trajectory feature matrix, and for
adding a raw-trajectory ceiling model. It is signed and frozen *before*
any AUC is computed on the comparison datasets.

The motivation comes from two observations on the current Stage-5
`length_only` / `trajectory_shape` / `trajectory_full` logistic-regression
baseline:

1. Logistic regression is linear; if there is meaningful interaction
   signal between trajectory features (e.g. *long answer × low
   monotonicity → fail*), it cannot capture it.
2. All features are summary statistics over the full trajectory. The
   sequence order itself is discarded after aggregation.

Stage-5 model comparison tests both gaps with two new models. To keep the
test honest, the protocol is fixed before any model is fit on the
comparison datasets.

## In-scope models

| Name | Input | Why |
|---|---|---|
| `logreg_trajectory_full` (incumbent) | trajectory_full features (~63 cols) | Existing baseline, untouched |
| `lightgbm_trajectory_full` (new) | trajectory_full features (~63 cols) | Tests interaction signal on the same features |
| `minirocket` (new, Step 6 only) | Raw 2-channel trajectory `(N, T, 2)` (reasoning + answer, length-T interpolated series) | Tests whether sequence order itself carries signal beyond aggregated features |

## In-scope datasets (7 runs)

Only runs that already have `trajectory_features.csv` and ≥2 examples in
each target class are eligible. The 1.5B distill (20 items, 1 success) is
excluded — CV is not identifiable.

| # | Dataset / run slug | N | Pos rate | Min class | Notes |
|---|---|---|---|---|---|
| 1 | `frontierscience/deepseek-v3.2_reasoning-auto` | 153 | 0.40 | 61 | Primary FS run |
| 2 | `frontierscience/R1-Distill-32B-cloudjudge_reasoning-auto` | 148 | 0.18 | ~27 | Weak generator, cloud judge |
| 3 | `frontierscience/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto` | 145 | 0.21 | ~31 | Local SH7 32B run |
| 4 | `swe-agent-trajectories/model-all` | 645 | 0.31 | 202 | Largest, reasoning-side detectors |
| 5 | `processbench/{omnimath, gsm8k, olympiadbench}` | 400 each | TBD | TBD | Process-level error labels |
| 6 | `agenthallu/framework-all` | 693 | 0.36 | 250 | Heterogeneous, current null result |
| 7 | `gpqa-diamond/deepseek-v3.2_reasoning-auto` | 192 | 0.77 | 45 | Heavy class imbalance, success-majority |

Datasets 1, 4, 6, 7 + at least one of (2, 3) and at least one of the
processbench splits constitute the **5+ dataset comparison set**.

## Train / eval policy

- **Per-dataset, independently.** No cross-dataset transfer.
- **5-fold StratifiedKFold**, `shuffle=True`, **`random_state=42`**.
  Identical to the existing logreg baseline so fold assignments line up
  exactly and OOF predictions are directly comparable per item.
- The existing `prediction_feasibility()` check in
  `failure_analysis.py` shrinks `cv_folds` when the minority class is
  too small. New models inherit the same shrunk fold count for the
  same dataset, so all three models always see the same folds.
- **No per-dataset hyperparameter search** in this first pass.
  Hyperparameters are frozen below. A separate, post-hoc tuning pass
  may be considered *only after* this comparison is filed and the
  finding is logged.

## Frozen hyperparameters

### LightGBM

```python
LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=10,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
    verbose=-1,
)
```

Wrapped in a `Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", LGBMClassifier(...))])`.
No `StandardScaler` — gradient boosting is scale-invariant. Median
imputation matches the logreg pipeline so missing-value handling is
identical.

### MiniRocket (Step 6 only)

```python
MiniRocketMultivariate(num_kernels=10000, random_state=42)
+ RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")
```

Input shape: `(N_items, n_channels=2, T=interp_points)` — reasoning and
answer trajectories on a shared per-item-centered scale, interpolated to
`interp_points` (currently 20). RidgeClassifierCV is the standard
companion classifier; we expose decision-function output as a probability
proxy via `1 / (1 + exp(-margin))` for ROC-AUC computation.

If a dataset has any item with `has_answer_chunks=False` (entire
answer-channel column is NaN — happens on swe-agent), the
answer channel is replaced with zeros and a flag column is added so
MiniRocket sees the same input shape across items.

## Decision rule

For each new model, compute Δ-AUC vs. `logreg_trajectory_full` on the
**same OOF folds**, with a **95% percentile-bootstrap CI** on the
difference (paired by item; 1000 resamples).

A new model **wins on a dataset** iff:

1. point estimate Δ-AUC ≥ **+0.03**, and
2. CI lower bound on Δ-AUC > **0**.

A new model is **considered to carry the comparison** iff it wins on
**≥3 of 5 datasets** in the comparison set. A simple cross-dataset
average is *not* the decision rule — averaging hides regressions on
small datasets.

### Stop conditions

| Outcome | Implication | Next step |
|---|---|---|
| LightGBM wins on ≥3/5 | Interaction signal exists | Treat LightGBM as new baseline; consider per-dataset tuning later |
| LightGBM wins on 1–2/5 | Weak / dataset-specific signal | Note in findings, do not promote, proceed to Step 6 only if motivated |
| LightGBM wins on 0/5 | No interaction signal in current features | Skip Step 6 (sequence model unlikely to help on aggregated input either); pivot to feature engineering (TA pack, multi-scale, cross-trajectory) |
| MiniRocket wins on ≥3/5 over LightGBM | Sequence order carries signal beyond aggregations | Feature engineering should target sequence-derived signals first |
| MiniRocket wins on 0/5 vs LightGBM | Aggregated features capture all extractable signal | Future work focuses on better features, not better sequence models |

## Artifacts to persist

For every (dataset × run × model) triple:

```
reports/{dataset}/{run_slug}/artifacts/
├── oof_predictions_{model}.parquet     # id, fold, prob, target, run_slug
├── feature_importance_{model}.csv      # only for tree models (lightgbm)
├── cv_metadata_{model}.json            # random_state, fold sizes, lib versions, git sha, wall time
└── trajectory_arrays.npz               # one per run, only created when Step 6 fires
```

And per-run reports:

```
reports/{dataset}/{run_slug}/
├── failure_prediction.md               # existing logreg report (unchanged)
├── failure_prediction_lightgbm.md      # new
├── failure_prediction_minirocket.md    # Step 6 only
└── model_comparison.md                 # paired Δ-AUC + CI vs logreg
```

Cross-dataset aggregates (Step 5z):

```
reports/_cross_dataset/
├── model_comparison_table.csv          # one row per (dataset, run, model)
├── model_comparison.md                 # rendered table with pass/fail flags
└── delta_auc_bootstrap.png             # forest plot of Δ-AUC + CI per dataset
```

## Findings log

Every step that produces a decision must write a dated entry under:

```
experiments/sh6_llm-pairwise-slod/findings/
├── 2026-04-29_step1_design.md          # this protocol, signed
├── 2026-04-29_step2_failure_models.md
├── 2026-04-29_step3_integration.md
├── 2026-04-29_step4_smoke_test.md
├── 2026-04-29_step5_lightgbm_baseline.md
└── 2026-04-29_step6_minirocket_ceiling.md   # only if Step 6 fires
```

Each entry follows the same skeleton: **hypothesis → result → decision →
follow-ups**, ≤30 lines.

## What this protocol explicitly does NOT do

To keep the comparison falsifiable, the following are out of scope for
this pass and require a fresh pre-registration if pursued:

- Per-dataset hyperparameter tuning of LightGBM or MiniRocket.
- Adding new features (TA-pack, multi-scale, FFT, cross-trajectory).
- Ensembling (logreg + LightGBM + MiniRocket).
- Threshold tuning for precision/recall trade-offs (we report AUC only).
- Including the 1.5B distill or any future runs that don't yet have a
  `trajectory_features.csv` checked in.

These are listed by name so future-you knows they were considered and
deliberately deferred, not forgotten.
