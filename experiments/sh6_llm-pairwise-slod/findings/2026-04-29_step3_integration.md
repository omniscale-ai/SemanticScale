# Step 3 — Integration & OOF Artifact Persistence

**Date:** 2026-04-29
**Status:** completed

## What was built

`src/semanticscale/sh6/failure_analysis_runner.py` — separate driver that
loads one run's `trajectory_features.csv`, executes the requested models
through the registry, and writes per-model artifacts to
`reports/{dataset}/{run_slug}/artifacts/`:

```
artifacts/
├── oof_predictions_<model>.parquet     # id, row_index, fold, target, prob
├── feature_importance_<model>.csv      # feature, importance, signed
└── cv_metadata_<model>.json            # AUC, fold sizes, git SHA, lib versions, wall time
```

The driver was kept **separate from `failure_analysis.py`** by design:
the existing pipeline still controls the canonical
`failure_prediction.md` report, and Step 3 only adds new artifacts.
That keeps the comparison reversible — the new track can be deleted
without touching the incumbent.

Shared logic that the driver reuses without duplication:

- `choose_feature_sets` — same `length_only`/`trajectory_shape`/`trajectory_full` split
- `prediction_feasibility` — same fold shrinkage when minority class is small
- `target_stats` — same class-count reporting
- Auto target resolution: `final_answer_correct` first, then `is_correct`

This means logreg via `run_models_on_run` is byte-identical to logreg
via the incumbent — **no path divergence**.

## Verification on real data

Run: `frontierscience/deepseek/deepseek-v3.2_reasoning-auto` (153 items, 63 features)

| Model    | AUC mean ± std        | Wall time |
|----------|----------------------|-----------|
| logreg   | **0.8348 ± 0.0847**  | 0.38 s    |
| lightgbm | **0.8735 ± 0.0601**  | 1.67 s    |

- logreg AUC matches existing report (0.835) and Step 2 byte-for-byte (0.834769).
- LightGBM point estimate is **+0.0387** above logreg — above the +0.03
  decision threshold from the protocol. **But this is one dataset and a
  point estimate only**; the win is not declared until Step 5 computes
  the paired bootstrap CI on Δ-AUC and tallies the ≥3-of-5 datasets rule.

All six artifacts written (~32 KB total per run) and parquet files load
back cleanly.

## Implementation choices worth recording

- **Pandas-output imputer** for LightGBM. Without `set_output("pandas")`
  on `SimpleImputer`, sklearn drops feature names between fit and
  predict, and LightGBM emits a UserWarning on every fold. Setting
  pandas output is the cleanest fix and matches the incumbent's
  feature-naming behavior.
- **`id` plus `row_index` in the OOF parquet.** Step 5 needs to align
  paired predictions across models; using both lets us verify
  alignment regardless of how the cross-dataset aggregator joins.
- **Lazy lightgbm import** inside `LightGBMSpec.build_estimator`.
  Machines that only need the logreg path don't have to install
  lightgbm.
- **Git SHA and library versions** captured in metadata. The protocol
  pins hyperparameters; metadata pins everything else that matters for
  reproducibility.

## Risks remaining

- LightGBM at default `n_estimators=300` can overfit small folds. The
  smoke test on 153 items already shows lower variance (std 0.060 vs
  0.085) — encouraging but not yet a win until paired CI is computed.
- The `feature_set="trajectory_full"` default was kept fixed for this
  comparison, matching the incumbent baseline. Sweeping the
  `length_only` / `trajectory_shape` cuts is a follow-up, not part of
  this protocol.

## Decision

Driver is sound, equivalence with incumbent is preserved, artifacts are
on disk and load cleanly. Proceeding to Step 4 (orchestration script
`05b_lightgbm_comparison.py` and end-to-end smoke run).

## Files written

- `src/semanticscale/sh6/failure_analysis_runner.py`
- `experiments/sh6_llm-pairwise-slod/findings/2026-04-29_step3_integration.md`
