# Step 2 — Model Registry Implementation

**Date:** 2026-04-29
**Status:** completed

## What was built

`src/semanticscale/sh6/failure_models.py` — small registry that wraps two
models behind a uniform interface:

```python
spec = get_spec(name)         # name in {"logreg", "lightgbm"}
result = run_oof(spec, X, y, cv, random_state)
# result: OOFResult with probabilities, fold_assignments, fold_metrics,
# confusion_matrix, feature_importance
```

Key design points:

- **Identical CV semantics for every model.** ``run_oof`` materializes
  fold assignments once and reuses the same iterator for both
  ``cross_validate`` and ``cross_val_predict``, so OOF probabilities for
  any two models are paired by item and by fold. Paired bootstrap on
  Δ-AUC becomes well-defined.
- **LogReg spec is a re-implementation of the incumbent**, not a wrapper
  around the existing function. It must be byte-identical to
  `failure_analysis._make_estimator` — verified below.
- **LightGBM spec uses pinned hyperparameters** from
  `DESIGN-stage5-models.md`. No per-dataset tuning.
- **Imputer set_output("pandas")** in the LightGBM pipeline preserves
  feature names through SimpleImputer, so LightGBM's training-time and
  predict-time feature names match. Without it, sklearn raises
  UserWarning ("X does not have valid feature names but ... was fitted
  with feature names") on every fold.
- **Feature importance is per-spec**: |coef| for logreg, gain for
  LightGBM. Both share the same `(feature, importance, signed)` schema
  so downstream tables don't branch on model name.

## Verification

Two smoke tests:

1. **Synthetic data, 80×8.** Both models train, both produce OOF
   probabilities aligned with input row order, both populate
   feature-importance frames. No warnings under `-W error`.
2. **Equivalence with incumbent.** Ran `run_oof(LogRegSpec(), ...)` on
   the actual `frontierscience/deepseek-v3.2` features (153 items,
   ~63 cols) and compared to `failure_analysis._evaluate_feature_set`:
   - AUC: **0.834769** (new) vs **0.834769** (incumbent) — exact match
   - OOF probabilities: `np.allclose(...) == True`
   - Confusion matrix: identical
   - The 0.835 in the existing per-run report agrees too.

This is the load-bearing claim: any AUC that the new pipeline reports
for `logreg` is interchangeable with the value from the existing report.
Δ-AUC vs LightGBM is therefore a pure model-capacity comparison, not a
plumbing artefact.

## Risks remaining for Step 3

- Right now `run_oof` always retrains on every fold to produce
  `feature_importance`. For LightGBM that's fine (300 trees × 5 folds is
  seconds), but it costs a small amount of wall time vs. `cross_validate`
  alone. Worth keeping until we know we don't need importances at scale.
- `LightGBMSpec` imports lightgbm lazily at `build_estimator` time, so a
  machine without lightgbm can still use the logreg path. Step 3 should
  not require lightgbm at module-import time.

## Decision

Registry is sound and equivalence-tested. Proceeding to Step 3
(integrate into `failure_analysis.evaluate_prediction_models`, persist
OOF artifacts to disk).

## Files written

- `src/semanticscale/sh6/failure_models.py`
- `experiments/sh6_llm-pairwise-slod/findings/2026-04-29_step2_failure_models.md`
