# Step 1 — Stage-5 Model-Comparison Protocol Frozen

**Date:** 2026-04-29
**Status:** completed

## Hypothesis being set up to test

The current Stage-5 baseline is L2-regularized logistic regression on
~63 aggregated trajectory features. Two architectural questions remain
open:

- **H1.** A nonlinear classifier on the same features will recover
  interaction signal that logreg cannot (e.g. *long answer × low
  monotonicity → fail*).
- **H2.** A sequence model on the raw 20-point reasoning+answer
  trajectory will recover signal beyond aggregated features.

Step 1 freezes the protocol; later steps test the hypotheses.

## What was decided and recorded

- 7 eligible runs identified (1.5B distill excluded — only 1 success in 20).
- 5+ comparison set fixed: `frontierscience/deepseek-v3.2`,
  `swe-agent/model-all`, `agenthallu/framework-all`,
  `gpqa-diamond/deepseek-v3.2`, plus one of the local-FS variants and at
  least one processbench split.
- Identical CV setup to incumbent logreg: 5-fold StratifiedKFold,
  `random_state=42`. Same feasibility shrinkage applies. Implication:
  OOF predictions across models are paired per item and per fold, which
  enables paired bootstrap on Δ-AUC.
- Frozen hyperparameters for LightGBM and MiniRocket (no per-dataset
  tuning in this pass).
- Decision rule: a model wins on a dataset iff Δ-AUC ≥ +0.03 AND
  CI-lower > 0 vs the incumbent. A model carries the comparison iff it
  wins on ≥3 of 5 datasets. **Cross-dataset average is explicitly not
  the rule** — average hides regressions on small datasets.
- Stop conditions written: if LightGBM wins on 0/5 we skip Step 6 and
  pivot to feature engineering instead.

## Risks I am accepting by freezing now

- 5-fold on the smaller datasets (gpqa-diamond minority=45,
  R1-Distill-32B minority≈27) gives folds with minority counts in the
  single digits. Bootstrap CI on those will be wide and we may end up
  in "inconclusive" territory regardless of model. This is a property
  of the data, not a flaw in the protocol.
- LightGBM with `n_estimators=300` is moderately strong but the small
  datasets risk overfitting. By freezing the hyperparameters before
  any AUC is observed, we avoid the temptation of tweaking until a
  win appears.
- MiniRocket on 20-point series is short for the 9-length kernels it
  uses internally. If MiniRocket loses, the right interpretation is
  "aggregated features are enough on T=20 series", not "sequence
  signal doesn't exist".

## Decision

Protocol is signed and frozen. Proceeding to Step 2 (`failure_models.py`
registry implementation).

## Files written

- `experiments/sh6_llm-pairwise-slod/DESIGN-stage5-models.md`
- `experiments/sh6_llm-pairwise-slod/findings/2026-04-29_step1_design.md`
  (this file)
