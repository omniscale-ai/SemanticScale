# Step 5 — LightGBM vs Logreg on Stage-5 Features

**Date:** 2026-04-29
**Status:** completed
**Protocol:** `DESIGN-stage5-models.md` (signed 2026-04-29)

## Hypothesis under test

A nonlinear classifier on the same Stage-5 trajectory features should
recover interaction signal that L2 logistic regression cannot, and lift
ROC-AUC by ≥ +0.03 with CI-lower > 0 on at least 3 of the 5
pre-registered datasets.

## What was run

10 eligible runs (1.5B distill skipped — too few examples per class).
Both models trained under identical 5-fold StratifiedKFold,
`random_state=42`, `class_weight="balanced"`. OOF probabilities saved
per item per model. Δ-AUC computed on pooled OOF predictions, with a
1000-draw paired percentile bootstrap CI on the difference.

## Result

```
Protocol runs evaluated: 5 / 5
Challenger wins         : 0
Significant regressions : 2  (gpqa-diamond, processbench/gsm8k)
≥3-of-5 carry rule      : FAIL
```

Per-run table (full table in
`reports/_cross_dataset/model_comparison_logreg_vs_lightgbm.md`):

| Run                                          | AUC logreg | AUC lightgbm | Δ-AUC  | CI low  | CI high | Verdict      | In set |
|----------------------------------------------|-----------:|-------------:|-------:|--------:|--------:|--------------|:------:|
| frontierscience/deepseek-v3.2                |     0.837  |       0.863  | +0.026 | -0.025  |  0.078  | inconclusive | ✓      |
| swe-agent-trajectories/model-all             |     0.858  |       0.863  | +0.004 |  0.000  |  0.009  | inconclusive | ✓      |
| agenthallu/framework-all                     |     0.503  |       0.515  | +0.012 | -0.039  |  0.065  | inconclusive | ✓      |
| gpqa-diamond/deepseek-v3.2                   |     0.696  |       0.663  | -0.033 | -0.115  |  0.053  | regress      | ✓      |
| processbench/gsm8k                           |     0.507  |       0.469  | -0.038 | -0.099  |  0.024  | regress      | ✓      |
| frontierscience/R1-Distill-32B-cloudjudge    |     0.566  |       0.522  | -0.044 | -0.138  |  0.058  | regress (oop)| ✗      |
| frontierscience/R1-Distill-32B-local         |     0.610  |       0.647  | +0.037 | -0.081  |  0.158  | inconclusive | ✗      |
| processbench/olympiadbench                   |     0.474  |       0.522  | +0.048 | -0.019  |  0.116  | inconclusive | ✗      |
| processbench/omnimath                        |     0.632  |       0.555  | -0.077 | -0.134  | -0.021  | regress      | ✗      |
| swe-agent-trajectories/model-all_steps-50    |     0.913  |       0.913  | +0.000 |  0.000  |  0.000  | inconclusive | ✗      |

Note: AUCs here are computed on pooled OOF predictions, which differs
slightly from the per-fold AUC means in `failure_prediction.md`. Both
quantities are valid; the pooled version is the correct one for the
paired bootstrap.

## Diagnosis

- **Random scatter around zero.** Across 10 runs the Δ-AUC distribution
  has 5 positive and 4 negative points, none significant in either
  direction except `processbench/omnimath` (regression, out of protocol
  set). This is what "no interaction signal" looks like.
- **Largest CIs are on the smallest runs.** R1-Distill variants
  (~145–147 items) and gpqa-diamond (192 items, very imbalanced
  77% positive class) each have CI widths of ±0.10. Even where the
  point estimate looks promising (e.g. R1-Distill-32B-local +0.037,
  olympiadbench +0.048), the CI straddles 0 by a wide margin.
- **The two strongest signals are roughly tied across models.**
  swe-agent (AUC ≈ 0.86) and swe-agent_steps-50 (AUC ≈ 0.91): both
  models are essentially saturated on these. The features carry the
  signal; the classifier choice barely matters.
- **The two weakest signals are tied at chance.** agenthallu and
  processbench/gsm8k both come in at AUC ≈ 0.50 for both models. The
  problem on these datasets is that **the trajectory features
  themselves do not separate success from failure** — switching the
  classifier cannot fix that.

## Decision

Per the stop conditions in `DESIGN-stage5-models.md` (LightGBM wins on
0/5 → skip Step 6, pivot to feature engineering): **LightGBM does not
carry the comparison. Step 6 (MiniRocket) is intentionally skipped.**
Reason for skipping: MiniRocket on raw 20-point trajectories is unlikely
to recover sequence signal that aggregated features missed when the
aggregated features themselves carry the same information that a
nonlinear model couldn't extract more of. The bottleneck is the feature
representation, not the model.

The right next track is **feature engineering**, in this order of
expected payoff:

1. **TA-pack on the per-item-centered trajectory.** RSI-5 / ATR-5 /
   Bollinger %B / drawdown-duration / Hurst exponent / FFT-band-power.
   These add ~10 interpretable features that distinguish "smooth trend"
   from "shaky trend with the same range" — something the current
   commitment/thrashing pair conflates.
2. **Multi-scale aggregations.** The same feature set computed on the
   first/middle/last third in addition to the full trajectory.
   Triples the feature count cheaply and is a known good fix for
   short series where global summaries flatten everything.
3. **Cross-trajectory features.** Lead-lag correlation, cointegration
   residuals, rolling correlation between reasoning and answer
   channels — the place where "answer disconnected from reasoning" is
   most likely to live.
4. **For agenthallu and processbench/gsm8k specifically:** verify that
   mean trajectories of correct vs wrong items differ at all on these
   datasets. If they don't, no feature engineering will help and the
   issue is that SLoD is not the right axis for those failure modes
   (hallucination ≠ abstraction shift).

The TA-pack track requires a fresh pre-registration document because
post-hoc feature selection on the same Δ-AUC criterion would re-introduce
the same circularity that biased the answer-side detector verdicts on
FrontierScience.

## Risks of misreading this result

- **"LightGBM doesn't help" ≠ "trees are bad."** The hyperparameters
  were frozen; they were not tuned. Per-dataset tuning could plausibly
  flip 1–2 inconclusive verdicts. But the protocol was deliberately
  written without per-dataset tuning, because tuning until a win
  appears is exactly the failure mode we are guarding against. If a
  follow-up wants to do tuning, it needs a held-out validation set
  that does not influence the comparison datasets.
- **0/5 wins on the carry rule is not the same as "no signal anywhere."**
  It just means the signal is not big enough on this set of features
  to pass the +0.03 / CI > 0 / 3-of-5 bar. Per-feature univariate AUC
  rankings (already in the existing reports) suggest several features
  carry useful signal individually; LightGBM apparently isn't finding
  *additional* multivariate structure beyond what logreg already linearly
  combines.

## Files written

- `experiments/sh6_llm-pairwise-slod/scripts/05z_aggregate_models.py`
- `experiments/sh6_llm-pairwise-slod/reports/_cross_dataset/model_comparison_logreg_vs_lightgbm.{csv,md,json}`
- `experiments/sh6_llm-pairwise-slod/reports/_cross_dataset/delta_auc_logreg_vs_lightgbm.png`
- per-run `artifacts/oof_predictions_*.parquet`,
  `feature_importance_*.csv`, `cv_metadata_*.json` for all 9 eligible runs
- per-run `logs/05b_lightgbm_*.log` for all runs
- `experiments/sh6_llm-pairwise-slod/findings/2026-04-29_step5_lightgbm_baseline.md`
  (this file)

## Followups (not part of this comparison)

- Pre-register a TA-pack track for Stage-5 feature extension.
- For datasets where mean trajectories of correct/wrong are
  indistinguishable, write a separate note rather than continuing to
  benchmark classifiers on noise.
- Reconsider per-dataset tuning *only after* the feature-engineering
  pass — model improvements are easier to interpret on top of better
  features.
