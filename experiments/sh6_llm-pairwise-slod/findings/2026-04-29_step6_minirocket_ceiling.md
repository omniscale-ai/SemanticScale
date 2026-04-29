# Step 6 — MiniRocket Ceiling: NOT EXECUTED

**Date:** 2026-04-29
**Status:** intentionally skipped per protocol

## Why this step did not run

The pre-registered protocol in `DESIGN-stage5-models.md` includes the
following stop condition:

> If LightGBM wins on 0/5: skip Step 6 (sequence model unlikely to help
> on aggregated input either); pivot to feature engineering (TA pack,
> multi-scale, cross-trajectory).

Step 5 produced exactly that outcome: 0 wins / 2 regressions on the 5
protocol datasets, paired bootstrap CI on Δ-AUC consistent with noise
around 0. The protocol's stop condition fires.

## Why honoring this stop condition matters

Skipping Step 6 is the *correct* protocol-following move, not a
shortcut. If we ran MiniRocket anyway:

- A win would be hard to interpret. We'd have to explain why a sequence
  model recovered signal that LightGBM missed even though both should
  be able to express any monotone transformation of the same features.
- A loss would just confirm what we already infer from Step 5 — the
  bottleneck is feature representation, not classifier capacity.
- Either way we'd be running an expensive comparison whose result was
  already implied by the pre-registered logic.

The protocol was written to avoid exactly this kind of motivated
post-hoc digging. Honoring the stop condition is what gives Step 5's
finding its credibility.

## What replaces Step 6

A new track: **Stage-5 feature engineering**, with a fresh
pre-registration document. Concretely:

- TA-pack on per-item-centered trajectory (RSI-5, ATR-5, Bollinger %B,
  drawdown duration, Hurst, FFT-band-power)
- Multi-scale aggregation (first/middle/last third)
- Cross-trajectory features (lead-lag, rolling correlation between
  reasoning and answer channels)
- Sanity test on agenthallu and processbench/gsm8k: do mean
  trajectories of correct vs wrong actually differ on those datasets?
  If they don't, SLoD is the wrong axis for those failure types and
  feature engineering won't help either.

That track is **not** opened automatically; it requires a new DESIGN
document because adding ~30 features and reusing the same Δ-AUC
criterion on the same datasets would re-introduce the post-hoc
selection problem already flagged for the answer-side detectors on
FrontierScience.

## When Step 6 would still be worth running

If the feature-engineering pass produces a meaningful AUC bump on the
same 5 datasets, MiniRocket on raw trajectories becomes a useful upper
bound: "are we close to the sequence-level ceiling, or is there more
signal hiding in the order beyond what aggregated features capture?".
Run it then, with a fresh pre-registration that anchors Δ-AUC vs the
new feature set rather than vs logreg.

## Files written

- `experiments/sh6_llm-pairwise-slod/findings/2026-04-29_step6_minirocket_ceiling.md`
  (this file — records the deliberate skip)
