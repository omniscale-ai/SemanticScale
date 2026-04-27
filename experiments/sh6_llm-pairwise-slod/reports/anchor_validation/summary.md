# SH6 Phase A — anchor-based absolute SLoD validation

## Anchor self-consistency (Step 1)

- n_anchors: 15
- n_pairs: 105
- n_ties: 13
- **Spearman ρ(μ, tier) = 0.945** (p = 1.13e-07)

| anchor | tier | μ |
|---|---|---|
| macro_01 | 3 | 43.01 |
| macro_04 | 3 | 38.29 |
| macro_05 | 3 | 38.15 |
| macro_03 | 3 | 37.16 |
| macro_02 | 3 | 36.21 |
| meso_02 | 2 | 27.65 |
| meso_05 | 2 | 26.84 |
| meso_04 | 2 | 24.73 |
| meso_01 | 2 | 23.13 |
| meso_03 | 2 | 22.61 |
| micro_03 | 1 | 14.08 |
| micro_02 | 1 | 13.02 |
| micro_04 | 1 | 11.26 |
| micro_01 | 1 | 10.49 |
| micro_05 | 1 | 8.35 |

## Per-dataset absolute SLoD (Step 2)

- Pairs compared: 10215  (681 chunks × 15 anchors)
- Traces sampled per dataset: 30
- Max chunks per trace: 6

| dataset | n_chunks | n_traces | mu_mean | mu_std | mu_min | mu_max | per_trace_std_median | per_trace_std_mean | per_trace_range_median | per_trace_range_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| agenthallu | 173.000 | 30.000 | 25.585 | 7.357 | 11.346 | 42.200 | 5.354 | 5.548 | 14.841 | 15.871 |
| frontierscience | 180.000 | 30.000 | 26.366 | 8.420 | 7.631 | 43.207 | 5.589 | 5.756 | 15.875 | 16.334 |
| processbench | 148.000 | 30.000 | 21.551 | 5.346 | 12.470 | 38.306 | 3.874 | 4.082 | 10.889 | 11.695 |
| swe-agent-trajectories | 180.000 | 15.000 | 23.297 | 5.203 | 10.816 | 37.225 | 4.129 | 3.938 | 13.270 | 13.130 |

Note: swe-agent-trajectories has only 15 unique trace IDs in the sample because many of its 30 sampled traces had no chunks attached (empty reasoning_chunks).

## Interpretation

Hypothesis was: processbench/agenthallu fail SLoD-based failure prediction (ROC-AUC ~0.50) because their traces lack absolute variance on the macro↔micro axis, while swe-agent/frontierscience succeed because their traces span levels.

**Result: hypothesis NOT supported by per-trace std.**

Ranking by per-trace std median (ascending):

| dataset | per_trace_std_median | failure ROC-AUC | predictive? |
|---|---|---|---|
| processbench | 3.873 | 0.504 | NO |
| swe-agent | 4.129 | 0.865 | YES |
| agenthallu | 5.354 | 0.505 | NO |
| frontierscience | 5.589 | 0.884 | YES |

swe-agent (predictive) and processbench (not predictive) have **almost identical** per-trace absolute variance. agenthallu (not predictive) has **larger** absolute variance than swe-agent (predictive). Absolute variance is therefore not the discriminator for whether SLoD-based failure prediction works.

## What this means

The signal that makes SH6 work on swe-agent and frontierscience must be in trajectory *shape patterns* (specific arrangements of high/low-abstraction chunks tied to errors), not in the gross magnitude of variance. The lift from anchor-based absolute features on processbench/agenthallu is therefore likely to be modest — Phase B as originally designed (mean, std, range, anchor_coverage) is unlikely to cross the 0.60 ROC-AUC bar.

Possible next directions:
1. **Absolute level (mu_mean)** still differs across datasets — agenthallu (25.6) and    frontierscience (26.4) sit higher on the axis than processbench (21.6) and swe-agent    (23.3). This is a per-dataset effect and won't help within-dataset failure prediction,    but may matter for cross-dataset transfer.
2. **Per-step expected level**: failures may correspond to chunks at the *wrong* abstraction    level for their position in the trace (e.g., agent staying meta-strategic when it should    commit to a tool call). This requires a domain prior, not just absolute scoring.
3. **Failure modes on processbench/agenthallu may simply not be SLoD-related** — they may    stem from factual errors (math) or hallucinated tool calls, which the abstraction axis    cannot detect. The honest conclusion may be that SLoD has a domain of validity and these    datasets fall outside it.

## Recommendation

Per the Phase A gate in the plan: **stop and reassess** before investing in Phase B. The specific fix proposed (anchor-based absolute features) is unlikely to lift processbench / agenthallu prediction to a useful level, since absolute variance does not separate the predictive from non-predictive datasets.

