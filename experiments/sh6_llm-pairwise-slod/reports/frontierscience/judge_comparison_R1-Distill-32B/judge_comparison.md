# Judge Comparison on R1-Distill-32B Traces

Same locally-generated R1-Distill-32B reasoning traces, run through two
different judges for grading and pairwise SLoD: the local Qwen3-30B-A3B-
Instruct-2507 served via vLLM, vs. cloud DeepSeek-V3.2 via OpenRouter.

## Grade agreement

### All items

|                | cloud=correct | cloud=wrong |
|----------------|---------------|-------------|
| **local=correct** | 22 | 9 |
| **local=wrong**   | 5 | 107 |

- N = 143
- Raw agreement: 0.902
- Cohen κ: 0.698
- Local correct rate: 0.217; cloud correct rate: 0.189

### FINAL ANSWER items only

- N = 85, raw agreement = 0.871, κ = 0.704
- Local correct: 0.329; cloud correct: 0.318

### Open-ended (rubric) items only

- N = 58, raw agreement = 0.948, κ = 0.000
- Local correct: 0.052; cloud correct: 0.000
- κ = 0 because the cloud judge marked all rubric items wrong (passed/total ratio is degenerate); both judges *agree* the rubric items mostly fail, but kappa cannot reward agreement when one cell is empty.

## Per-chunk SLoD agreement

- Items compared: 147
- Total chunks pooled: 22734
- Mean per-item Spearman ρ: 0.495
- Median ρ: 0.533
- 25th / 75th percentile: 0.420 / 0.598
- Pooled ρ across all chunks: 0.536

See `slod_rho_histogram.png`, `slod_scatter.png`, `per_item_slod_rho.csv`.

## Failure prediction AUC (5-fold CV)

| Model | Local Qwen3 judge | Cloud DeepSeek-V3.2 | Δ AUC |
|---|---|---|---|
| length_only | 0.639 ± 0.106 | 0.663 ± 0.082 | +0.024 |
| trajectory_shape | 0.596 ± 0.134 | 0.566 ± 0.083 | -0.030 |
| trajectory_full | 0.603 ± 0.135 | 0.561 ± 0.083 | -0.043 |
| mode_stack | 0.675 ± 0.059 | 0.629 ± 0.023 | -0.046 |

## Detector verdicts

| Mode | Local Qwen3 | Cloud DeepSeek-V3.2 |
|---|---|---|
| answer_drift | inconclusive | inconclusive |
| answer_meandering | inconclusive | confirmed |
| answer_overrange | inconclusive | inconclusive |
| answer_uncommitted | inconclusive | confirmed |
| answer_volatility | inconclusive | confirmed |
| derailment_late | inconclusive | inconclusive |
| no_commitment | inconclusive | confirmed |
| premature_exit | inconclusive | confirmed |
| rambling_overlong | inconclusive | inverted |
| thrashing | inverted | inconclusive |
| truncation_abort | insufficient_data | insufficient_data |

## Headline

- Judges *agree* on grades (κ = 0.698) and *moderately agree*
  on per-chunk SLoD ranking (ρ = 0.536).
- Failure-prediction AUC barely changes between judges (~0.6 either way).
- But the **named-mode detectors flip from 0 confirmed (local Qwen3 judge)
  to 5 confirmed (cloud DeepSeek)** on the *same* traces.
  The signal that survives only at chunk-level — visible to a smarter judge
  but smoothed away by a smaller one — is what the named detectors depend on.
