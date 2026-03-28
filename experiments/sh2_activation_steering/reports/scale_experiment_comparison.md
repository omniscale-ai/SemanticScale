# Scale Experiment Comparison: Mistral-7B vs Qwen2.5-14B

**Date:** 2026-03-16
**Experiment:** SH2 Activation Steering Along the SLoD Axis — Scale Comparison

## Hardware Note

L40S GPU with 44.4 GiB PyTorch-visible VRAM. Qwen2.5-72B and Qwen2.5-32B both failed to load due to OOM during weight loading (transformers 5.x loads shards in fp16/bf16 before quantizing, peaking at ~44 GB). Qwen2.5-14B-Instruct in 4-bit used ~27 GB and loaded successfully.

## Model Configuration

| Parameter | Mistral-7B | Qwen2.5-14B |
|---|---|---|
| Model | mistralai/Mistral-7B-Instruct-v0.3 | Qwen/Qwen2.5-14B-Instruct |
| Parameters | 7B | 14B |
| Quantization | bfloat16 (full) | 4-bit NF4 |
| Num layers | 32 | 48 |
| Hidden size | 4096 | 5120 (reported) |
| Candidate layers | [8, 16, 24] | [12, 24, 36] |

## Key Metrics Comparison

| Metric              | Mistral-7B              | Qwen2.5-14B    | Threshold   |
| ------------------- | ----------------------- | -------------- | ----------- |
| **Verdict**         | NOT CONFIRMED           | NOT CONFIRMED  | —           |
| H1 micro mean_delta | 0.0280                  | 0.0201         | —           |
| H1 micro Cohen's d  | 0.043                   | 0.020          | **d > 0.5** |
| H1 micro p-value    | 0.341                   | 0.658          | p < 0.05    |
| H2 n_significant    | 0/4                     | 1/4            | ≥ 2/4       |
| H3 factuality drop  | 0.0001                  | 0.0012         | < 0.05      |
| Selected layer      | 24 (abs-bug, wrong dir) | 36 (75% depth) | —           |
| Selected alpha      | 1.0                     | 4.0            | —           |

## Layer Shifts on Validation Set

**Mistral-7B** (with abs() bug — selected layer 24 which had highest abs but wrong direction):
- Layer 8: +0.181 (correct direction, but not selected due to bug)
- Layer 16: (not recorded)
- Layer 24: -0.291 (selected by abs() — wrong direction)

**Qwen2.5-14B** (with fixed signed selection):
- Layer 12: -0.231 (anti-micro direction)
- Layer 24: -0.258 (anti-micro direction)
- Layer 36: -0.061 (selected — least negative / closest to correct)

## Alpha Sweep (Qwen2.5-14B, Layer 36)

| Alpha | Mean SLoD shift |
|---|---|
| 0.5 | +0.022 |
| 1.0 | -0.061 |
| 2.0 | +0.034 |
| 4.0 | -0.071 |

The alpha sweep shows inconsistent signs (no monotonic relationship), indicating no meaningful signal.

## Qualitative Examples

Three examples comparing baseline vs micro-steered outputs (alpha=4.0, layer 36):

**Example 1** — Q: "How did they constrain training using the parameters?"
- Baseline: "They constrained training by freezing certain parameters of the parent model, allowing only specific parts to be fine-tuned..."
- Steered: "They constrained training by freezing certain parameters of the parent model, allowing only specific parts to be fine-tuned..." (identical start)

**Assessment:** Negligible visible difference. The steering vector at alpha=4.0 produces outputs that are nearly indistinguishable from baseline, consistent with the near-zero Cohen's d (d=0.020).

## Did Scale Help?

**No.** Scaling from 7B to 14B parameters (2x scale) did not improve the steering effectiveness:
- H1 Cohen's d: 0.043 (Mistral) → 0.020 (Qwen2.5-14B) — *worse*, not better
- The layer shifts are uniformly negative for Qwen2.5-14B, suggesting the micro steering vector in Qwen's representation space actually pushes outputs in the macro direction in SciBERT space
- The layered architecture (48 vs 32 layers) and cross-architecture representation mismatch both persist

## Root Cause Analysis

The core issue identified by the Analyst (iter 5) is confirmed by this scale experiment:

1. **Cross-architecture representation mismatch**: The SLoD direction computed from the generative model (Qwen2.5-14B, 5120-dim residual stream) does not correspond to the SLoD direction in SciBERT embedding space (768-dim encoder). This mismatch is not reduced by scaling.

2. **All layer shifts negative**: For Qwen2.5-14B, all three candidate layers show negative shifts, meaning the "micro" steering vector drives outputs toward the macro direction in SciBERT space. This is the inverse of the expected effect.

3. **Steering vector norms similar**: All layers have similar raw norms (~116-118), suggesting no layer is clearly dominant for encoding the SLoD distinction.

## Conclusion

The scale hypothesis (larger models have more linearly organized representations, making steering more effective) is **not supported** by this experiment. The NOT CONFIRMED verdict holds at 14B scale with the same statistical margins as 7B.

Recommended future directions:
- Fine-tuning (SFT) with SLoD-labeled data to explicitly align internal representations
- Prompt-based control (system prompts instructing abstraction level)
- Multi-model steering vector computed from in-distribution encoder-decoder models
