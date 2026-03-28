# SH2: Activation Steering Along the SLoD Axis — Conclusive Report

**Date:** 2026-03-16
**Analyst:** ANALYST agent
**Verdict: NOT CONFIRMED**
**Decision: Conclude (no iteration warranted)**

---

## 1. Summary

SH2 tested whether the SLoD (Semantic Level of Detail) abstraction direction — computed as a
difference-of-means steering vector in Mistral-7B's hidden states — could shift the abstraction
level of generated scientific QA answers without weight updates. The experiment was NOT CONFIRMED:
micro-direction steering produced a mean SLoD shift of +0.028 (Cohen's d = 0.043) on 500 QASPER
questions, far below the required d ≥ 0.5, p < 0.05 threshold.

---

## 2. Quantitative Results

### Primary Hypothesis H1 (SLoD Shift) — FAILED

| Metric | Micro direction | Macro direction |
|---|---|---|
| Mean Δ SLoD | +0.0280 | -0.0260 |
| Std Δ SLoD | 0.6575 | 0.5689 |
| p-value | 0.341 | 0.308 |
| Cohen's d | +0.043 | -0.046 |
| Required threshold | d ≥ 0.5, p < 0.05 | — |
| **Passed** | **No** | **—** |

The delta is 32× below the threshold mean shift (required ≈ 0.90 SLoD units for d = 0.5 given
baseline std = 1.804; achieved = 0.028).

### Secondary Hypothesis H2 (Surface Metrics) — FAILED

| Metric | Δ | p-value | Significant |
|---|---|---|---|
| Entity density | -0.0001 | 0.852 | No |
| Citation density | -0.000019 | 0.748 | No |
| Numeric density | +0.0010 | 0.155 | No |
| Mean sentence length | -0.144 | 0.351 | No |
| **Significant (of 4)** | **0** | — | **No** |

### Tertiary Hypothesis H3 (Factuality Preservation) — PASSED

| Condition | Mean Token-F1 |
|---|---|
| Baseline | 0.2712 |
| Steered (micro) | 0.2711 |
| Drop | 0.0001 |
| Threshold | ≤ 0.05 |
| **Passed** | **Yes** |

Steering had no detectable effect on factuality (trivially, because it had near-zero effect on
output content).

### Experimental Parameters

| Parameter | Value |
|---|---|
| Model | Mistral-7B |
| Selected layer | 24 (75% depth) |
| Selected alpha | 1.0 |
| N questions | 500 |
| Steering vector source | Difference-of-means, Mistral-7B hidden states, train split only |
| Evaluation metric | SciBERT projection onto centroid SLoD axis (d = 2.65) |

---

## 3. Root Cause Diagnosis

### 3.1 The Layer Selection Anomaly

The most prominent anomaly is that layer selection chose layer 24 with a **negative** validation
shift of -0.291, meaning micro-direction steering at that layer pushed SciBERT-measured SLoD
**downward** (toward macro). Layer 24 was selected because `select_best_layer` picks the layer
with the **largest absolute shift**, not the largest positive (micro-ward) shift. The correct
layer to use for micro-direction steering was layer 16 (validation shift = +0.181, the only
positive shift among all candidates).

| Candidate layer | Validation shift | Direction |
|---|---|---|
| 8 | -0.169 | Wrong (macro-ward) |
| 16 | +0.181 | Correct (micro-ward) |
| 24 | -0.291 | Wrong (macro-ward), **selected** |

**This is a methodological defect in the layer selection logic**: it optimized for steering
magnitude without constraining direction.

### 3.2 Would Layer 16 Have Changed the Verdict?

The validation shift for layer 16 was +0.181 SLoD units, corresponding to an effect size of
d ≈ 0.10 on the 50-question validation sample. Extrapolated to the full 500-question evaluation,
this would still be approximately 5× below the required d ≥ 0.5 threshold. Re-running stages
E, F, G with layer 16 would not change the NOT CONFIRMED verdict.

Additionally, the alpha sweep for layer 24 showed that higher alpha values did **not** increase
the effect monotonically:

| Alpha | Validation shift (layer 24) |
|---|---|
| 0.5 | -0.210 |
| 1.0 | -0.291 |
| 2.0 | -0.191 |
| 4.0 | -0.047 |

Shifts at all tested alphas (0.5–4.0) are within the expected noise range for n = 50
(σ_noise ≈ 1.8 / √50 ≈ 0.255), consistent with a near-zero true effect. The decreasing
magnitude from alpha = 1.0 to 4.0 suggests model degradation rather than saturation of a
real effect.

### 3.3 Evidence for Each Causal Explanation

**Explanation 1: Mistral-7B's SLoD axis does not generalize to output SLoD.**

*Evidence for:*
- All three candidate layers produced near-zero or anti-directional shifts
- Micro and macro directions produced mirror-small effects (+0.028 and -0.026), consistent with
  noise rather than genuine opposite-direction steering
- Token-F1 was unchanged, indicating the model's output vocabulary distribution was barely
  perturbed

*Evidence against:*
- Layer 16 showed a small but directionally correct shift (+0.181) on 50 questions, suggesting
  some minimal alignment exists

**Explanation 2: Alpha was too small.**

*Evidence for:*
- The tested range (0.5–4.0) is at the lower end compared to published activation addition
  experiments (Turner et al. use alpha up to 20+)

*Evidence against:*
- The alpha sweep shows diminishing returns from alpha = 1.0 to 4.0 at layer 24
- For layer 16, even if alpha = 20 linearly extrapolated the validation shift
  (0.181 × 20 = 3.62 SLoD units, d ≈ 2.0), such high alpha values typically produce degenerate
  incoherent text and cannot be trusted for QA
- The near-zero factuality drop (H3 passed easily) signals the steering was not strong enough
  to alter content at all, but this is intrinsic to the mechanism, not just alpha magnitude

**Explanation 3: Embedding space mismatch.**

The steering vector lives in Mistral-7B's 4096-dimensional residual stream; the evaluation axis
lives in SciBERT's 768-dimensional encoder space. These are fundamentally different spaces with
no guaranteed alignment. The SLoD axis in SciBERT captures surface-level lexical and syntactic
patterns associated with abstract vs. specific scientific writing. The Mistral-7B SLoD direction
may encode orthogonal information (e.g., writing register, argument structure) that does not
manifest as a SciBERT-measurable SLoD shift.

*Evidence for:*
- SciBERT SLoD axis has d = 2.65 on held-out spans — it is a strong classifier of SLoD in
  encoder space
- Baseline QA answers have mean SLoD = 2.90, sitting between macro (-0.53) and micro (4.71)
  centroids — the evaluation axis does discriminate in this range
- Yet steering produced essentially zero shift — the perturbation in Mistral space did not
  translate to a detectable shift in SciBERT space

### 3.4 Assessment: Iterate or Conclude?

**Decision: Conclude NOT CONFIRMED.**

Three lines of evidence argue against iteration:

1. **The magnitude gap is too large.** The best observed directionally-correct shift (layer 16,
   +0.181) is 5× below the required effect size, even before accounting for the difference between
   n=50 validation noise and n=500 evaluation reliability. Closing this gap with alpha tuning
   alone is implausible without generating incoherent text.

2. **The alpha sweep already covers plausible values.** The sweep from 0.5 to 4.0 shows
   diminishing returns. Published activation addition work achieves strong behavioral changes in
   sentiment or refusal tasks at alpha ≤ 5; SLoD-relevant content changes may require
   fundamentally different mechanisms.

3. **The null result is scientifically informative.** The finding that Mistral-7B's internal
   SLoD direction (micro - macro centroids in activation space) does not generalize to
   SciBERT-measured output SLoD is a meaningful negative result. It constrains the scope of
   Representation Engineering for fine-grained semantic properties like abstraction level.

---

## 4. Scientific Interpretation

### What This Experiment Tested

SH2 tested the strongest form of the SLoD control hypothesis: that a single difference-of-means
vector, injected into one layer of a causal LM's residual stream, can shift the abstraction level
of generated text along the SLoD axis as measured externally by SciBERT.

### What Was Found

The causal model's internal SLoD direction does not produce reliable SLoD shifts in output text
as measured by SciBERT projection. The effect size (d = 0.043) is approximately 12× below the
threshold and 60× below the SLoD axis's own discriminative power (d = 2.65 for span classification).

### Why This May Be

1. **Cross-architecture misalignment:** The SLoD axis may be encoded differently in causal LMs
   (auto-regressive, trained on diverse internet text) vs. scientific domain encoder models
   (SciBERT, fine-tuned on academic papers). The "micro direction" in Mistral-7B's hidden states
   may capture writing-style features that do not map to scientific detail level.

2. **Token-level vs. discourse-level:** Activation steering works at the token level (a constant
   vector is added at every generated token). SLoD is a discourse-level property of full
   paragraphs or answers. The mapping from token-level activation perturbation to discourse-level
   property change may require a much larger, more sustained, or coherent perturbation than a
   single fixed vector can provide.

3. **Representation Engineering limitations for semantic content:** Published successes of
   Representation Engineering are predominantly in sentiment, refusal, or role-play tasks —
   properties that are closely tied to individual tokens or short phrases. Scientific abstraction
   level is a structurally different property: it emerges from choices about what content to
   include (citations, numbers, specific methodology) rather than lexical sentiment. The
   residual stream perturbation may not influence these higher-order content selection decisions.

### Contrast with SH5d

SH5d (CONFIRMED, ρ = +0.219) showed that the SLoD axis computed in SciBERT space predicts QA
quality, validating that SLoD is a meaningful property of model-generated answers. SH2 shows that
this property cannot be steered via residual stream injection in the generative model itself.
These findings are compatible: SLoD is real and predictive, but not accessible for direct
intervention via this mechanism.

---

## 5. Comparison with Prior SH Results

| SH | Status | Key Metric | Relevance to SH2 |
|---|---|---|---|
| SH0 | CONFIRMED | Document structure → macro/meso/micro labels | Provided training data for SV computation |
| SH1 | CONFIRMED | SciBERT probe macro-F1 = 0.72 | SciBERT SLoD axis is valid (d = 2.65) |
| SH5d | STRONG | SLoD axis ρ = +0.219 | SLoD measurable in LLM output space |
| SH2 | NOT CONFIRMED | d = 0.043 | SLoD not steerable via activation injection |

The SH2 null result does not invalidate SH1 or SH5d. It adds a boundary condition: SLoD is
representable in encoder space and predictive of quality, but not directly controllable via
decoder residual stream injection.

---

## 6. Recommendations for Future Work

### Near-term

1. **Supervised fine-tuning (SFT) for SLoD control:** Rather than unsupervised steering,
   fine-tune the generative model with SLoD-labeled answer pairs (macro vs. micro answers to
   the same question). This directly optimizes the output distribution rather than perturbing
   the residual stream.

2. **Prompt-based SLoD control (SH2a):** Use few-shot exemplars at different SLoD levels as
   a prompt prefix. This is a lower-variance approach and directly conditions the generation
   at the input level.

3. **Layer sweep with signed selection:** If activation steering is revisited, the layer
   selection criterion must use signed shift (selecting the layer with largest **positive**
   shift for micro-ward steering), not absolute shift. Layer 16 is the better candidate.

4. **Higher alpha with degeneration tracking:** A sweep of alpha ∈ [8, 16, 32] at layer 16
   with explicit degeneration detection (e.g., perplexity, repetition rate) would empirically
   determine whether higher alpha produces SLoD shift before quality collapse.

### Longer-term

5. **Cross-architecture steering:** Encode the SLoD direction in a shared semantic space
   (e.g., via a learned adapter) and steer via that shared representation, rather than relying
   on architecture-specific residual stream coordinates.

6. **SLoD as a reward signal:** Incorporate SciBERT SLoD projection as a reward in RLHF or
   DPO training to optimize generation toward target abstraction levels.

---

## 7. Methodological Notes for Archival

- **Layer selection bug:** `select_best_layer` in `src/steering.py` (line 201) selects by
  `abs(shift)` rather than signed shift. This caused selection of layer 24 (anti-correlated)
  over layer 16 (correctly directional). Future experiments should use
  `max(layers, key=lambda l: layer_shifts[l])` for micro-ward steering.

- **Alpha sweep on wrong layer:** The alpha sweep in Stage E was conducted after layer selection,
  using the already-selected (wrong) layer 24. All swept alpha values produced negative shifts,
  consistent with using an anti-correlated layer rather than an alpha saturation effect.

- **Small validation sample:** Layer and alpha selection on n = 50 questions introduced high
  noise (σ_noise ≈ 0.25 SLoD units). A validation set of n ≥ 200 would have given more reliable
  signal.

---

*Analyst: ANALYST agent — 2026-03-16*
