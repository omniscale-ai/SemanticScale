# SLoD in LLM Contexts — Research Roadmap

> Initial plan generated via multi-model consensus, 2026-03-08
> Updated 2026-03-18 with SH2 activation steering results (CONFIRMED on summarization)

## Unified Core Thesis

> **Frozen LLM representations encode a continuous Semantic Level of Detail (SLoD) axis that is linearly decodable without retraining. Exploiting this axis — to match retrieval granularity to query abstraction, to detect output drift, and to steer generation — measurably improves scientific QA attribution and TKH extraction quality. Continuous embedding-space dynamics in reasoning chains provide supporting behavioral evidence.**

Three contribution layers: **mechanistic** (SLoD is in the embedding space) + **systems** (routing improves RAG) + **behavioral** (embedding dynamics reveal reasoning quality).

---

## Execution Status

> **All planned experiments complete.** SH0–SH5d plus SH2 activation steering have been executed.

| Sub-Hypothesis | Status | Verdict | Key Result | Code |
|---|---|---|---|---|
| **SH0** | DONE | **CONFIRMED** | Document structure provides reliable macro/meso/micro labels across 50 papers | `experiments/sh0_weak_labels/` |
| **SH1** | DONE | **CONFIRMED** | SciBERT+LogReg probe: macro-F1 = 0.72 | `experiments/sh1_linear_probe/` |
| **SH2 (QA)** | DONE | **NOT CONFIRMED** | Doc-span steering d=0.043; cross-space mismatch + layer selection bug | `experiments/sh2_activation_steering/` |
| **SH2b (QA)** | DONE | **PARTIAL** | QA-context steering \|d\|=0.546 but direction-inverted + H3 fail | `experiments/sh2_activation_steering/` |
| **SH2-summ** | DONE | **CONFIRMED** | Summarization steering d=0.679, ROUGE-L improves, H1+H2+H3 all pass | `experiments/sh2_activation_steering/` |
| **SH3** | DONE | **PARTIAL** | Soft SLoD-weighted retrieval beats baseline on attribution F1; hard routing fails | `experiments/sh3_hierarchical_rag/` |
| **SH4** | DONE | **PARTIAL** | Combined model AUROC=0.676; drift-only ≈ random (0.52) | `experiments/sh4_drift_extraction/` |
| **SH5** | DONE | **NOT CONFIRMED** | Scalar jump rate ↔ token-F1: ρ=0.003, p=0.90 (null) | `experiments/sh5_jump_rate/` |
| **SH5a** | DONE | **CONFIRMED** | Transition matrix: macro→macro self-loop ↔ attr-F1 ρ=-0.197; 2 reasoning styles | `experiments/sh5a_transition_matrix/` |
| **SH5c** | DONE | **CONFIRMED** | Context-reasoning alignment ↔ attr-F1 ρ=-0.135; SLoD routing improves alignment | `experiments/sh5c_context_alignment/` |
| **SH5d** | DONE | **STRONG** | Continuous SLoD-axis projection ↔ attr-F1 ρ=+0.219; SLoD-specific (3× > orthogonal) | `experiments/sh5d_continuous_projection/` |

---

## Background & Motivation

**SLoD** (Semantic Level of Detail): text/knowledge represented at multiple abstraction levels (macro/meso/micro), grounded in Poincare ball geometry. Abstract concepts cluster near the center, specific entities near the boundary; the heat kernel acts as a "zoom operator."

**Key insight:** abstraction level may already be geometrically encoded in frozen LLM embeddings (supported by Poincare embedding literature, Nickel & Kiela 2017, and transformer probing studies).

**Project relevance (TKH):** Temporal Knowledge Hypergraphs have natural granularity levels (paper -> section -> method -> value). Routing queries to the right level and filtering extraction errors by drift signal directly improves TKH construction at scale.

---

## Dependency Graph

```
SH0 (weak labels)
  └→ SH1 (linear probe, F1=0.72) ──────────────────────────────┐
       ├→ SH2 ❌(QA) → SH2-summ ✅ (steering d=0.679)          │
       ├→ SH3 🟡 (soft routing +0.031 F1)                       │
       └→ SH4 🟡 (combined AUROC=0.676) ──────────────────────┤
                                                                 └→ SH5 ❌ → SH5a ✅ → SH5c ✅ → SH5d ✅✅
```

---

## Revised Thesis Narrative

1. **Mechanism (SH1) — Strong.** SLoD axis is linearly decodable (F1=0.72) and geometrically coherent (SH5d validates Cohen's d=2.65 separation).

2. **Control (SH2) — Confirmed on summarization.** Activation steering shifts generation toward target abstraction (d=0.679) without degrading quality. Key design principle: **task-domain alignment between steering target and evaluation metric is necessary.**

3. **Systems (SH3) — Moderate.** Soft SLoD-weighted retrieval with parent expansion produces consistent but modest improvements. Hard routing fails.

4. **Application (SH4) — Weak on drift, useful combined.** SLoD drift alone does not predict extraction quality. Combined with surface features it adds marginal value.

5. **Behavioral (SH5→SH5d) — Strong, revised.** Cross-level reasoning is beneficial, not harmful. "Macro-stuck" reasoning is the failure mode. Continuous SLoD-axis projection is the strongest single predictor (ρ=+0.219).

---

## Negative Results Worth Reporting

| What failed | Why it matters |
|---|---|
| SH3 hard routing | Single-level routing destroys diversity. SLoD is better as a soft scoring signal. |
| SH3 Specter2 embeddings | Domain-specific underperformed general-purpose (MiniLM) at paragraph-level retrieval. |
| SH4 drift-only features | SLoD drift between expected and realized abstraction does not predict extraction correctness. |
| SH5 scalar jump rate | The simplest behavioral metric carries zero signal for answer quality. |
| SH2 doc-span steering | Doc-span difference-of-means vectors do not produce usable steering directions. |
| SH2 QA evaluation ceiling | SciBERT SLoD axis cannot discriminate QA answers above d≈0.121 — genre mismatch. |
| SH2 abs() layer selection bug | Using `max(abs(shift))` chose anti-correlated layer. Signed selection is required. |

---

## Open Directions

| Direction | Description | Effort | Priority |
|---|---|---|---|
| **SH6** | Human/model preference for SLoD-steered summaries | 3–5 days | High |
| **SH7** | Cross-domain portability (biomedical, legal, news) | 3–5 days | High |
| **SH2-QA-v2** | QA-genre SLoD evaluation axis | 3–5 days | Medium |
| **SH8** | Combined retrieval + steering pipeline | 5–7 days | Medium |
| **Cross-domain validation** | SH1 probe on bio/physics papers | 2–3 days | High |
| **Larger retrieval benchmark** | SH3 on LoCoMo or S2ORC QA | 3–5 days | Medium |
| **ADAM-Bench SLoD analysis** | SLoD typing on 27K papers, 7M evidence objects | 3–5 days | Medium |
| **Combined SH5 predictor** | Feature selection on SH5a+SH5c+SH5d features | 1–2 days | Low |
