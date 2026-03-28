# SemanticScale: Semantic Level of Detail in LLM Representations

> Frozen LLM representations encode a continuous Semantic Level of Detail (SLoD) axis that is linearly decodable without retraining. Exploiting this axis — to match retrieval granularity to query abstraction, to detect output drift, and to steer generation — measurably improves scientific QA attribution and TKH extraction quality. Continuous embedding-space dynamics in reasoning chains provide supporting behavioral evidence.

## Overview

**SLoD** (Semantic Level of Detail) describes text at multiple abstraction levels — **macro** (high-level concepts), **meso** (section-level descriptions), and **micro** (specific entities, methods, values). This project demonstrates that these levels are geometrically encoded in frozen LLM embedding spaces and can be detected, steered, and observed in reasoning traces.

The research is organized as 10 experiments (SH0–SH5d) that build a five-layer proof:

1. **Mechanism** (SH1) — SLoD is linearly decodable from frozen embeddings
2. **Control** (SH2) — Activation steering shifts generation along the SLoD axis
3. **Systems** (SH3) — SLoD-aware retrieval improves evidence attribution
4. **Application** (SH4) — Combined SLoD + surface features predict extraction quality
5. **Behavioral** (SH5→SH5d) — Continuous SLoD projection predicts reasoning quality

## Experiment Summary

| Experiment | Hypothesis | Verdict | Key Result |
|---|---|---|---|
| **SH0** | Document structure provides SLoD labels | **CONFIRMED** | Reliable macro/meso/micro labels across 50 papers |
| **SH1** | Linear probe decodes SLoD from frozen embeddings | **CONFIRMED** | SciBERT macro-F1 = 0.72 |
| **SH2 (QA)** | Activation steering moves SLoD in QA | **NOT CONFIRMED** | d=0.043; evaluation axis genre mismatch |
| **SH2 (Summ)** | Activation steering moves SLoD in summarization | **CONFIRMED** | d=0.679, ROUGE-L improves |
| **SH3** | SLoD routing improves retrieval | **PARTIAL** | Soft routing +0.031 F1; hard routing fails |
| **SH4** | SLoD drift predicts extraction quality | **PARTIAL** | Combined AUROC=0.676; drift-only ≈ random |
| **SH5** | Jump rate correlates with answer quality | **NOT CONFIRMED** | ρ=0.003 (null) |
| **SH5a** | Transition matrices reveal reasoning styles | **CONFIRMED** | Macro-stuck ρ=−0.197; 2 reasoning styles |
| **SH5c** | Context-reasoning alignment predicts quality | **CONFIRMED** | Alignment gap ρ=−0.135 |
| **SH5d** | Continuous SLoD projection predicts quality | **STRONG** | ρ=+0.219 (3× > orthogonal control) |

## Quick Start

```bash
git clone git@github.com:omniscale-ai/SemanticScale.git
cd SemanticScale
pip install -r requirements.txt

# Download experiment data (3.4 GB total)
python setup_data.py

# Or download specific experiments only
python setup_data.py --experiments sh0 sh1

# Run an experiment
cd experiments/sh1_linear_probe
python scripts/run_sh1.py
```

## Project Structure

```
SemanticScale/
├── README.md
├── requirements.txt
├── setup_data.py                 # Data download from Hugging Face
├── docs/
│   ├── narrative.md              # Research narrative
│   ├── roadmap.md                # Full research roadmap with results
│   ├── data_dictionary.md        # Data file documentation
│   └── reproduction.md           # End-to-end reproduction guide
├── experiments/
│   ├── sh0_weak_labels/          # Heuristic SLoD labeling from document structure
│   ├── sh1_linear_probe/         # Linear decodability of SLoD from frozen embeddings
│   ├── sh2_activation_steering/  # Activation steering along SLoD axis
│   ├── sh2pre_baseline_steering/ # Pre-scale steering baseline
│   ├── sh3_hierarchical_rag/     # SLoD-routed hierarchical RAG
│   ├── sh4_drift_extraction/     # Abstraction drift vs extraction quality
│   ├── sh5_jump_rate/            # Jump rate as CoT behavioral signature
│   ├── sh5a_transition_matrix/   # Transition matrix decomposition
│   ├── sh5c_context_alignment/   # Context-reasoning SLoD alignment
│   └── sh5d_continuous_projection/ # Continuous SLoD-axis projection
└── data/                         # Downloaded via setup_data.py (not in git)
```

Each experiment directory contains:
- `README.md` — Hypothesis, method, results summary
- `DESIGN.md` — Detailed experimental design
- `config.yaml` — Experiment configuration
- `scripts/` — Numbered pipeline scripts (01–07)
- `src/` — Python modules
- `reports/` — Conclusive reports and figures

## Data

Experiment data (~3.4 GB) is hosted on [Hugging Face](https://huggingface.co/datasets/omniscale-ai/SemanticScale-data) and downloaded via `setup_data.py`. See [docs/data_dictionary.md](docs/data_dictionary.md) for detailed documentation of each data file.

| Dataset | Size | Contents |
|---|---|---|
| sh0 | 155 MB | QASPER spans with SLoD labels |
| sh1 | 315 MB | Embeddings (SciBERT, MiniLM, Specter2), annotations |
| sh2 | 172 MB | Steering vectors, baseline/prompted/steered answers |
| sh3 | 2.1 GB | Paragraph embeddings, retrieval indices, results |
| sh4 | 390 MB | LLM extractions, features, model outputs |
| sh5 | 16 MB | CoT traces, SLoD tags, answer scores |

## Key Findings

**Task-domain alignment** is the critical design principle discovered through the SH2 series: the evaluation axis can only measure what it was trained to see. Five QA steering experiments "failed" because the SciBERT SLoD axis cannot discriminate QA answers (out-of-distribution). Switching to summarization — which produces document-like text — yielded d=0.679.

**The signal is in level, not dynamics** (SH5d): traces that operate at more micro-level detail produce better evidence attribution (ρ=+0.219), and this effect is SLoD-specific (3× stronger than orthogonal controls). The original scalar jump rate hypothesis was wrong, but finer decomposition revealed the real signal.

## Dependency Graph

```
SH0 ✅ (weak labels)
  └→ SH1 ✅ (linear probe, F1=0.72) ──────────────────────────────┐
       ├→ SH2 ❌(QA) → SH2-summ ✅ (steering d=0.679)             │
       ├→ SH3 🟡 (soft routing +0.031 F1)                          │
       └→ SH4 🟡 (combined AUROC=0.676) ──────────────────────────┤
                                                                    └→ SH5 ❌ → SH5a ✅ → SH5c ✅ → SH5d ✅✅
```

## Citation

```bibtex
@misc{semanticscale2026,
  title={SemanticScale: Semantic Level of Detail in LLM Representations},
  author={omniscale-ai},
  year={2026},
  url={https://github.com/omniscale-ai/SemanticScale}
}
```

## License

TBD
