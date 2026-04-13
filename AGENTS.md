# AGENTS.md

## What This Project Is

SemanticScale investigates **Semantic Level of Detail (SLoD)** in LLM representations. The core claim: frozen embeddings encode a continuous abstraction axis (macro/meso/micro) that is linearly decodable, steerable, and observable in reasoning traces. The project proves this across 10 completed experiments (SH0-SH5d) on scientific papers from QASPER.

**Three contribution layers:**
- **Mechanistic** (SH0-SH1): SLoD is geometrically encoded in embedding space, linearly separable with F1=0.72
- **Systems** (SH2-SH4): The axis can steer generation (d=0.679 on summarization) and improve retrieval (+0.031 F1 with soft routing)
- **Behavioral** (SH5-SH5d): Continuous SLoD projection is the strongest single predictor of reasoning quality (rho=+0.219)

## Quick Start

```bash
uv run --env-file .env python setup_data.py                    # downloads ~3.4 GB from HuggingFace
uv run --env-file .env python setup_data.py --experiments sh0 sh1  # or download selectively

uv run --env-file .env python experiments/sh1_linear_probe/scripts/run_sh1.py
```

## Repository Layout

```
SemanticScale/
├── experiments/                  # 10 self-contained experiments
│   ├── sh0_weak_labels/          # Heuristic SLoD labeling (83K spans)
│   ├── sh1_linear_probe/         # Linear decodability proof (F1=0.72)
│   ├── sh2pre_baseline_steering/ # Preliminary steering baseline
│   ├── sh2_activation_steering/  # Generation steering (QA + summarization)
│   ├── sh3_hierarchical_rag/     # SLoD-routed retrieval
│   ├── sh4_drift_extraction/     # Extraction quality prediction
│   ├── sh5_jump_rate/            # Scalar jump rate (null result)
│   ├── sh5a_transition_matrix/   # Transition decomposition
│   ├── sh5c_context_alignment/   # Context-reasoning alignment
│   └── sh5d_continuous_projection/ # Continuous SLoD projection (strongest result)
├── docs/                         # narrative.md, roadmap.md, data_dictionary.md,
│                                 #   reproduction.md, index.html (GitHub Pages)
├── data/                         # ~3.4 GB, gitignored, from HuggingFace
├── setup_data.py                 # HuggingFace data downloader
├── requirements.txt              # Consolidated deps (torch, transformers, scikit-learn, etc.)
└── research-map.md               # Detailed methodology reference (untracked)
```

### Experiment Directory Structure

Every experiment follows the same layout:

```
sh{N}_{name}/
├── config.yaml        # All hyperparameters, model names, data paths
├── DESIGN.md          # Detailed experimental design and hypotheses
├── README.md          # Quick results summary
├── scripts/           # Numbered pipeline steps: 01_*.py, 02_*.py, ...
│   └── run_sh{N}.py   # Some have an orchestrator that runs all steps
├── src/               # Python modules (task-specific + utils.py)
│   └── utils.py       # Config loader, logging setup, I/O helpers
└── reports/           # Generated output: markdown reports, PNGs, JSON metrics
```

## Experiment Dependency Graph

```
SH0 (83K labeled spans)
  └─> SH1 (linear probe, F1=0.72) ─────────────────────────┐
       ├─> SH2 (steering: QA d=0.043, summ d=0.679)       │
       ├─> SH3 (soft routing +0.031 F1)                    │
       └─> SH4 (combined AUROC=0.676)         ─────────────┤
                                                            v
       SH3 retrieval results ───────────────────────> SH5 (2000 CoT traces)
                                                       ├─> SH5a (rho=-0.197)
                                                       ├─> SH5c (rho=-0.135)
                                                       └─> SH5d (rho=+0.219)
```

**What this means in practice:**
- SH0 must run first (produces labeled spans)
- SH1 depends on SH0 output (embeddings + probe)
- SH2, SH3, SH4 each depend on SH0 + SH1 (can run in parallel)
- SH5 depends on SH3 retrieval results + SH1 probe
- SH5a, SH5c, SH5d reanalyze SH5 data (can run in parallel after SH5)

Full reproduction order: `docs/reproduction.md`

## Experiment Results Summary

| Exp | What It Tests | Verdict | Key Metric |
|-----|--------------|---------|------------|
| SH0 | Heuristic SLoD labels from document structure | CONFIRMED | 84.9% human agreement |
| SH1 | Linear decodability from frozen embeddings | CONFIRMED | SciBERT macro-F1=0.72 |
| SH2-QA | Activation steering on QA | NOT CONFIRMED | d=0.043 (genre mismatch) |
| SH2-summ | Activation steering on summarization | CONFIRMED | d=0.679 |
| SH3 | SLoD-routed hierarchical RAG | PARTIAL | Soft +0.031 F1, hard fails |
| SH4 | Abstraction drift predicts extraction quality | PARTIAL | Combined AUROC=0.676, drift-only=0.521 |
| SH5 | Jump rate correlates with answer quality | NOT CONFIRMED | rho=0.003, p=0.90 |
| SH5a | Transition matrices reveal reasoning patterns | CONFIRMED | rho=-0.197 |
| SH5c | Context-reasoning SLoD alignment | CONFIRMED | rho=-0.135 |
| SH5d | Continuous SLoD-axis projection | STRONG | rho=+0.219 (strongest predictor) |

## Negative Results and Pitfalls

These are critical context for anyone extending the project:

| Failure | Root Cause | Lesson |
|---------|-----------|--------|
| SH2 QA steering d=0.043 | SciBERT probe trained on doc-spans can't discriminate QA answers | **Task-domain alignment is required** between steering target and evaluation |
| SH2 `abs()` layer selection bug | `max(abs(shift))` chose an anti-correlated layer | **Always use signed shift** for layer selection |
| SH3 hard routing F1=0.199 | Single-level retrieval destroys diversity | SLoD is a **soft scoring signal**, not a hard filter |
| SH3 Specter2 underperformed MiniLM | Domain-specific embeddings worse at paragraph-level retrieval | Don't assume domain-specific is always better |
| SH4 drift-only AUROC=0.521 | Granularity mismatch: probe trained on short spans, applied to long paragraphs | **Match probe granularity to input granularity** |
| SH5 scalar jump rate rho=0.003 | Too coarse a metric | Scalar aggregation destroys structure; need matrix decomposition (SH5a) |

**Pattern:** Most failures stem from domain or granularity mismatch. The SLoD axis is real but has a specific domain of validity.

## Data

- All data lives in `data/` at repo root (gitignored)
- Hosted at https://huggingface.co/datasets/anaderi/semantic-scale-data
- Download: `python setup_data.py` (or `--experiments sh0 sh1` for partial)
- Sizes: sh0=155MB, sh1=315MB, sh2=172MB, sh3=2.1GB, sh4=390MB, sh5=16MB, sh5a/c/d=~30MB
- Experiments reference data via `../../data/sh{N}` in `config.yaml`
- Override with `SLOD_DATA_ROOT` environment variable
- Full schema: `docs/data_dictionary.md`

## Key Technical Details

**Embedding models used:** SciBERT (`allenai/scibert_scivocab_uncased`), Specter2 (`allenai/specter2`), MiniLM (`all-MiniLM-L6-v2`). SciBERT is primary for SLoD probing; MiniLM is better for retrieval at paragraph level.

**LLMs used:** Mistral-7B (SH2 steering), Claude Haiku (SH4 extraction, SH5 CoT generation).

**Core Python stack:** torch, transformers, sentence-transformers, scikit-learn, scipy, pandas, matplotlib, seaborn. See `requirements.txt` for pinned versions.

**Config pattern:** Every experiment reads `config.yaml` via `src/utils.py:load_config()`. All hyperparameters, model names, data paths, and thresholds live there. Scripts never hardcode values.

## Running Experiments

Use `uv` for running code and managing dependencies:

```bash
uv run --env-file .env python experiments/sh1_linear_probe/scripts/run_sh1.py
```

Pipeline scripts are numbered and should be run in order (01, 02, ..., 07). Some experiments have a single orchestrator script (`run_*.py`) that runs all steps.

## Conventions for Modifying Code

1. **No absolute paths.** Use `config.yaml` paths or `SLOD_DATA_ROOT`. Run `grep -r "/home/" experiments/` to verify.
2. **Config-driven.** All hyperparameters go in `config.yaml`, not in Python code.
3. **Preserve script numbering.** `01_`, `02_`, ... signals execution order within each experiment.
4. **Experiments are self-contained.** No cross-imports between experiment `src/` directories. Shared library extraction to a `slod/` package is planned but not yet done.
5. **Reports are generated artifacts.** Never hand-edit files in `reports/`. Regenerate by re-running the pipeline scripts.
6. **Data path fix pattern.** If you encounter a broken path (e.g., `data_sh0/`, `data_sh5/`, bare `data/`), redirect to `../../data/sh{N}` relative to the experiment directory. This is leftover from the pre-consolidation era when experiments lived in separate repos.

## Migration History

This repo was created by merging five independent experiment branches, each with its own data layout. All paths were rewritten to `../../data/sh{N}`, but stray references to old conventions may still exist. The fix is always the same: point to `../../data/sh{N}` relative to the experiment directory.

## Testing

No formal test suite. Verify changes by:
1. Running the relevant experiment's pipeline scripts
2. Checking `reports/` output matches expected metrics (see each experiment's README.md)
3. `grep -r "/home/" experiments/` should return nothing

## Documentation Index

| Document | What It Contains |
|----------|-----------------|
| `docs/roadmap.md` | Research roadmap, status table, dependency graph, open directions |
| `docs/narrative.md` | Story-form research narrative ("The Case of the Hidden Geometry") |
| `docs/data_dictionary.md` | Schema for every data file across all experiments |
| `docs/reproduction.md` | Step-by-step reproduction guide in dependency order |
| `docs/index.html` | GitHub Pages site with interactive dependency graph |
| `research-map.md` | Detailed methodology, exit criteria, pseudo-code (untracked) |

## Open Directions

| Direction | Description | Priority |
|-----------|------------|----------|
| SH6 | Human/model preference evaluation for steered summaries | High |
| SH7 | Cross-domain portability (biomedical, legal, news) | High |
| SH2-QA-v2 | QA-genre SLoD evaluation axis (fix the genre mismatch) | Medium |
| SH8 | Combined retrieval + steering pipeline | Medium |
| `slod/` package | Extract shared utilities from experiments into reusable library | Planned |
