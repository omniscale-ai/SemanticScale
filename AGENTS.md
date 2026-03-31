# CLAUDE.md

## Project Overview

SemanticScale investigates Semantic Level of Detail (SLoD) in LLM representations. The project contains 10 completed experiments (SH0–SH5d) demonstrating that frozen embeddings encode a continuous abstraction axis that is linearly decodable, steerable, and observable in reasoning traces.

## Repository Structure

```
experiments/sh{N}_{name}/    # Each experiment is self-contained
  config.yaml                # Experiment configuration
  scripts/                   # Numbered pipeline scripts (01–07)
  src/                       # Python modules
  reports/                   # Results, figures, conclusive reports
  README.md                  # Hypothesis, method, results
  DESIGN.md                  # Detailed experimental design
```

Shared documentation lives in `docs/`. Data (~3.4 GB) is external on Hugging Face.

## Migration History

This repository was created by merging five individual experiment branches that were developed independently. Each branch originally had its own data layout — some used symlinks to an external directory, others had local `data/` folders with different naming conventions. During consolidation, all data paths were rewritten to point to a single `data/` directory at the repo root, but **there may still be stray references to old path conventions** (e.g., `data_sh0/`, `data_sh5/`, or bare `data/` meaning the experiment's own data). If you encounter a broken path, the fix is almost always to redirect it to `../../data/sh{N}` relative to the experiment directory.

## Data

- All experiment data lives in `data/` at the repo root (gitignored, not committed)
- Hosted at https://huggingface.co/datasets/anaderi/semantic-scale-data
- Download via `python setup_data.py` (or `--experiments sh0 sh1` for partial)
- Experiments reference data via relative paths like `../../data/sh1` in `config.yaml`
- Set `SLOD_DATA_ROOT` env var to override the default data location

## Running Experiments
Use `uv` for running the code and managing dependencies:

```bash
uv run --env-file .env python experiments/sh1_linear_probe/scripts/run_sh1.py
```

Pipeline scripts are numbered and should be run in order (01, 02, ..., 07). Some experiments have a single orchestrator script (`run_*.py`) that runs all steps.

## Key Conventions

- **Config-driven**: All hyperparameters and paths are in `config.yaml` per experiment
- **Reports are artifacts**: `reports/` directories contain generated markdown and PNGs — treat as output, not source
- **Experiments are independent**: Each has its own `src/` with no cross-imports between experiments (shared library extraction is planned but not yet done)
- **Data dependencies**: SH1 depends on SH0 output, SH2/SH3/SH4 depend on SH1, SH5* depend on SH5. See `docs/roadmap.md` for the full dependency graph.

## When Modifying Code

- Do not add hardcoded absolute paths — use `config.yaml` or `SLOD_DATA_ROOT`
- Preserve the numbered script convention (01_, 02_, etc.) for pipeline ordering
- Keep experiment directories self-contained — shared utilities go in a future `slod/` package
- Results and figures in `reports/` are generated output; regenerate rather than hand-edit

## Testing

There is no formal test suite yet. Verify changes by:
1. Running the relevant experiment's pipeline scripts
2. Checking that `reports/` output matches expected metrics (see each experiment's README.md)
3. `grep -r "/home/" experiments/` should return nothing (no hardcoded paths)
