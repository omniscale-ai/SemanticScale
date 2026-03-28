# SH0 -- Weak Label Bootstrap

Produces a heuristically labeled dataset of scientific paper text spans classified as
macro/meso/micro semantic levels of detail (SLoD), using document structure from QASPER.

## Status: COMPLETE

All exit criteria met. 83,135 labeled spans with 84.9% human agreement.

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Full pipeline (downloads QASPER ~150MB on first run)
python scripts/run_labeling.py

# Validation sampling
python scripts/run_validation.py --skip-llm
```

Pipeline takes ~18 minutes (dominated by spaCy sentence processing of 83K paragraphs).

## Data

This experiment expects a `data/` directory. Set `SLOD_DATA_ROOT` environment variable or
create a symlink: `ln -s /path/to/SLoD_data/SH0 data`

### Outputs

| File | Description | Size |
|---|---|---|
| `data/qasper_slod_spans.jsonl` | Full labeled dataset | 83,135 spans |
| `data/qasper_slod_length_matched.jsonl` | Length-controlled subset | 37,278 spans (12,426/class) |
| `data/validation_report.json` | Quality metrics | JSON |
| `data/label_distribution.png` | Distribution visualization | PNG |

## Architecture

```
src/
  load_qasper.py        -- QASPER data loading from HuggingFace
  heuristic_labeler.py  -- Section regex + position rules + content scoring
  length_matcher.py     -- Length-matched subset creation
  quality_checks.py     -- 6 automated sanity checks
  utils.py              -- I/O, logging, config loading

scripts/
  run_labeling.py       -- End-to-end orchestrator (steps 1-4)
  run_validation.py     -- Stratified sampling + optional LLM validation
```

## Key Design Decisions

- QASPER only (1,585 NLP papers). S2ORC deferred due to API access complexity.
- Section names parsed with `:::` delimiter splitting for QASPER subsection hierarchy.
- Micro score threshold: 0.35 (lowered from initial 0.5).
- Section coverage threshold: 70% span-weighted.
- Synthetic fallback not triggered -- heuristic quality was sufficient.

## Downstream

Outputs feed into SH1 (linear probe training on frozen embeddings).
