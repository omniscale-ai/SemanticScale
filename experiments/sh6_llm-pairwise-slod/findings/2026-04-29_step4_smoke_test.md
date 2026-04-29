# Step 4 — Orchestration Script & End-to-End Smoke

**Date:** 2026-04-29
**Status:** completed

## What was built

`experiments/sh6_llm-pairwise-slod/scripts/05b_lightgbm_comparison.py` —
thin CLI that:

1. Resolves the run directory (via `--config` like the incumbent
   05-script, or `--features-csv` for ad-hoc cases).
2. Calls `failure_analysis_runner.run_models_on_run` on
   `[logreg, lightgbm]` (overridable with `--models`).
3. Streams INFO logs both to stdout and to
   `reports/{dataset}/{run_slug}/logs/05b_lightgbm_<UTC-ts>.log`.

## Verification

```
uv run python experiments/sh6_llm-pairwise-slod/scripts/05b_lightgbm_comparison.py \
    --config experiments/sh6_llm-pairwise-slod/config/frontierscience-deepseek.yaml
```

Output (relevant lines):
```
Run dir: .../reports/frontierscience/deepseek/deepseek-v3.2_reasoning-auto
logreg   — AUC 0.8348 ± 0.0847 (0.31s)
lightgbm — AUC 0.8735 ± 0.0601 (1.66s)
```

Artifacts on disk after the run:
```
reports/frontierscience/deepseek/deepseek-v3.2_reasoning-auto/
├── artifacts/
│   ├── cv_metadata_lightgbm.json
│   ├── cv_metadata_logreg.json
│   ├── feature_importance_lightgbm.csv
│   ├── feature_importance_logreg.csv
│   ├── oof_predictions_lightgbm.parquet
│   └── oof_predictions_logreg.parquet
└── logs/
    └── 05b_lightgbm_20260428T223400Z.log
```

Three checks passed:

- **Path resolution.** First try used `config_path.parent.parent` and
  resolved to `.../experiments/reports/...` (wrong tree). Fixed by
  reading `config["_project_root"]` exactly like
  `05_analyze_failure_modes.py`. Now the new artifacts land alongside
  the incumbent report rather than in a parallel directory.
- **Logreg equivalence.** The script's `logreg` AUC = 0.8348 matches
  Step 2 byte-for-byte and the existing `failure_prediction.md` (0.835).
- **LightGBM lift.** Δ-AUC = +0.039 vs incumbent on this single dataset.
  Above the +0.03 decision threshold but the win is *not declared* yet —
  it requires the paired-bootstrap CI on Δ-AUC and the ≥3-of-5 rule
  from Step 5.

## Implementation notes worth recording

- The script never overwrites incumbent reports
  (`failure_prediction.md`, `failure_modes.csv`, `*.png`). All new files
  go into `artifacts/` or `logs/` subdirs that didn't previously exist.
- Default models = `[logreg, lightgbm]`. Re-running logreg on every
  invocation is intentional — it gives Step 5 a guaranteed paired
  baseline even if the incumbent report is missing or out of sync.
- Log filename uses UTC ISO timestamp (`YYYYMMDDTHHMMSSZ`) to keep
  multiple runs sortable.

## Decision

Smoke run is clean. Proceeding to Step 5: fan out across all 7 eligible
runs, build the cross-dataset comparison table, run the paired bootstrap
on Δ-AUC, decide whether LightGBM carries the comparison.

## Files written / modified

- `experiments/sh6_llm-pairwise-slod/scripts/05b_lightgbm_comparison.py`
- `experiments/sh6_llm-pairwise-slod/reports/frontierscience/deepseek/deepseek-v3.2_reasoning-auto/artifacts/*`
- `experiments/sh6_llm-pairwise-slod/reports/frontierscience/deepseek/deepseek-v3.2_reasoning-auto/logs/05b_lightgbm_*.log`
- `experiments/sh6_llm-pairwise-slod/findings/2026-04-29_step4_smoke_test.md`
