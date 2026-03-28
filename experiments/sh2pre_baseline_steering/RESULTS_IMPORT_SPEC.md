# SH2 Results Import Specification

## What to Return

When the experiment is complete, package the following directory structure and send it back. The upstream team will unpack it into `playground/SLoD-SH2/` and integrate with the research vault.

## Required Files

```
results_sh2/
├── data/
│   ├── environment_check.json          # Stage A: GPU info, model paths, data validation
│   ├── eval_slod_axis.npz              # Stage B: SLoD evaluation axis + validation stats
│   ├── steering_vectors.npz            # Stage C: Per-layer steering vectors + selected layer
│   ├── baseline_answers.jsonl          # Stage D: 500 baseline answers
│   ├── steered_answers.jsonl           # Stage E: Steered answers (both directions, selected α)
│   └── results/
│       ├── evaluation_results.json     # Stage F: All metrics (SLoD shift, surface, quality)
│       ├── alpha_sweep.json            # Stage E: α sensitivity analysis
│       └── layer_selection.json        # Stage C: Layer selection results + justification
├── reports/
│   ├── SH2_results.md                  # Stage G: Auto-generated report with verdict
│   ├── SH2_CONCLUSIVE_REPORT.md        # Analyst: Final conclusive report (if analyst agent ran)
│   └── figures/
│       ├── slod_axis_validation.png    # Stage B: SH1 test set axis validation
│       ├── slod_shift_distribution.png # Stage G: Δ_SLoD histogram
│       ├── alpha_sensitivity.png       # Stage G: SLoD shift vs α
│       ├── layer_comparison.png        # Stage G: SLoD shift by layer
│       ├── quality_preservation.png    # Stage G: Token-F1 baseline vs steered
│       ├── surface_metrics.png         # Stage G: Surface metric changes
│       └── example_outputs.png         # Stage G: Side-by-side examples (optional)
├── src/                                # Final source code (with any modifications)
│   ├── __init__.py
│   ├── utils.py
│   ├── embedding.py
│   ├── slod_axis.py
│   ├── steering.py
│   ├── evaluate.py
│   └── visualization.py
├── scripts/                            # Final scripts
│   ├── 01_setup_environment.py
│   ├── 02_compute_slod_axis.py
│   ├── 03_compute_steering_vector.py
│   ├── 04_generate_baseline.py
│   ├── 05_generate_steered.py
│   ├── 06_evaluate.py
│   └── 07_report.py
├── config.yaml                         # Final config (may differ if agent adjusted settings)
├── COORDINATION.md                     # Updated with final status, all iterations, verdict
└── git.bundle                          # `git bundle create git.bundle --all`
```

## How to Create the Bundle

```bash
# From the SLoD-SH2 project directory:

# 1. Create git bundle (preserves full commit history)
git bundle create git.bundle --all

# 2. Package everything
mkdir -p results_sh2
cp -r data/ results_sh2/data/
cp -r reports/ results_sh2/reports/
cp -r src/ results_sh2/src/
cp -r scripts/ results_sh2/scripts/
cp config.yaml COORDINATION.md git.bundle results_sh2/

# 3. Archive
tar czf results_sh2.tar.gz results_sh2/
```

## Data Format Specifications

### `steering_vectors.npz`
```python
# Keys:
"vectors"        # (n_layers, hidden_dim) — all per-layer steering vectors
"selected_layer" # int — index of best layer
"layer_shifts"   # (n_tested,) — SLoD shift per tested layer
"model_name"     # str — generative model used
"n_macro_spans"  # int — number of macro spans used
"n_micro_spans"  # int — number of micro spans used
```

### `baseline_answers.jsonl`
```json
{"question_id": "1604.02201__q2", "answer_text": "The model was trained using...", "n_tokens": 128, "answer_type": "abstractive"}
```

### `steered_answers.jsonl`
```json
{"question_id": "1604.02201__q2", "direction": "micro", "alpha": 2.0, "answer_text": "Specifically, the model was trained with...", "n_tokens": 145}
```

### `evaluation_results.json`
```json
{
  "h1_slod_shift": {
    "steered_micro": {"mean_delta": 0.15, "std_delta": 0.08, "p_value": 0.001, "cohens_d": 0.82},
    "steered_macro": {"mean_delta": -0.12, "std_delta": 0.07, "p_value": 0.002, "cohens_d": -0.71},
    "pass": true
  },
  "h2_surface": {
    "steered_micro": {
      "entity_density": {"baseline_mean": 0.05, "steered_mean": 0.08, "p_value": 0.01},
      "citation_density": {"baseline_mean": 0.02, "steered_mean": 0.03, "p_value": 0.15},
      "numeric_density": {"baseline_mean": 0.04, "steered_mean": 0.06, "p_value": 0.03},
      "mean_sentence_length": {"baseline_mean": 18.5, "steered_mean": 16.2, "p_value": 0.04}
    },
    "n_significant": 3,
    "pass": true
  },
  "h3_factuality": {
    "baseline_mean_f1": 0.35,
    "steered_micro_mean_f1": 0.33,
    "steered_macro_mean_f1": 0.34,
    "max_drop": 0.02,
    "pass": true
  },
  "overall_verdict": "CONFIRMED",
  "model_used": "mistralai/Mistral-7B-Instruct-v0.3",
  "selected_layer": 16,
  "selected_alpha": 2.0,
  "n_questions": 500
}
```

## Import Procedure (for upstream team)

```bash
# 1. Unpack results
tar xzf results_sh2.tar.gz

# 2. Restore git history
cd playground/SLoD-SH2
git init  # if not already
git bundle unbundle results_sh2/git.bundle
git checkout master  # or main

# 3. Copy data and reports
cp -r results_sh2/data/* data/
cp -r results_sh2/reports/* reports/

# 4. Update COORDINATION.md if not already updated
cp results_sh2/COORDINATION.md .

# 5. Verify
python -c "
import json, numpy as np
# Check steering vectors
sv = np.load('data/steering_vectors.npz', allow_pickle=True)
print(f'Steering vectors shape: {sv[\"vectors\"].shape}')
print(f'Selected layer: {sv[\"selected_layer\"]}')
# Check evaluation results
with open('data/results/evaluation_results.json') as f:
    r = json.load(f)
print(f'Verdict: {r[\"overall_verdict\"]}')
print(f'H1 pass: {r[\"h1_slod_shift\"][\"pass\"]}')
print(f'H3 pass: {r[\"h3_factuality\"][\"pass\"]}')
"

# 6. Update roadmap
# Edit 1-Project/2026-CKL-KnowledgeDiscovery/Semantic-emergence-SLoD-roadmap.md
# Update SH2 row in Execution Status table
```
