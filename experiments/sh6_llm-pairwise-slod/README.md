# SH6 — LLM Pairwise SLoD

This experiment studies whether pairwise LLM judgments of Semantic Level of
Detail (SLoD) over reasoning traces can explain or predict problem-solving
success and failure.

## Pipeline

1. `01_traces.py`
   Loads or generates traces and assigns an outcome label.
   For `frontierscience`, this means inference plus grading.
   For `processbench`, traces already exist and include step-level error labels.

2. `02_slod.py`
   Runs pairwise SLoD comparisons over trace chunks and aggregates them with a
   Bradley-Terry model.
   Output: `chunk_rankings.jsonl`

3. `03_analyze_accuracy.py`
   Summarises run-level accuracy across available runs of a dataset.

4. `04_plot_trajectories.py`
   Visualises mean and example SLoD trajectories for correct vs wrong items.

5. `05_analyze_failure_modes.py`
   Converts trajectories into a tabular feature matrix and tests whether those
   features predict success or failure out of sample.

6. `06_anchor_validation.py`
   Performs absolute SLoD calibration using hand-tiered anchors to determine if
   datasets lack absolute variance on the SLoD axis.

7. `07_advanced_failure_analysis.py`
   Advanced failure-prediction analyses including length-residualized features
   and gradient-boosted models with subject as a feature.

Other scripts:
- `05b_lightgbm_comparison.py`: Tests interaction signal using LightGBM.
- `05z_aggregate_models.py`: Aggregates model results across datasets.
- `05d/e/f/g`: UMAP diagnostics and categorical analyses for specific datasets.
- `run_sh6.py`: Orchestrator for the full pipeline.

## Stage 5: Failure Analysis

The failure-analysis stage lives in:

- Script entry point: [scripts/05_analyze_failure_modes.py](/home/kna/SemanticScale/experiments/sh6_llm-pairwise-slod/scripts/05_analyze_failure_modes.py)
- Core feature/model code: [src/semanticscale/sh6/failure_analysis.py](/home/kna/SemanticScale/src/semanticscale/sh6/failure_analysis.py)

### Inputs

- `traces.jsonl` from Stage 1
- `chunk_rankings.jsonl` from Stage 2

The two files are merged by `id`.

### Target label

`05_analyze_failure_modes.py` prefers `final_answer_correct` when available.
If that field does not exist, it falls back to `is_correct`.

This choice matters:

- `final_answer_correct` asks whether the final answer is right.
- `is_correct` may encode a stricter notion of reasoning cleanliness.

For datasets that contain both, the report includes their agreement rate.

### Feature extraction

Each reasoning and answer trajectory is:

1. mean-centered within the item
2. interpolated to a fixed number of points
3. summarised into interpretable features

Feature families:

- `length`
  Number of reasoning and answer chunks.
- `commitment`
  Range, monotonicity, and related signals about how strongly a trajectory
  commits to one abstraction direction.
- `landing`
  End-state features such as end value and end-minus-start.
- `thrashing`
  Direction changes, zero crossings, curvature, and total variation.
- `timing`
  Where major peaks, troughs, rises, and drops occur.
- `derailment`
  Largest drops and how much the trace falls from its peak.
- `answer_alignment`
  Cross-features comparing reasoning and answer trajectories on the shared
  within-item SLoD scale.

Two diagnostic columns are also written:

- `reasoning_pair_density`
- `answer_pair_density`

These describe tournament coverage, not reasoning behavior, so they are kept
in the CSV but excluded from the predictive models.

### Model comparison

The stage evaluates three feature sets:

- `length_only`
  Structural baseline using chunk counts only.
- `trajectory_shape`
  Trajectory-derived features only.
- `trajectory_full`
  Both together.

The default predictive model is a median-imputed, standardised logistic
regression with class balancing. It is the versatile "overall correctness"
predictor and is retained as-is.

This is intentionally simple. The goal is to answer a falsifiable question:
do trajectory-derived features predict success/failure beyond trivial
baselines?

### Interpretable failure-mode detectors

Layered on top of the overall predictor, Stage 5 runs a set of named detectors
(see [src/semanticscale/sh6/failure_modes.py](/home/kna/SemanticScale/src/semanticscale/sh6/failure_modes.py))
that each target one interpretable reasoning failure mode observed in SWE-agent
and FrontierScience traces:

Reasoning-trajectory detectors (SWE-agent-style failures):

| Mode | What it catches |
|---|---|
| `premature_exit` | Very short reasoning — model gave up or answered without exploring. |
| `rambling_overlong` | Reasoning runs far longer than on successful traces. |
| `thrashing` | Many SLoD direction changes — flip-flopping between levels. |
| `no_commitment` | Low end-to-end monotonicity — no clear abstraction arc. |
| `derailment_late` | Trace peaks on the SLoD axis and then falls away, failing to land. |
| `truncation_abort` | Agent exited on context/budget/format (SWE-agent style). |

Answer-trajectory detectors (FrontierScience-style failures, where a correct
answer is short and monotonic and a wrong answer rambles or over-claims):

| Mode | What it catches |
|---|---|
| `answer_drift` | Answer SLoD disconnected from the reasoning conclusion. |
| `answer_meandering` | Long oscillating answer — many SLoD direction changes inside the answer. |
| `answer_volatility` | Sudden SLoD jumps in the answer — a hedging / confabulation pattern. |
| `answer_uncommitted` | Low monotonicity inside the answer — never commits to a clear arc. |
| `answer_overrange` | Answer covers a wider SLoD range than the reasoning earned. |

Each continuous detector is calibrated against the **success-class
distribution** (90th percentile by default), so a flag has the concrete
meaning: "unusual compared to successful traces on this feature". The binary
`truncation_abort` fires on `exit_status` markers like `exit_context`,
`early_exit`, `exit_format`, `exit_cost`.

Each detector encodes a pre-registered directional hypothesis — *higher score
implies more failure-like*. The stage tests that hypothesis on every run with
a 95% percentile-bootstrap CI on the failure-AUC and assigns one of four
verdicts:

- `confirmed` — CI lower bound > 0.5; the directional claim holds.
- `inverted` — CI upper bound < 0.5; the score predicts *success* on this run.
- `inconclusive` — CI straddles 0.5.
- `insufficient_data` — too few rows scored or only one class.

Flag-level metrics (precision, recall, F1, lift) are reported **only when the
verdict is `confirmed`** — otherwise they would suggest evidence we don't have.
The raw AUC and CI are always reported so any verdict is auditable.

The two detector families are not symmetric by design — they target the
empirical patterns each dataset actually produces. Reasoning-side detectors
fire on SWE-agent (`thrashing` AUC≈0.78, `truncation_abort` AUC≈0.65,
`no_commitment` AUC≈0.71) and largely stay inconclusive on FrontierScience.
Answer-side detectors fire on FrontierScience (`answer_meandering` AUC≈0.86,
`answer_volatility` AUC≈0.87, `answer_uncommitted` AUC≈0.81) and report
`insufficient_data` on SWE-agent (no answer chunks). The framework reports
those mismatches honestly rather than averaging them away.

#### Methodological caveat

The two families were defined under different methodologies:

- **Reasoning-side detectors are pre-registered.** Their feature and direction
  were chosen from a qualitative read of SWE-agent failure traces *before*
  AUCs were computed. The `confirmed` verdict on SWE-agent is therefore a
  real falsification test.
- **Answer-side detectors are post-hoc.** They were discovered by ranking
  trajectory features by univariate AUC on FrontierScience and keeping the
  top performers, with the score sign chosen to make AUC > 0.5. The
  `confirmed` verdict on FrontierScience is therefore circular by
  construction; the bootstrap CI does not correct for selection from ~60
  candidate features. Verdicts on other datasets remain unbiased because
  those datasets did not influence the selection.

Concrete consequence for the **Signal capture** number: the FrontierScience
captured-fraction (≈95%) is inflated and should be read as a descriptive
upper bound. The SWE-agent captured-fraction (≈90%) is the trustworthy
estimate. Validating the answer-side detectors out of sample (e.g. on a
different FS generator) would convert their FS verdicts from circular to
falsifiable.

### Feasibility rules

The script will not report prediction metrics if the run is not statistically
identifiable, for example:

- only one target class is present
- there are too few examples per class for cross-validation

In those cases it still writes the feature table and a markdown report
explaining why prediction was skipped.

## Outputs

Stage 5 writes the following files under
`reports/{dataset}/{run_slug}/`:

- `trajectory_features.csv`
  Per-item feature matrix.
- `failure_modes.csv`
  Per-item detector scores and flags, with metadata for slicing.
- `failure_prediction_summary.json`
  Machine-readable metrics and feature rankings, including a
  `mode_detectors` block with per-detector metrics.
- `failure_prediction.md`
  Human-readable report.
- `failure_prediction_roc.png`
  Cross-validated ROC curves for the feature-set baselines.
- `failure_feature_coefficients.png`
  Top coefficients from the full logistic model.
- `failure_mode_detectors.png`
  Per-detector precision vs. base rate and score ROC-AUC.

## Running

Examples:

```bash
uv run python experiments/sh6_llm-pairwise-slod/scripts/05_analyze_failure_modes.py \
  --config experiments/sh6_llm-pairwise-slod/config-processbench.yaml

uv run python experiments/sh6_llm-pairwise-slod/scripts/05_analyze_failure_modes.py \
  --config experiments/sh6_llm-pairwise-slod/config-frontierscience-deepseek.yaml
```

You can override the target label manually:

```bash
uv run python experiments/sh6_llm-pairwise-slod/scripts/05_analyze_failure_modes.py \
  --config experiments/sh6_llm-pairwise-slod/config-frontierscience-deepseek.yaml \
  --target is_correct
```

## Config knobs

The optional `failure_analysis` block controls Stage 5:

```yaml
failure_analysis:
  target_label: auto
  interp_points: 20
  cv_folds: 5
  random_state: 42
  top_k_features: 12
```

Meaning:

- `target_label`
  `auto`, `final_answer_correct`, or `is_correct`
- `interp_points`
  Number of points used when resampling trajectories
- `cv_folds`
  Requested number of stratified folds
- `random_state`
  Seed for cross-validation and the logistic baseline
- `top_k_features`
  Number of coefficients shown in the coefficient plot

## Interpretation guidance

- If `trajectory_shape` clearly beats `length_only`, the failure-mode
  hypothesis has predictive support.
- If only `length_only` performs well, the signal may be mostly structural.
- Coefficients should be treated as descriptive aids for naming candidate
  failure modes, not as causal evidence.
- The feature CSV is the right place to start if you want to do per-subject,
  per-model, or cluster-based follow-up analysis.

## Running with a local model (SH7)

The cloud configs depend on OpenRouter / OpenAI calls for trace generation,
grading, and pairwise SLoD. SH7 swaps both for locally-served LLMs (vLLM,
OpenAI-compatible) so the whole pipeline can run without external API calls.

Two roles, two models:

- **Trace generator** (`traces.model`) — the model whose reasoning we're
  studying. DeepSeek-R1-Distill-Qwen-1.5B for the pilot, -32B for the real
  run.
- **Judge** (`grader.model` and `pairwise_slod.model`) — a non-reasoning
  model for deterministic JSON-schema judgments.
  Qwen3-30B-A3B-Instruct-2507 (MoE, 3B active) is the default. R1-Distill
  models are unsuitable as judges: their chat template forces a `<think>`
  block, which often consumes the entire token budget before any JSON gets
  emitted. A non-thinking instruct model is the right tool here.

Two configs are provided:

- `config-frontierscience-deepseek-r1-distill-1p5b-local.yaml` — pilot,
  `max_samples: 20`.
- `config-frontierscience-deepseek-r1-distill-32b-local.yaml` — real run,
  full FrontierScience.

### Servers (vLLM)

Run two vLLM servers, one per role, on different ports.

**Pilot (1.5B distill on port 8000, judge on port 8001):**

The 1.5B distill ships with a broken `tokenizer_class` declaration
(`LlamaTokenizerFast` for what's actually a Qwen2 byte-level BPE), which
makes `AutoTokenizer` pick the wrong class and mangle decoded output. Patch
it once:

```python
# scripts/_patch_distill_tokenizer.py (one-off)
from huggingface_hub import snapshot_download
from pathlib import Path
import json, os
src = Path(snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))
dst = Path.home() / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-fixed"
dst.mkdir(parents=True, exist_ok=True)
for name in os.listdir(src):
    sp, dp = src / name, dst / name
    if dp.exists() or dp.is_symlink(): dp.unlink()
    if name == "tokenizer_config.json":
        cfg = json.loads(sp.read_text())
        cfg["tokenizer_class"] = "Qwen2Tokenizer"
        dp.write_text(json.dumps(cfg, indent=2))
    else:
        os.symlink(sp.resolve(), dp)
```

Then:

```bash
# Trace generator
vllm serve ~/models/DeepSeek-R1-Distill-Qwen-1.5B-fixed \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --reasoning-parser deepseek_r1 \
  --port 8000 --max-model-len 16384

# Judge
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.9
```

**Real run (32B distill, two GPUs):** the R1-Distill-32B claims both GPUs
via tensor-parallel, so the judge runs sequentially. Order:

```bash
# 1) Generate traces
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --reasoning-parser deepseek_r1 \
  --tensor-parallel-size 2 --port 8000 --max-model-len 16384

uv run python experiments/sh6_llm-pairwise-slod/scripts/01_traces.py \
  --config experiments/sh6_llm-pairwise-slod/config-frontierscience-deepseek-r1-distill-32b-local.yaml

# 2) Stop trace server (Ctrl-C), then start judge on port 8001
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 2 --port 8001 --max-model-len 8192

# 3) Re-grade and run remaining stages (script can be re-run; existing
#    correctly-graded traces are kept).
uv run python experiments/sh6_llm-pairwise-slod/scripts/02_slod.py               --config $cfg
uv run python experiments/sh6_llm-pairwise-slod/scripts/03_analyze_accuracy.py   --config $cfg
uv run python experiments/sh6_llm-pairwise-slod/scripts/04_plot_trajectories.py  --config $cfg
uv run python experiments/sh6_llm-pairwise-slod/scripts/05_analyze_failure_modes.py --config $cfg
```

`--reasoning-parser deepseek_r1` is only useful on the trace server; it
makes vLLM split `<think>…</think>` blocks into `message.reasoning_content`.
The judge server should NOT use it — Qwen3-Instruct-2507 doesn't emit
`<think>` blocks anyway.

### Why a non-reasoning judge

The judge calls (pairwise SLoD, grading) need to be a **deterministic
function of inputs**: the same pair of passages should always rank the same
way, and the same answer should always grade the same way. Two requirements
follow:

- `temperature: 0.0` on judge calls (greedy decoding).
- A model that emits JSON directly under guided decoding, rather than
  thinking-then-answering. Reasoning models can satisfy this only if you
  give them a generous token budget, which is wasteful and still has tail
  failures (truncated `<think>`, partial JSON).

Trace generation, by contrast, keeps `temperature: 0.6` (DeepSeek's
recommended R1 setting) because we *want* diverse reasoning trajectories
for the analysis.

### Differences vs the cloud config

| Field | Cloud | Local |
|---|---|---|
| `traces.model.base_url` | `https://openrouter.ai/api/v1` | `http://localhost:8000/v1` |
| `grader.model.name` / `pairwise_slod.model.name` | same as `traces.model` (DeepSeek-V3.2) | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `grader.model.base_url` / `pairwise_slod.model.base_url` | `https://openrouter.ai/api/v1` | `http://localhost:8001/v1` |
| `*.model.backend` | (auto-detected as openrouter) | `local` |
| `*.model.api_key_env` | `OPENROUTER_API_KEY` | `LOCAL_LLM_API_KEY` (optional) |
| `*.model.extra_body` | (omitted) | `temperature`, `top_p`, `max_tokens` |
| `*.max_concurrent` | 20 | 8 (tuned for local throughput) |

Outputs land under
`reports/frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-{1.5B,32B}_reasoning-auto/`,
in parallel to the cloud `deepseek/deepseek-v3.2_reasoning-auto/` run, so
you can compare detector verdicts and `trajectory_shape` AUC side by side.

