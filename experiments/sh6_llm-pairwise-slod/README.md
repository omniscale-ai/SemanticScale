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
regression with class balancing.

This is intentionally simple. The goal is to answer a falsifiable question:
do trajectory-derived features predict success/failure beyond trivial
baselines?

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
- `failure_prediction_summary.json`
  Machine-readable metrics and feature rankings.
- `failure_prediction.md`
  Human-readable report.
- `failure_prediction_roc.png`
  Cross-validated ROC curves for the feature-set baselines.
- `failure_feature_coefficients.png`
  Top coefficients from the full logistic model.

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
