# SH2 Design: Activation Steering Along the SLoD Axis

## Hypothesis

Activation steering in a generative LLM's residual stream can shift the Semantic Level of Detail (SLoD) of generated answers, as measured by SciBERT embedding projections onto a validated SLoD axis.

- **H1 (SLoD Shift):** Adding a steering vector derived from macro/micro span centroids shifts generated text along the SLoD axis (paired t-test p < 0.05, Cohen's d > 0.5).
- **H2 (Surface Manifestation):** Steered text shows measurable surface changes in entity density, citation density, numeric density, and/or sentence length (>=2 of 4 metrics shift significantly).
- **H3 (Factuality Preservation):** Steering does not substantially degrade answer quality (token-F1 drop < 0.05 absolute vs gold answers).

## Method

### Pipeline Overview

Seven stages executed as Python scripts. Each stage is idempotent.

1. **Environment Setup** (`scripts/01_setup_environment.py`) -- Verify GPU, download models (Mistral-7B-Instruct, SciBERT), validate data files.
2. **Compute SLoD Evaluation Axis** (`scripts/02_compute_slod_axis.py`) -- Reuse proven `src/slod_axis.py` from SH5d. Compute centroid and LDA axes from train split, validate on test split (expect Cohen's d ~2.65).
3. **Compute Steering Vector** (`scripts/03_compute_steering_vector.py`) -- Extract per-layer hidden states for macro/micro spans, compute `steering_vector[L] = normalize(micro_centroid[L] - macro_centroid[L])`, select best layer via SLoD shift on validation questions.
4. **Generate Baseline Answers** (`scripts/04_generate_baseline.py`) -- 500 QASPER questions, no steering, temperature=0.
5. **Generate Steered Answers** (`scripts/05_generate_steered.py`) -- Same questions with residual stream hook adding `alpha * direction * steering_vector`. Alpha selected via validation sweep. Both micro (add) and macro (subtract) directions.
6. **Evaluate** (`scripts/06_evaluate.py`) -- SLoD shift measurement, surface metrics, factuality (token-F1) preservation.
7. **Report** (`scripts/07_report.py`) -- Figures and auto-generated markdown report.

### Steering Hook Implementation

```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    hidden_states = hidden_states + alpha * direction_sign * steering_vector
    return (hidden_states,) + output[1:]

handle = model.model.layers[selected_layer].register_forward_hook(steering_hook)
```

### Layer Selection

Test layers at fractional depths (e.g., layers 8, 16, 24 for 32-layer model). Select the layer producing the largest absolute SLoD shift on validation questions with alpha=1.0.

## Analysis

### SLoD Shift (H1)
- Embed all answers with SciBERT, project onto evaluation axis.
- Compute per-question delta_SLoD = steered - baseline.
- Statistical test: paired t-test, effect size: Cohen's d.

### Surface Metrics (H2)
- Entity density, citation density, numeric density, mean sentence length.
- Paired t-tests for steered_micro vs baseline and steered_macro vs baseline.

### Factuality Preservation (H3)
- Token-F1 of each answer vs gold answer text.
- Criterion: mean drop < 0.05 absolute.

### Exit Criteria

| Verdict | Condition |
|---|---|
| **CONFIRMED** | H1 + H3 pass |
| **PARTIAL** | H1 passes, H3 fails |
| **NOT CONFIRMED** | H1 fails |

## Code Structure

```
src/
  utils.py           -- Config loader, JSONL I/O, label constants
  embedding.py       -- SciBERT [CLS] embedding
  slod_axis.py       -- SLoD axis computation + validation
  steering.py        -- Activation steering: hooks, vector computation, generation
  evaluate.py        -- SLoD shift measurement, surface metrics, token-F1
  visualization.py   -- All plotting functions

scripts/
  01_setup_environment.py
  02_compute_slod_axis.py
  03_compute_steering_vector.py
  04_generate_baseline.py
  05_generate_steered.py
  06_evaluate.py
  07_report.py
```

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| Model too large for GPU | Use 4-bit quantization |
| Steering causes degenerate output | Cap alpha; add perplexity monitoring |
| SciBERT doesn't capture LLM-generated text SLoD | Validate by embedding baseline answers |
| Layer selection is model-dependent | Test multiple layers systematically |
