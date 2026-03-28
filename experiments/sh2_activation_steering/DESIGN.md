# SH2 -- Activation Steering Along the SLoD Axis: Design

## Hypothesis

Activation steering in a generative LLM's residual stream can shift the semantic level
of detail (SLoD) of generated answers along the macro-micro axis, as measured by SciBERT
embedding projections, while preserving factual accuracy.

### Sub-Hypotheses

- **H1 (SLoD Shift):** Adding a steering vector computed from macro/micro span centroids
  to a generative model's hidden states produces measurable SLoD shift (paired t-test
  p < 0.05, Cohen's d > 0.5).
- **H2 (Surface Manifestation):** Steered outputs show changes in at least 2 of 4 surface
  metrics (entity density, citation density, numeric density, mean sentence length).
- **H3 (Factuality Preservation):** Token-F1 vs gold answers drops by less than 0.05
  absolute after steering.

## Method

### Pipeline (7 stages)

**Stage A: Environment Setup** -- Verify GPU, download generative model
(Mistral-7B-Instruct-v0.3, 4-bit if VRAM < 16 GB), validate SH0/SH1 data files.

**Stage B: Compute SLoD Evaluation Axis** -- From SH1 train split SciBERT embeddings,
compute centroid axis `normalize(micro_centroid - macro_centroid)` and LDA axis. Validate
on test split (expect Cohen's d ~2.65).

**Stage C: Compute Steering Vector** -- Extract hidden states from generative model for
macro and micro spans. Compute per-layer steering vector as
`normalize(micro_centroid - macro_centroid)` in generative model's representation space.
Select best layer via validation on subset of questions.

**Stage D: Generate Baseline Answers** -- 500 QASPER questions, no steering, temperature=0.

**Stage E: Generate Steered Answers** -- Same questions with activation steering hook on
selected layer. Sweep alpha values, select alpha with largest SLoD shift without quality
collapse. Generate both micro (add vector) and macro (subtract vector) directions.

**Stage F: Evaluate** -- Embed all answers with SciBERT, project onto evaluation axis.
Compute SLoD shift, surface metrics, and token-F1 vs gold answers.

**Stage G: Report** -- Generate figures and markdown report with verdict.

### Steering Hook Implementation

```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    hidden_states = hidden_states + alpha * direction_sign * steering_vector
    return (hidden_states,) + output[1:]
```

Registered on the target layer's residual stream via `register_forward_hook`.

## Analysis / Exit Criteria

| Hypothesis | Criterion | Metric |
|---|---|---|
| H1 (SLoD Shift) | Paired t-test p < 0.05 AND Cohen's d > 0.5 | delta_SLoD (SciBERT projection) |
| H2 (Surface) | >= 2 of 4 surface metrics p < 0.05 | Entity/citation/numeric density, sentence length |
| H3 (Factuality) | Token-F1 drop < 0.05 absolute | Token-F1 vs gold answers |

| Verdict | Condition |
|---|---|
| CONFIRMED | H1 + H3 pass |
| PARTIAL | H1 passes, H3 fails |
| NOT CONFIRMED | H1 fails |
