# SH3 Design: SLoD-Routed Hierarchical RAG

## Hypothesis

Routing retrieval to the granularity level matching query SLoD improves evidence attribution F1 and QA accuracy vs all fixed-level baselines on QASPER.

## Method

### Overview

Build a retrieval system with three document index levels (macro / meso / micro) corresponding to different granularities of QASPER paper content. A query SLoD classifier (reusing the SH1 SciBERT+LogReg probe) routes each question to the appropriate index level.

### Design Decisions

1. **Retrieval-only evaluation.** No answer generation. Evidence attribution F1 is the primary metric.
2. **MiniLM for retrieval embeddings.** `all-MiniLM-L6-v2` (384-dim) for semantic similarity. SciBERT only for query SLoD classification.
3. **Cosine similarity retrieval.** sklearn `cosine_similarity` -- sufficient for per-paper retrieval (~50 paragraphs).
4. **Per-paper retrieval.** Questions are about specific papers, so retrieval is scoped accordingly.

### Data Preparation

For each QASPER paper, build three document collections:

- **Macro Level:** Abstract + first paragraph of each section (proxy summaries).
- **Meso Level:** Full paragraphs from `full_text.paragraphs`.
- **Micro Level:** Sentence chunks (1-3 sentences, ~50-150 words) with 1-sentence overlap, plus table/figure captions.

### Retrieval Conditions

1. **Chunks-only (Meso baseline):** Search meso-level index only.
2. **Summaries-only (Macro baseline):** Search macro-level index only.
3. **Naive Hybrid:** Search all three levels, rank by cosine similarity.
4. **SLoD-Routed:** Classify query SLoD with SciBERT probe, route to matching index level.

Top-k sweep: k in {1, 3, 5, 10, 20} on validation set.

### Embedding Strategy

- Document embeddings: MiniLM for all three levels, saved as `.npz` per level.
- Query embeddings: MiniLM for retrieval similarity.
- Query SLoD classification: SciBERT + retrained LogReg probe from SH1 data.

## Analysis

### Evidence Attribution F1 (Primary Metric)

- Token-level matching: tokenize retrieved and gold passages, compute token overlap F1.
- A retrieved passage matches a gold paragraph if token-F1 >= 0.5.
- Report macro-averaged F1 across all questions.

### Secondary Metrics

- **Recall@k** -- fraction of gold evidence covered at different k values.
- **MRR** -- rank of first matching retrieved passage.
- **Token cost** -- total tokens retrieved per question (efficiency proxy).

### Statistical Significance

- Paired bootstrap test (10,000 resamples) comparing SLoD-routed vs each baseline.
- Report p-values and 95% confidence intervals.

### Breakdown Analyses

- By predicted SLoD class (macro/meso/micro performance).
- By answer type (extractive, abstractive, yes/no).
- Confusion analysis: when routing hurts, what went wrong?

### Exit Criteria

| Verdict | Condition |
|---|---|
| **Success** | SLoD-routed > all baselines at optimal k, bootstrap p < 0.05 |
| **Partial** | SLoD-routed beats >= 2 of 3 baselines, or improvement not significant |
| **Failure** | SLoD-routed does not beat best fixed-level baseline |

## Code Structure

```
src/
  utils.py             -- Config loading, logging, I/O helpers
  data_prep.py         -- Load QASPER, build 3-level index
  embed.py             -- MiniLM embedding for documents and queries
  query_classifier.py  -- SciBERT+LogReg query SLoD classifier (reuse SH1)
  query_classifier_v2.py -- Alternative classifier implementation
  hyde_classifier.py   -- HyDE-based classification
  retrieve.py          -- Retrieval logic for all 4 conditions
  rerank.py            -- Re-ranking logic
  evaluate.py          -- Attribution F1, recall@k, MRR, bootstrap tests
  analyze.py           -- Breakdown analyses, visualization, report

scripts/
  01_prepare_data.py
  02_embed.py
  02b_embed_specter2.py
  03_classify_queries.py
  03b_train_question_probe.py
  03c_hyde_classify.py
  04_retrieve.py
  05_evaluate.py
  06_analyze.py
```

## Data Dependencies

- QASPER: HuggingFace cache
- SH1 embeddings: `data_sh1/embeddings/scibert_length_matched.npz`
- SH1 splits: `data_sh1/splits.json`
- SH0 spans: `data_sh0/qasper_slod_length_matched.jsonl`

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| SLoD probe has only 0.72 F1 | Report confusion analysis; known limitation |
| Gold evidence granularity mismatch | Use token-F1 matching threshold (0.5) |
| Macro level has few documents per paper | Cap k at available docs |
| Memory pressure from large embedding matrices | Process per-paper during retrieval |
