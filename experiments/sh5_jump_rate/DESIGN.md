# SH5 Design: Jump Rate as Behavioral Signature

## Hypothesis

Lower abstraction-level jump rate in CoT steps (tagged with SH1 probe post-hoc) correlates with higher QA/extraction correctness. Systems using SLoD routing (SH3) show fewer jumps.

SH5 closes the SLoD argument loop: mechanism (SH1) -> utility (SH3/SH4) -> behavioral signature (SH5). We generate chain-of-thought reasoning traces for QASPER questions across multiple SH3 retrieval conditions, tag each reasoning step with its SLoD level using the SH1 probe, compute jump metrics, and correlate those metrics with answer correctness.

## Method

### Data Sources

- **From SH3:** 1352 test questions (`data_sh3/questions.jsonl`), retrieval results at k=5 for 4 conditions (chunks_only, naive_hybrid, slod_weighted, slod_weighted_parent), evaluation metrics
- **From SH1:** SciBERT embeddings (37,278 spans x 768 dims), splits, probe results (SciBERT+LogReg, macro-F1=0.72)
- **From QASPER:** Gold answers (extractive_spans, free_form_answer, yes_no)

### Pipeline Stages

1. **Stage A: Sample Selection** — Stratified sample of 500 test questions, extract gold answers from QASPER
2. **Stage B: CoT Generation** — Generate step-by-step reasoning traces via Claude Haiku for each question x condition (2000 API calls)
3. **Stage C: SLoD Tagging** — Retrain SciBERT+LogReg probe from SH1, embed CoT steps with SciBERT, classify each step as macro/meso/micro
4. **Stage D: Jump Metrics** — Compute per-trace metrics: jump_count, mean_abs_delta, max_jump, slod_variance, direction_changes, normalized_jump_rate
5. **Stage E: Answer Correctness** — Token-F1 against gold answers, per-question attribution F1 from SH3 retrieval
6. **Stage F: Correlation Analysis** — Spearman correlations, Wilcoxon tests, multiple regression
7. **Stage G: Visualization** — Box plots, scatter plots, heatmaps, report generation

### Key Design Decisions

1. Reuse SH3 data entirely for retrieval results and questions
2. Claude Haiku for CoT generation (cost-effective, ~$0.40 total)
3. SciBERT+LogReg probe from SH1 (macro-F1=0.72)
4. Token-F1 for answer correctness (objective, reproducible)
5. 4 conditions at k=5 (2 baselines + 2 SLoD-routed)
6. 500 test questions (power > 0.80 for Spearman rho = -0.10)

## Analysis

### Hypothesis 1: Jump rate correlates negatively with correctness
- Spearman rank correlation of normalized_jump_rate with answer_token_f1 and attribution_f1
- 95% confidence intervals via bootstrap (10,000 resamples)
- Primary test: Spearman rho < -0.10, p < 0.05

### Hypothesis 2: SLoD-routed conditions show lower jump rates
- Paired Wilcoxon signed-rank tests: slod_weighted vs chunks_only, slod_weighted vs naive_hybrid
- Effect size (Cohen's d or rank-biserial correlation)
- Primary test: mean normalized_jump_rate lower for SLoD-routed, p < 0.05

### Additional Analyses
- Multiple regression: answer_token_f1 ~ normalized_jump_rate + condition + answer_type
- Mediation analysis (exploratory): condition -> jump_rate -> correctness
- By answer_type breakdown

## Exit Criteria

- **Confirmed:** Both correlation (rho < -0.10, p < 0.05) and condition difference (Wilcoxon p < 0.05) pass
- **Partial:** Only one criterion met
- **Not Confirmed:** Neither criterion met
