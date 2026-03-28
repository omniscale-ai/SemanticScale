# SH5c Design: Per-Question SLoD Consistency Predicts Quality

## Hypothesis

Context-reasoning SLoD alignment predicts answer quality: when the retrieval context's SLoD distribution matches the reasoning chain's SLoD distribution, answers are better. SLoD-routed retrieval conditions produce better alignment than unrouted conditions.

## Method

### Data Summary

- **SH5 Data:** 2000 traces (500 questions x 4 conditions), 6101 steps with SLoD labels
- **SH3 Retrieval Data:** doc_id encodes SLoD level (e.g., `1912.01214__meso__s7_p1`)
- **Critical finding:** chunks_only has zero context diversity (all meso-level chunks), serving as a useful control

### Pipeline Stages

1. **Stage A: Data Loading & Context SLoD Extraction** — Parse SLoD level from retrieved chunks' doc_ids, compute context and reasoning SLoD profiles
2. **Stage B: Alignment Feature Engineering** — Compute 11 alignment features per trace:
   - mean_alignment_gap, max_alignment_gap, JSD, dominant_level_match
   - context_reasoning_correlation, weighted_alignment_gap
   - context_diversity, reasoning_diversity, diversity_ratio
   - soft_mean_gap, soft_jsd
3. **Stage C: Statistical Analysis** — Spearman correlations, paired Wilcoxon tests, logistic regression, partial correlations
4. **Stage D: Subgroup Analysis** — By answer_type, context diversity, and condition
5. **Stage E: Visualization & Report** — Violin plots, scatter plots, heatmaps, ROC curves

### Key Design Decisions

1. Jensen-Shannon divergence between reasoning and context SLoD distributions
2. Both hard-label and soft (probability-weighted) alignment features
3. chunks_only as natural control (zero context diversity)
4. Bonferroni correction for 22 tests (11 features x 2 quality metrics)
5. Three logistic models: alignment-only, jump-only, combined

## Analysis

### H1 — Alignment-Quality Correlation
- Spearman correlation of each alignment feature with token-F1 and attribution-F1
- Criterion: |rho| > 0.10, Bonferroni p < 0.05

### H2 — Cross-Condition Alignment
- Paired Wilcoxon signed-rank tests: SLoD-routed vs baseline conditions
- Criterion: SLoD-routed conditions have significantly lower alignment gap, p < 0.05

### H3 — Predictive Power
- 5-fold stratified cross-validation logistic regression
- Three models: alignment-only, jump-only (SH5 metrics), combined
- Criterion: alignment-only AUROC > 0.60

### Partial Correlations
- Control for n_steps and answer_type

## Exit Criteria

- **Confirmed:** H1 + (H2 or H3)
- **Partial:** Only one hypothesis confirmed
- **Not Confirmed:** None confirmed
