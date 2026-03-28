# SH5a Design: Transition Matrix Predicts Reasoning Quality

## Hypothesis

SH5 found that scalar jump rate does NOT correlate with answer token-F1 (rho=0.003, p=0.90), but weakly correlates with attribution F1 (rho=0.092, p=4.1e-5). SH5a decomposes the per-trace SLoD level sequence into a 3x3 transition matrix to test whether specific transition patterns capture signal that scalar metrics missed.

## Method

### Data Summary
- 2000 traces (500 questions x 4 conditions)
- 6101 steps with SLoD labels + probability vectors
- Conditions: chunks_only, naive_hybrid, slod_weighted, slod_weighted_parent
- Steps per trace: mean=3.05, median=3, min=2, max=6

### Pipeline Stages

1. **Stage A: Data Loading** — Load SH5 JSONL files, reconstruct per-trace step sequences, merge into unified trace records
2. **Stage B: Transition Matrix Construction** — Per-trace 3x3 hard and soft transition matrices; per-question aggregated matrices
3. **Stage C: Feature Engineering** — Extract 9 transition cells, self-loop ratio, transition entropy, oscillation index, dominant transition, macro-micro shuttle ratio, upward/downward ratios (both hard and soft versions)
4. **Stage D: Statistical Analysis** — Spearman correlations with Bonferroni correction, K-means clustering, condition comparison (chi-squared), logistic regression
5. **Stage E: Visualization & Report** — Transition heatmaps, correlation bar charts, cluster profiles, scatter plots

### Key Design Decisions

1. No 1-step filtering needed: all traces have min 2 steps (1+ transitions)
2. Soft transitions: outer product of consecutive probability vectors captures uncertainty
3. Per-question aggregation: averaging across 4 conditions gives denser matrices (~8 transitions vs ~2)
4. Bonferroni correction: conservative multiple comparison correction (60 tests)
5. Both per-trace and per-question analysis

## Analysis

### H1 — Transition Signature
- Spearman correlation of each transition feature with answer_token_f1 and attribution_f1
- Bonferroni correction (alpha = 0.05 / 60 = 0.000833)

### H2 — Pattern Clusters
- K-means clustering on flattened 9-element soft transition vectors
- One-way ANOVA on token-F1 and attribution-F1 across clusters

### H3 — Condition Effect
- Chi-squared test: are transition counts independent of condition?
- Pairwise comparisons: routed vs unrouted

### Logistic Regression
- Binary target: above-median token-F1
- L2-regularized logistic regression on all transition features
- Report accuracy, AUC-ROC, top-5 features

### Comparison with SH5
- Best transition feature rho vs jump_rate rho (0.003 for token-F1, 0.092 for attribution-F1)

## Exit Criteria

- **H1 Confirmed:** Any transition feature |rho| > 0.10, Bonferroni p < 0.05
- **H2 Confirmed:** K-means clusters with ANOVA p < 0.05 on mean answer quality
- **H3 Confirmed:** Chi-squared p < 0.05
- **Overall Confirmed:** H1 or H2 passes
- **Partial:** Only H3 passes
- **Not Confirmed:** None pass
