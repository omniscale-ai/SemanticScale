# SH5a Conclusive Report: Transition Matrix Analysis — Structured Oscillation Predicts Quality

## Executive Summary

**Verdict: CONFIRMED (H1 + H2 pass; H3 fails)**

SH5a decomposed per-trace SLoD level sequences into 3x3 transition matrices and tested whether specific transition patterns  ⏫📅2026-03-17
- [ ] correlate with answer quality. The analysis reveals that transition matrix features contain significantly more signal than SH5's scalar jump rate, particularly for attribution F1.

Key findings:
- **20 transition features** achieve Bonferroni-corrected significance (out of 60 comparisons)
- The strongest predictor is **soft macro->macro self-transition** (rho = -0.197, p = 5.0e-19 for attribution-F1), meaning traces that stay stuck at macro level produce worse evidence citations
- The best token-F1 predictor is also **soft macro->macro** (rho = -0.108, p = 1.3e-6) — 36x stronger than SH5's jump rate (rho = 0.003)
- K-means clustering reveals **2 reasoning styles** with significantly different attribution quality (ANOVA p = 1.9e-8)
- SLoD routing conditions do NOT produce detectably different transition distributions (chi2 p = 0.989)

## Experimental Setup

### Data
- **2000 CoT traces** (500 questions x 4 retrieval conditions) from SH5
- **6101 reasoning steps** with SLoD labels (macro/meso/micro) + probability vectors
- No new API calls — pure reanalysis of existing data

### Method
1. **Transition matrices**: For each trace, built a 3x3 matrix counting transitions between SLoD levels. Two variants:
   - *Hard*: based on argmax label assignments
   - *Soft*: outer product of consecutive probability vectors (captures classifier uncertainty)
2. **Feature extraction**: 9 transition cells + 6 derived features (entropy, self-loop ratio, oscillation index, macro-micro shuttle, upward/downward ratio) x 2 variants = 30 features
3. **Statistical tests**: Spearman correlation (Bonferroni-corrected for 60 comparisons), K-means clustering with ANOVA, chi-squared condition comparison, logistic regression

### Trace characteristics
- Mean 3.05 steps per trace (median 3, min 2, max 6)
- Mean 2.05 transitions per trace
- All traces have at least 1 transition (no filtering needed)
- 95.2% of traces have exactly 3 steps (2 transitions)

## Results

### H1: Transition Signature — CONFIRMED

20 of 60 feature-target pairs achieve Bonferroni-corrected significance (alpha = 0.000833).

#### Top Correlations with Attribution F1 (N=2000)

| Feature | rho | p-value | Interpretation |
|---------|-----|---------|----------------|
| soft_macro->macro | -0.197 | 5.0e-19 | Staying at macro level hurts attribution |
| hard_macro->macro | -0.188 | 2.7e-17 | Same finding with hard labels |
| soft_micro->meso | +0.167 | 6.7e-14 | Upward transitions from micro help attribution |
| soft_oscillation_index | +0.133 | 2.5e-9 | More diverse transitions help |
| soft_meso->micro | +0.124 | 2.8e-8 | Drilling down from meso helps |

#### Top Correlations with Token F1 (N=2000)

| Feature | rho | p-value | Interpretation |
|---------|-----|---------|----------------|
| soft_macro->macro | -0.108 | 1.3e-6 | Macro self-loops hurt answer quality |
| hard_macro->macro | -0.087 | 1.1e-4 | Same with hard labels |
| soft_micro->meso | +0.085 | 1.6e-4 | Micro-to-meso transitions help |
| hard_meso->meso | +0.083 | 2.1e-4 | Meso stability helps |
| soft_macro->micro | -0.075 | 7.5e-4 | Jumping directly macro->micro hurts |

#### Per-Question Aggregated Analysis (N=500)

3 features remain significant after Bonferroni correction when aggregating across conditions:
- hard_macro->macro vs mean_attribution_f1: rho = -0.249, p = 1.7e-8
- soft_macro->macro vs mean_attribution_f1: rho = -0.238, p = 7.4e-8
- soft_micro->meso vs mean_attribution_f1: rho = +0.198, p = 7.9e-6

The per-question effect sizes are **stronger** than per-trace, suggesting within-question consistency.

### H2: Pattern Clusters — CONFIRMED (for attribution F1 only)

K-means clustering on soft transition vectors (k=2, silhouette=0.293):

| Cluster | N | Mean Token-F1 | Mean Attribution-F1 | Profile |
|---------|---|---------------|---------------------|---------|
| 0 | 1020 | 0.279 | 0.279 | Lower macro self-loop, more micro/meso activity |
| 1 | 980 | 0.262 | 0.218 | Higher macro self-loop, less micro/meso activity |

- **ANOVA on attribution-F1: F=31.8, p=1.9e-8** (highly significant)
- ANOVA on token-F1: F=1.8, p=0.176 (not significant)

The two clusters distinguish "exploratory" reasoning (more diverse transitions, better attribution) from "macro-stuck" reasoning (dominated by macro self-loops, worse attribution).

### H3: Condition Effect — NOT CONFIRMED

- Chi-squared (4 conditions x 9 cells): chi2 = 10.95, p = 0.989
- Routed vs unrouted: chi2 = 0.27, p = 1.000

The four retrieval conditions produce indistinguishable transition distributions. SLoD routing does not measurably alter the reasoning trajectory pattern — the model follows similar transition paths regardless of how evidence is retrieved.

### Logistic Regression

Predicting above-median quality from all 30 transition features:

| Target | CV Accuracy | CV AUC-ROC |
|--------|-------------|------------|
| Token-F1 | 0.555 | 0.574 |
| Attribution-F1 | 0.620 | 0.603 |

Token-F1 prediction is barely above chance (0.50), but attribution-F1 prediction achieves AUC 0.603 — modest but consistent with the correlation findings.

## Comparison with SH5

| Metric | SH5 (scalar jump rate) | SH5a (best transition feature) | Improvement |
|--------|------------------------|--------------------------------|-------------|
| Token-F1 rho | +0.003 (p=0.90) | -0.108 (p=1.3e-6) | 36x stronger |
| Attribution-F1 rho | +0.092 (p=4.1e-5) | -0.197 (p=5.0e-19) | 2.1x stronger |

The transition matrix representation captures substantially more signal than scalar jump rate:
- For **token-F1**: SH5 found zero signal; SH5a finds a weak but significant negative correlation with macro self-loops
- For **attribution-F1**: SH5 found rho=0.092; SH5a finds rho=-0.197 (opposite sign because macro self-loops *reduce* rather than increase the quality, while SH5's jump rate captures a different aspect)

## Interpretation

### The Macro Self-Loop Story

The dominant finding is that **staying at macro level hurts quality**. The soft_macro->macro transition probability is the single strongest predictor of both answer quality and attribution quality (negative correlation). This makes physical sense:

- **Macro-level reasoning** corresponds to high-level summarization and surface-level claims
- **Traces stuck at macro** never drill into evidence details, so they produce less grounded answers
- **Traces with transitions to micro and meso** engage with specific evidence passages and intermediate reasoning, producing better-attributed answers

### Why Transitions Beat Scalar Jump Rate

SH5's jump rate collapses all transitions into a single number. SH5a reveals that:
1. **Self-loops (diagonal) are qualitatively different from transitions (off-diagonal)** — specifically, macro self-loops are harmful while meso self-loops are beneficial
2. **Direction matters** — micro->meso transitions (upward synthesis) correlate positively with quality, while macro->micro (jumping past meso) correlates negatively
3. **The probability-weighted (soft) representation** consistently outperforms hard labels, suggesting that transition uncertainty carries additional signal

### Why Conditions Are Indistinguishable

H3 fails completely (p=0.989), indicating that the retrieval routing strategy has no effect on reasoning trajectory patterns. The model's SLoD-level transitions are determined by the question and its internal reasoning dynamics, not by how evidence was retrieved. This is consistent with SH5's finding that routing conditions had minimal effect on quality.

## Limitations

1. **Short traces**: 95% of traces have exactly 3 steps (2 transitions), yielding very sparse per-trace matrices. The soft transition approach mitigates this but cannot fully compensate.
2. **Effect sizes are modest**: The strongest correlation (rho=-0.197) explains only ~4% of variance in attribution-F1. Transition patterns are one signal among many.
3. **Logistic regression barely beats chance for token-F1** (AUC 0.574), suggesting transition features are insufficient for practical answer quality prediction.
4. **Correlation, not causation**: Higher macro self-loop rates may be a symptom of harder questions rather than a cause of lower quality.
5. **Single model**: All traces come from one LLM (GPT-4o-mini). The findings may not generalize.

## Implications for SLoD Research

1. **Transition matrices are a richer representation** of reasoning dynamics than scalar summaries. Future SLoD analyses should track the full transition structure.
2. **"Macro-stuck" reasoning is a detectable failure mode** — traces that never leave the macro level produce worse evidence attribution. A monitoring system could flag such traces.
3. **Meso-level engagement matters** — the most beneficial transitions involve the meso level (micro->meso synthesis, meso->micro drilling), suggesting that intermediate-level reasoning is the key connector.
4. **Routing strategy alone cannot fix trajectory patterns** — different retrieval methods do not change how the model navigates between reasoning levels. Improving reasoning trajectories may require prompt engineering or training interventions.

## Artifacts

| Artifact | Location |
|----------|----------|
| Unified traces | `data/traces.jsonl` (2000 records) |
| Transition matrices | `data/transition_matrices.npz` (2000 + 500 aggregated) |
| Feature vectors | `data/features.jsonl`, `data/features_agg.jsonl` |
| Correlation results | `data/results/correlation_results.json` |
| Clustering results | `data/results/clustering_results.json` |
| Condition comparison | `data/results/condition_comparison.json` |
| Logistic regression | `data/results/logistic_results.json` |
| Summary | `data/results/summary.json` |
| Condition heatmaps | `reports/figures/condition_heatmaps_soft.png`, `_hard.png` |
| Correlation chart | `reports/figures/correlation_bars.png` |
| Cluster profiles | `reports/figures/cluster_profiles_soft.png` |
| Cluster quality | `reports/figures/cluster_quality_boxplot.png` |
| SH5 comparison | `reports/figures/sh5_comparison.png` |
| Auto-report | `reports/SH5a_auto_report.md` |
| Analysis plan | `PLAN.md` |
| Configuration | `config.yaml` |
