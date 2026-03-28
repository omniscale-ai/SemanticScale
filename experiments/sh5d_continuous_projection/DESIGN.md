# SH5d Design: Probe-Free Embedding Distance as Behavioral Signature

## Hypothesis

Continuous embedding-space metrics (cosine distances between consecutive CoT step embeddings, projections onto a data-driven SLoD axis) capture reasoning coherence better than discrete SLoD labels from a noisy probe. The SLoD axis (macro-to-micro direction in embedding space) explains more variance in answer quality than orthogonal directions.

## Method

### Data Sources
- SH5 CoT step texts (6101 steps), answer scores (2000 traces), jump metrics
- SH1 SciBERT embeddings (37,278 x 768) with macro/meso/micro labels for axis computation

### Pipeline Stages

1. **Stage A: Data Loading** — Load SH5 step records, answer scores, SH1 embeddings
2. **Stage B: SciBERT Embedding** — Embed all 6101 CoT step texts with SciBERT [CLS] (batch_size=32)
3. **Stage C: SLoD Axis Computation** — Compute macro-micro centroid axis and LDA axis from SH1 train split; validate separation on test set
4. **Stage D: Feature Engineering** — 15 features per trace:
   - **Full-embedding features (768-dim):** mean/max cosine distance, path length, displacement, path efficiency
   - **SLoD-axis features (1-dim projection):** mean, variance, range, drift mean/max, direction, monotonicity
   - **Orthogonal-space features (control):** orthogonal drift mean, orthogonal variance, slod_ratio
5. **Stage E: Statistical Analysis** — Spearman correlations (Bonferroni-corrected), ablation by feature group, logistic regression (per-group and combined AUROC)
6. **Stage F: Visualization & Report** — Axis validation histograms, scatter plots, correlation comparison across SH5/5a/5c/5d, ablation charts

### Key Design Decisions

1. Probe-free: no discrete classification step, continuous distances only
2. SLoD axis from SH1 centroid difference (macro vs micro)
3. Orthogonal features as control to test SLoD-specificity
4. Comparison with SH5 (rho=0.003), SH5a (rho=-0.197), SH5c (rho=-0.135) baselines

## Analysis

### H1 — Embedding Drift Predicts Quality
- Spearman correlation of each feature with token-F1 and attribution-F1
- Bonferroni correction for 30 tests
- Criterion: any feature |rho| > 0.10, p < 0.05

### H2 — SLoD Axis Specificity
- Compare max |rho| across feature groups: SLoD-axis vs full-embedding vs orthogonal
- Criterion: SLoD-axis features outperform orthogonal features

### H3 — Improvement over Prior Experiments
- Compare best SH5d feature correlation with SH5a (-0.197) and SH5c (-0.135)
- Criterion: any feature |rho| >= 0.197 or AUROC > 0.60

### H4 — Combined Model
- Combined SH5d + SH5a + SH5c features in logistic regression
- Criterion: AUROC > 0.65

## Exit Criteria

- **Confirmed:** H1 + H2
- **Strong:** Also H3 or H4
- **Partial:** H1 only
- **Not Confirmed:** H1 fails
