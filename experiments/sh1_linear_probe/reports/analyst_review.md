# SH1 Analyst Review: Linear Decodability of SLoD from Frozen Embeddings

**Analyst:** Claude (automated review)
**Date:** 2026-03-09
**Verdict:** SH1 CONFIRMED, with caveats noted below

---

## 1. Exit Criteria Verification

| Criterion | Threshold | Observed | Pass? | Notes |
|-----------|-----------|----------|-------|-------|
| **Primary:** 3-way macro-F1 on length-matched test | > 0.60 | **0.7200** (SciBERT+LogReg) | YES | Exceeds threshold by 0.12 |
| **Fallback:** Binary macro-vs-micro F1 | > 0.75 | Not run | N/A | Primary met; fallback not needed |
| **Baseline gap:** Best model - random | > 0.15 | **0.3917** (0.7200 - 0.3283) | YES | Large margin; clearly non-trivial |
| **Confound check:** Length-matched F1 within 0.10 of full F1 | |gap| < 0.10 | **0.054** | YES | See caveats in Section 2.2 |
| **Cross-domain:** Noted as limitation | N/A | Documented | YES | QASPER (NLP) only; deferred to SH2/SH3 |

**All exit criteria are met.** The primary criterion is passed by all 6 model-classifier combinations (range: 0.656--0.720), not just the best one. This is a strong result.

---

## 2. Scientific Rigor Assessment

### 2.1 Length Confound Control

The length confound was addressed through two complementary mechanisms:

1. **Length-matched dataset:** Primary evaluation uses 37,278 spans balanced at 12,426 per class with controlled word-count distributions. This is the correct approach.

2. **Word-count-only baseline:** Achieves macro-F1 = 0.2625 on the length-matched test set -- *below* the random baseline of 0.3283. This is a strong indicator that word count alone carries no useful signal on the length-matched data, and may even hurt (the classifier learns a spurious pattern that does not generalize). The per-class breakdown confirms this: the word-count model achieves F1=0.41 for macro, F1=0.37 for meso, and **F1=0.00 for micro** -- it never predicts micro at all (confusion matrix shows all micro predictions go to macro or meso columns).

**Assessment:** The length confound is well controlled on the primary dataset.

### 2.2 Confound Check Methodology -- Caveats

The confound check has two methodological issues worth noting:

**Issue 1: Subsampling instead of full dataset.** The plan called for embedding the full 83K dataset (Phase C), but the runner used a stratified subsample of 9,999 spans (3,333 per class) to save time (~30 min vs ~4 hours). This is a *reasonable pragmatic choice* for an MVE, but introduces two concerns:

- The subsample is *balanced* (3,333 per class), while the full dataset is *imbalanced* (22%/63%/15%). The confound check was designed to test whether the imbalanced class-length correlation inflates performance. A balanced subsample partially neutralizes the very confound being tested.
- With only ~1,500 test spans (15% of 9,999), the F1 estimate has higher variance than the full dataset test set would.

**Issue 2: Hyperparameter mismatch.** The confound check used C=1.0 for LogReg, while the best length-matched model used C=0.01 (selected via validation sweep). This introduces a confound in the comparison: the gap of 0.054 may partly reflect suboptimal C, not genuine domain differences. A fairer comparison would use the same C value.

**Impact assessment:** The gap is 0.054, well under the 0.10 threshold. Even accounting for these issues, it is unlikely the gap would exceed 0.10. The confound check passes, but these caveats should be documented if reporting in a paper.

**Recommendation for paper:** Re-run the confound check on the full 83K dataset with the same C=0.01 and preserving the natural class imbalance. This is important for a published result.

### 2.3 Per-Class F1 Consistency

| Class | Expected Difficulty | SciBERT+LogReg F1 | Consistent? |
|-------|-------------------|-------------------|-------------|
| macro | Easiest | 0.8206 | YES -- highest F1, broad-scope text is distinctive |
| meso | Hardest | 0.6156 | YES -- intermediate scope, overlaps with both extremes |
| micro | Intermediate | 0.7237 | YES -- specific methodological details are identifiable |

The confusion matrix for SciBERT+LogReg confirms the expected pattern:
- macro is misclassified mainly as meso (229/1864 = 12.3%), rarely as micro (91/1864 = 4.9%)
- meso errors split between macro (265) and micro (498) -- the latter being larger, which makes sense since meso/micro boundaries are fuzzier
- micro confuses mainly with meso (383), rarely with macro (90)

This *ordinal* error pattern (most confusion between adjacent SLoD levels) is scientifically meaningful and supports the interpretation that SLoD represents a genuine gradient in text, not just an artifact.

### 2.4 t-SNE Visualization Assessment

**SciBERT t-SNE:** Shows partial but noisy clustering. Macro spans (orange/salmon) tend to cluster in certain regions (left side, some upper patches), while micro (green) concentrates in others. Meso (yellow/light) is dispersed throughout, consistent with its intermediate nature and lower F1. The clusters are *not* cleanly separated, which is realistic for a 72% classifier -- you would expect ~28% of points to be in "wrong" regions. This plot is consistent with the numeric results.

**MiniLM t-SNE:** Shows weaker clustering than SciBERT, consistent with its lower F1 (0.659). The three classes are more intermixed.

**Specter2 t-SNE:** Similar to MiniLM, slightly better structure. Again consistent with its F1 (0.701) being between MiniLM and SciBERT.

**Assessment:** The t-SNE plots are consistent with the quantitative results and show that the signal is real but moderate. They would benefit from being generated with a fixed perplexity value noted in the caption for reproducibility.

### 2.5 Model Ranking

| Rank | Model | Test F1 (LogReg) | Notes |
|------|-------|-------------------|-------|
| 1 | SciBERT | 0.7200 | Domain-specific pre-training helps |
| 2 | Specter2 | 0.7011 | Scientific triplet training helps less than expected |
| 3 | MiniLM | 0.6586 | General-purpose; still above threshold |

The ranking is scientifically sensible: domain-specific models outperform general-purpose ones. The gap between SciBERT and Specter2 (0.019) is small and may not be statistically significant -- worth noting. Specter2 was trained for document-level classification tasks, which may not align well with span-level SLoD classification.

LogReg and SVM perform nearly identically across all models (max difference: 0.005), suggesting the decision boundary is well-captured by either linear method.

### 2.6 Validation-Test Consistency

| Model | Val F1 | Test F1 | Delta |
|-------|--------|---------|-------|
| SciBERT+LogReg | 0.7044 | 0.7200 | +0.016 |
| SciBERT+SVM | 0.7105 | 0.7190 | +0.009 |
| Specter2+LogReg | 0.7017 | 0.7011 | -0.001 |
| MiniLM+LogReg | 0.6536 | 0.6586 | +0.005 |

Test F1 is slightly higher than validation F1 for most configurations. The deltas are small (< 0.02), indicating no overfitting. The fact that test > val is likely just random split variance.

---

## 3. Strengths

1. **Clear positive result.** All 6 model-classifier combinations exceed the 0.60 threshold. This is not a borderline finding.

2. **Strong baseline gap.** The 0.39 gap over random is large. Word-count-only and random-embedding baselines confirm that the signal comes from learned semantic representations, not trivial features.

3. **Ordinal confusion pattern.** Adjacent SLoD levels confuse more than distant ones, supporting construct validity.

4. **Consistent model ranking.** Domain-specific > general-purpose, as expected from prior NLP literature.

5. **Proper methodology.** Train/val/test splits, StandardScaler fit on train only, stratified splits, hyperparameter selection on validation set. No data leakage detected.

6. **Reproducibility.** All splits, embeddings, and results are saved to disk with fixed random seeds.

---

## 4. Weaknesses and Concerns

1. **Confound check methodology.** The balanced subsample does not fully replicate the intended test on the imbalanced full dataset. The C mismatch (1.0 vs 0.01) adds noise. This should be re-run properly before publication.

2. **Single domain.** QASPER covers NLP papers only. Generalization to other scientific domains is unknown. This is explicitly deferred to SH2/SH3 and properly documented.

3. **No confidence intervals or significance tests.** A single train/test split does not give error bars. Bootstrap resampling of the test set (or k-fold cross-validation) would strengthen the claims.

4. **No binary fallback was run.** Since the primary criterion was met, this was correctly skipped per the plan. However, the binary macro-vs-micro F1 would be an easy addition and would provide a useful "upper bound" datapoint.

5. **PCA analysis shows high dimensionality.** The SciBERT PCA plot shows only ~73% variance explained by the first 50 components (of 768). The SLoD-relevant information is distributed across many dimensions, which is normal for frozen embeddings but worth noting.

6. **Best C=0.01 for all models.** This very low regularization value means the probes use nearly all dimensions with minimal shrinkage. It suggests the embeddings have many weakly informative features rather than a few strongly informative ones.

7. **No per-paper analysis.** Spans from the same paper may cluster together for reasons unrelated to SLoD (topic, writing style). Checking whether removing paper identity preserves performance would strengthen the finding.

---

## 5. Recommendations for Paper

### Is this sufficient for MVE-alpha?

**Yes.** The primary exit criterion is met with comfortable margin. The baselines are sound. The confound checks pass. The known limitations are documented. This is sufficient to claim that "frozen transformer embeddings encode linearly decodable SLoD signal" in a paper or internal report.

### Additional analyses before publication

| Priority | Analysis | Effort | Why |
|----------|----------|--------|-----|
| **High** | Re-run confound check on full 83K dataset with C=0.01 | ~4 hours CPU | Addresses methodological caveat in confound check |
| **High** | Bootstrap 95% CI on test F1 (1000 resamples) | ~5 min | Provides error bars for paper |
| **Medium** | Run binary macro-vs-micro fallback | ~2 min | Provides upper-bound reference point |
| **Medium** | Paper-stratified cross-validation | ~30 min | Controls for paper-level confounds |
| **Low** | Per-section error analysis | ~15 min | Identifies systematic failure modes (e.g., abstracts vs methods) |
| **Low** | Add Sentence-BERT (all-mpnet-base-v2) as 4th model | ~15 min | Stronger general-purpose baseline |

### For SH2/SH3

The SH1 results provide a solid foundation. Key next steps:
- Cross-domain validation on S2ORC (biomedical, physics) is the critical gap
- Non-linear probes (MLP) could recover additional signal from the ~28% misclassified spans
- Fine-tuning experiments would establish the ceiling

---

## 6. Summary

SH1 demonstrates that frozen SciBERT embeddings encode SLoD-level information that is linearly decodable with macro-F1 = 0.72, well above the 0.60 threshold and 0.39 above random chance. The result holds across all three embedding models and both classifier types. Length confounds are controlled via a matched dataset, and baselines confirm the signal is semantic rather than surface-level. The main caveats are (a) single-domain evaluation and (b) a methodologically imperfect confound check that should be redone properly. Overall, this is a clean positive result suitable for MVE-alpha.
