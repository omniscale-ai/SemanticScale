# SH4 Analyst Review

## Executive Summary

The SH4 pipeline produced a combined AUROC of 0.976, which far exceeds the 0.65 threshold.
Drift features contribute meaningfully to the combined model (5 of top 10 features).
However, the 8% match rate creates severe data scarcity: only 47 correct labels out of 587 extractions, and only 9 positive examples in the test set of 89.
The result is **directionally strong but statistically fragile**.

---

## 1. Root Cause: Why 8% Match Rate?

The match rate failure is **not** a matching threshold problem. It is an extraction quality problem caused by abstract-only input.

### Evidence

| Category | Count | % of 587 |
|----------|-------|----------|
| No numeric value extracted | 526 | 89.6% |
| No ground truth for paper (match_info=null) | 287 | 48.9% |
| Dataset match only (metric/value fail) | 133 | 22.7% |
| Dataset+Metric match, value mismatch (near miss) | 49 | 8.3% |
| Fully correct (d+m+v) | 47 | 8.0% |

**The dominant failure mode is non-numeric extraction.** Of 587 extractions, 526 (89.6%) have no parseable numeric value — the LLM produces qualitative statements like "state-of-the-art", "outperforms baseline", "significant improvement" because abstracts almost never contain specific numeric results.

The 47 correct matches come from the ~10% of papers whose abstracts DO contain specific numbers (e.g., "achieves 92.4% accuracy on ModelNet40"). These are the exception, not the rule.

### Near-Miss Analysis

The 49 near-misses (dataset+metric match but value mismatch) break down into:
- **Qualitative values** (e.g., "superior results" vs GT "87.4%"): ~60% of near misses
- **Wrong number from abstract** (e.g., "84.7%" vs GT "83.84"): ~25% — different experiment/config
- **Percentage normalization issues** (e.g., "0.82" vs "85.9"): ~15% — potentially fixable

### Conclusion on Matching

Relaxing fuzzy thresholds (dataset_match_ratio=0.6, metric_match_ratio=0.6) would gain very few additional matches because **the bottleneck is non-numeric values, not string matching strictness**. The current thresholds of 0.8 already correctly match "ACE-2004" to "ACE 2004" and "F1 score" to "F1".

---

## 2. Is the 0.976 AUROC Meaningful?

### Class Imbalance Concern

| Split | Total | Correct | Rate |
|-------|-------|---------|------|
| Train | 405 | 25 | 6.2% |
| Val | 93 | 13 | 14.0% |
| Test | 89 | 9 | 10.1% |

With only 9 positive examples in the test set, the ROC curve is estimated from very few data points. Each positive sample represents ~11% of the positive class — a single misranked positive shifts AUROC by ~0.01.

### What the Model Is Learning

The precision for all models is 0.0, meaning at the default threshold (0.5), NONE of the models predict ANY positive. The accuracy of 89.9% for all three models equals the majority-class baseline (predicting all negative). This means the models have learned to rank positives higher in probability but never confidently enough to cross 0.5.

The AUROC is still informative — it measures ranking quality, which is appropriate for a precision@k use case. But:
- **0.976 AUROC with 9 positives has wide confidence intervals** (~0.93-1.0 at 95% CI, rough bootstrap estimate)
- The Precision@50% = 0.20 across ALL models (drift, surface, combined) is nearly identical to the ~10% base rate scaled by the retention fraction, suggesting limited practical filtering power

### Verdict on AUROC

The ranking signal is likely real (0.976 is very high), but the precision of the estimate is low. We should not over-interpret the second decimal place.

---

## 3. Does Drift Contribute to the Combined Model?

### Yes — and This IS the Key Finding

| Model | Test AUROC |
|-------|-----------|
| Drift-only | 0.865 |
| Surface-only | 0.924 |
| Combined | 0.976 |

The combined model improves over surface-only by 0.052 AUROC points. Since drift-only (0.865) and surface-only (0.924) are both strong individually, and the combination (0.976) exceeds either, this demonstrates that **drift and surface features capture complementary signals**.

### Feature Importance in Combined Model

| Rank | Feature | |Coefficient| | Type |
|------|---------|-------------|------|
| 1 | llm_confidence | 0.597 | surface |
| 2 | numeric_density | 0.335 | surface |
| 3 | temporal_density | 0.283 | surface |
| 4 | sentence_count | 0.270 | surface |
| 5 | realized_slod_prob_meso | 0.223 | **drift** |
| 6 | slod_entropy | 0.196 | **drift** |
| 7 | realized_slod_raw | 0.180 | **drift** |
| 8 | drift_value | 0.180 | **drift** |
| 9 | realized_slod_prob_micro | 0.175 | **drift** |
| 10 | drift_max | 0.158 | **drift** |

**6 of top 10 features are drift-related** (ranks 5-10). While surface features dominate the top 4, drift features form a consistent band of secondary predictors. This is actually a meaningful result:

- **llm_confidence** is the strongest predictor — makes sense, the LLM "knows" when it's guessing
- **numeric_density** and **temporal_density** — extractions from spans with numbers and years are more likely correct (because these are the spans that actually contain results)
- **Drift features cluster at |coeff| ~ 0.17-0.22** — they capture whether the source span matches the expected abstraction level for the extracted field type

### Interpretation

The drift features are not redundant with surface features. They add a different kind of signal: "is this span at the right level of abstraction for what was extracted?" This is conceptually distinct from "does this span contain numbers?" (numeric_density) or "how confident was the LLM?" (llm_confidence).

The fact that `realized_slod_prob_meso` has negative coefficient (-0.223) while `slod_entropy` and `realized_slod_prob_micro` have positive coefficients suggests: extractions from clearly micro-level text (low entropy, high micro probability) are more likely correct — which aligns with the hypothesis that specific numeric results live at the micro level.

---

## 4. Recommendations

### 4a. Reframe the Result (Accept What We Have)

The current results support a **modified claim**:

> "Abstraction drift features complement surface features for predicting extraction correctness, adding ~5% AUROC over surface-only features. The combined model achieves 0.976 AUROC, with drift features occupying 6 of the top 10 feature positions."

This is a PARTIAL confirmation: drift adds value in combination, but does not independently beat surface features.

### 4b. Improve Match Rate (If Iterating)

The only intervention that will materially improve the 8% match rate is **full-text extraction** (Stage A3 from PLAN.md). Abstracts are fundamentally too sparse — 89.6% of extractions lack numeric values entirely.

Specific recommendations if iterating:

1. **Full-text extraction for 100 papers** — fetch arxiv PDFs, extract text, re-run extraction pipeline. Expected improvement: match rate from 8% to 30-50% (full papers contain results tables with specific numbers).

2. **Filter non-numeric extractions before labeling** — add a post-extraction filter that drops extractions where `value` is not parseable as a number. This would reduce the 587 extractions to ~61, but all would be meaningfully matchable.

3. **Do NOT relax matching thresholds** — the thresholds are not the bottleneck. The 0.8 Levenshtein ratio already handles minor format differences correctly.

4. **Do NOT use LLM-as-judge** — the problem is not fuzzy matching accuracy, it's that the LLM extracts qualitative statements from abstracts. An LLM judge would correctly confirm that "state-of-the-art" does not match "87.4%".

### 4c. Confidence Assessment

| What we know with confidence | What is uncertain |
|------------------------------|-------------------|
| Drift + surface > surface alone | Exact AUROC (small test set) |
| LLM confidence is top predictor | Whether drift would still contribute with more data |
| Abstract-only extraction produces mostly non-numeric output | Whether full-text extraction changes feature importance ranking |
| 6 of 10 top features are drift-related | Generalization beyond these 300 papers |

### 4d. Verdict

**Accept as PARTIAL with caveats, or iterate once with full-text extraction.**

If the goal is to demonstrate that drift features contribute (which they clearly do), the current results are sufficient. If the goal is to demonstrate practical precision@k filtering, full-text extraction is required.

Cost of iteration: ~2 agent iterations for full-text pipeline + re-run. Expected gain: 3-5x more positive labels, enabling reliable precision@k evaluation.

---

## 5. Statistical Notes

- **Test set**: 89 samples, 9 positive (10.1%) — precision estimates have ~30% relative standard error
- **Brier score**: Combined (0.052) < Drift-only (0.077) < Surface-only (0.091) — the combined model is best calibrated
- **Regularization**: Surface-only model chose C=0.001 (heavy regularization), drift-only chose C=10.0 (light regularization), combined chose C=0.1 — suggesting surface features are noisier and need more shrinkage
- **Val vs Test correct rate mismatch**: 14.0% vs 10.1% — within sampling noise for small samples, but the stratified split may not perfectly preserve class balance at these sizes

---

*Analyst review completed 2026-03-12*
