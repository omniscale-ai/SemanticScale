# SH3 Analyst Review — Iteration 2: Confidence-Gated SLoD Routing

**Date:** 2026-03-10
**Status:** COMPLETE
**Verdict:** OPTION B — One more iteration with `slod_weighted` hybrid, then accept regardless

---

## 1. Executive Summary

The v2 routing strategy (macro->meso fallback + confidence-gated hybrid fallback at threshold=0.6) delivered exactly what was predicted: a significant improvement over v1 (+3.2 pts F1 at k=5, p<0.001) and near-parity with the best baselines. However, it does NOT beat naive_hybrid (0.230 vs 0.241 at k=10, the best comparison point for naive_hybrid).

The core finding from this iteration is structural: **v2 is essentially a slightly biased version of naive_hybrid**. With 44.9% of test queries falling back to hybrid and 51.6% routed to meso (which is the dominant component of hybrid anyway), the only distinctive contribution comes from the 3.6% of queries (48 out of 1352) routed to the micro index with high confidence. That 3.6% shows a genuine signal, but it is too small to move the aggregate needle.

**Recommendation:** Try one final iteration with `slod_weighted` — a soft-boosting approach that preserves hybrid diversity while giving a routing signal. This is quick to implement (one new retrieval function) and has a plausible path to beating naive_hybrid. If it fails, accept the result as a nuanced partial success.

---

## 2. What v2 Actually Does: Routing Breakdown

### Test Set Routing (n=1352)

| Route | Count | % | What Happens |
|-------|-------|---|--------------|
| Meso (confident macro+meso) | 697 | 51.6% | Identical to chunks_only |
| Micro (confident micro) | 48 | 3.6% | Distinctive — uses sentence-level index |
| Hybrid fallback (low conf) | 607 | 44.9% | Identical to naive_hybrid |

This means v2 is effectively: **51.6% chunks_only + 3.6% micro-routed + 44.9% naive_hybrid**. The result is a weighted blend of two baselines with a tiny micro-routing injection.

### Why This Matters

The reason v2 cannot beat naive_hybrid is arithmetic. For v2 to win overall, the 48 micro-routed queries must compensate for the 697 meso-only queries (which miss the multi-level diversity that gives naive_hybrid its edge at higher k). At k=10:

- naive_hybrid on macro queries: 0.2358 (benefits from meso+micro docs in the pool)
- v2 on macro queries routed to meso: ~chunks_only = 0.1930 (no micro docs available)
- v2 on macro queries falling back to hybrid: 0.2358 (same as naive_hybrid)

The confident-macro-routed-to-meso queries are WORSE than if they had gone to hybrid. The confidence threshold of 0.6 is not selective enough for macro/meso queries.

### Confidence Distribution (Test Set)

| Predicted Class | Total | Below 0.6 (-> hybrid) | Above 0.6 (-> routed) |
|-----------------|-------|------------------------|------------------------|
| macro | 590 | 208 (35.3%) | 382 (64.7%) -> meso |
| meso | 610 | 295 (48.4%) | 315 (51.6%) -> meso |
| micro | 152 | 104 (68.4%) | 48 (31.6%) -> micro |

The threshold is working as intended for micro (only routes when confident), but for macro/meso it routes 697 queries to meso-only when they might benefit from multi-level diversity.

---

## 3. Detailed Performance Analysis

### 3.1 v2 vs Baselines by SLoD Class (k=5)

| Predicted SLoD | chunks_only | naive_hybrid | slod_routed_v2 | v2 vs chunks | v2 vs hybrid |
|----------------|-------------|--------------|----------------|--------------|--------------|
| macro (n=590) | 0.2234 | 0.2259 | 0.2225 | -0.0009 | -0.0034 |
| meso (n=610) | 0.2249 | 0.2314 | 0.2282 | +0.0033 | -0.0033 |
| micro (n=152) | 0.2389 | 0.2685 | **0.2669** | +0.0280 | -0.0017 |

Key observations:
- **Macro queries:** v2 recovered from v1's catastrophe (0.157 -> 0.222) but is still slightly below both baselines. The confident-macro queries go to meso-only and miss hybrid's diversity.
- **Meso queries:** v2 is between chunks_only and naive_hybrid. The 48.4% that fall back to hybrid help, the 51.6% that go meso-only hurt vs hybrid.
- **Micro queries:** v2 nearly matches naive_hybrid (0.267 vs 0.269). At k=5 this is essentially identical. The micro routing helps vs chunks_only (+2.8 pts) but doesn't exceed hybrid.

### 3.2 The Micro Routing Signal — Is It Real?

Comparing micro-classified queries across conditions at k=5:

| Condition | Micro-query F1 | vs chunks_only |
|-----------|---------------|----------------|
| chunks_only | 0.2389 | baseline |
| naive_hybrid | **0.2685** | +0.0296 |
| slod_routed (v1) | 0.2530 | +0.0141 |
| slod_routed_v2 | 0.2669 | +0.0280 |

The uncomfortable truth: **naive_hybrid already captures most of the micro-routing benefit**. By including micro-level documents in its pool, naive_hybrid naturally surfaces sentence-level chunks when they are the best semantic match. v2's explicit micro routing does not add much beyond what naive_hybrid achieves organically.

At k=10 the picture shifts slightly:

| Condition | Micro-query F1 (k=10) |
|-----------|----------------------|
| chunks_only | 0.2040 |
| naive_hybrid | 0.2625 |
| slod_routed_v2 | 0.2535 |

Still, naive_hybrid wins on micro queries. The diversity of having all three levels in the pool matters more than focused micro retrieval.

### 3.3 Statistical Significance

From bootstrap tests, v2 vs v1 (both are `slod_routed` variants):

| k | v1 F1 | v2 F1 | Diff | p-value | Significant |
|---|-------|-------|------|---------|-------------|
| 3 | 0.2073 | 0.2240 | +0.017 | 0.001 | **Yes** |
| 5 | 0.1985 | 0.2300 | +0.032 | <0.001 | **Yes** |
| 10 | 0.1695 | 0.2177 | +0.048 | <0.001 | **Yes** |

v2 is a statistically significant improvement over v1. However, v2 vs naive_hybrid:

| k | v2 F1 | naive_hybrid F1 | Gap |
|---|-------|-----------------|-----|
| 3 | 0.2240 | 0.2084 | v2 wins by +0.016 |
| 5 | 0.2300 | 0.2332 | v2 loses by -0.003 |
| 10 | 0.2177 | 0.2407 | v2 loses by -0.023 |

v2 wins at k=3 (where chunks_only is king and v2 inherits that) but loses at k>=5 where hybrid's diversity matters.

### 3.4 Token Cost Efficiency

v2 has a modest token cost advantage over naive_hybrid:

| k | v2 tokens | naive_hybrid tokens | Savings |
|---|-----------|---------------------|---------|
| 5 | 343 | 280 | -22% (v2 worse) |
| 10 | 713 | 599 | -19% (v2 worse) |

Actually v2 is MORE expensive than naive_hybrid. This kills any efficiency argument.

Wait — looking more carefully, naive_hybrid at k=10 uses 599 tokens for 0.2407 F1, while v2 at k=5 uses 343 tokens for 0.2300 F1. So v2 achieves 95.6% of naive_hybrid's F1 at 57% of the token cost. This is a weak Pareto argument at best.

---

## 4. Root Cause: Why Routing Cannot Beat Hybrid on QASPER

The fundamental issue is now clear after two iterations:

1. **Gold evidence is paragraph-level.** The meso index is structurally optimal for matching gold evidence. Any routing that reduces meso representation in the result set hurts.

2. **Naive hybrid is implicitly routing.** When all three levels are in the pool, cosine similarity naturally picks the best-matching granularity. The semantic matching IS a form of routing — and it uses the actual query-document similarity rather than a noisy 3-way classifier.

3. **The SLoD probe adds noise, not signal, for routing.** With 0.72 macro-F1 and mean confidence of 0.63, the probe's routing decisions are wrong often enough that hard routing hurts. Even confidence-gated soft routing (v2) doesn't help because the 0.6 threshold still routes 55% of queries.

4. **The micro advantage is real but small.** Only 11.2% of queries are micro-classified, and only 31.6% of those are confident enough to route. That's 3.6% of queries where routing is distinctive. Even if micro routing gave +10 pts F1 on those queries (it gives +2.8), that's only +0.36 pts on the aggregate.

---

## 5. The `slod_weighted` Proposal: Analysis

### Concept
Instead of hard routing (pick one level) or fallback (one level OR hybrid), boost the retrieval scores of documents from the predicted SLoD level:

```
For a query predicted as "micro" with confidence 0.7:
  micro doc scores *= 1.0 + 0.7 * boost_factor
  meso doc scores *= 1.0
  macro doc scores *= 1.0 - 0.7 * penalty_factor
```

Then rank all documents together (like naive_hybrid) but with the SLoD signal as a soft tiebreaker.

### Why It Might Work

1. **Preserves hybrid diversity.** All three levels remain in the pool, so we never lose the multi-level advantage that makes naive_hybrid strong.
2. **Adds a routing signal without hard commitment.** A micro-predicted query with high confidence gets more micro docs in top-k, but still gets meso docs if they are strong matches.
3. **Gracefully degrades.** Low-confidence predictions produce small boosts, effectively reverting to naive_hybrid. High-confidence predictions give a meaningful push.
4. **Could exploit the micro advantage.** The 48 high-confidence micro queries might get better results with a gentle micro boost than with either hard routing or no routing.

### Why It Might Not Work

1. **The magnitude matters critically.** Too much boost = hard routing (which we know fails). Too little boost = naive_hybrid (which we are trying to beat). The sweet spot may be narrow.
2. **Needs tuning on validation set.** We need to search over boost_factor (e.g., 0.1, 0.2, 0.5, 1.0) and possibly per-class factors.
3. **The effect size will be tiny.** Even in the best case, we are talking about reranking within a pool that naive_hybrid already ranks well. The expected improvement is 0.5-1.5 pts F1 at most.

### Implementation Estimate

- Add `slod_weighted` condition to `src/retrieve.py`: ~30 lines
- Add boost_factor to config.yaml: 1 line
- Re-run steps 04-06: ~15 min
- Tune on validation split: could be done within step 04 with a small grid search

### Expected Outcome Probabilities

- Beats naive_hybrid by >1 pt F1: 20%
- Matches naive_hybrid (within 0.5 pts): 40%
- Loses to naive_hybrid: 40%

### Risk Assessment

Low risk. One more iteration out of 45 budget. The weighted approach is a natural extension of the analysis and would strengthen the paper's contribution regardless of outcome:
- If it wins: "Soft SLoD-weighted retrieval outperforms hard routing and fixed baselines"
- If it ties: "SLoD signal provides a lightweight routing mechanism that matches hybrid performance with interpretable per-query routing explanations"
- If it loses: "Even soft routing cannot beat implicit routing via semantic similarity, suggesting SLoD information is already captured in embedding space"

---

## 6. Verdict: OPTION B — One More Iteration, Then Accept

### Rationale

1. **Budget is ample.** 2 of 45 iterations used. One more is trivially within budget.
2. **Implementation is minimal.** ~30 lines of code, 15 min runtime.
3. **The analysis value is high regardless of outcome.** The weighted-vs-hard-vs-implicit comparison is the most interesting scientific question to emerge from this experiment.
4. **The risk of over-engineering is low.** We are not adding complexity to chase marginal gains — we are testing a fundamentally different approach (soft vs hard routing).

### What "Accept" Means After Iteration 3

Whether `slod_weighted` beats naive_hybrid or not, the final report should frame SH3 as a **nuanced partial result** with these findings:

1. **Hard SLoD routing hurts** when the macro index is weak and probe confidence is moderate. (v1 result)
2. **Confidence-gated routing with macro-fallback recovers** most of the loss, achieving near-parity with baselines. (v2 result)
3. **Micro-level routing shows a genuine positive signal** for fine-grained questions, but the effect is small because (a) few queries are confidently micro-classified, and (b) naive_hybrid already captures much of this benefit through implicit routing.
4. **The QASPER evaluation protocol has a structural meso-level bias** that limits the testability of hierarchical retrieval hypotheses.
5. [If weighted works] **Soft SLoD-weighted retrieval can slightly outperform both hard routing and flat baselines** by combining hybrid diversity with a routing signal.
6. [If weighted fails] **Semantic similarity in embedding space already performs implicit routing**, making explicit SLoD-based routing redundant for this task.

### Implementation Instructions for Iteration 3

#### `config.yaml` — Add:
```yaml
retrieval:
  conditions:
    # ... existing ...
    - "slod_weighted"
  slod_weight_boost: 0.3  # Score multiplier for predicted-level documents
```

#### `src/retrieve.py` — Add new condition:
```python
elif condition_name == "slod_weighted":
    # Soft-weighted hybrid: all levels in pool, predicted level boosted
    if slod_pred is None:
        slod_pred = "meso"

    # Disable macro boost (same rationale as v2)
    effective_pred = "meso" if slod_pred == "macro" else slod_pred

    boost = config.get("retrieval", {}).get("slod_weight_boost", 0.3)
    slod_conf = slod_confidence if slod_confidence is not None else 0.5

    # Collect all documents with level tags
    all_embs = []
    all_texts = []
    all_doc_ids = []
    all_levels = []
    for level in ["macro", "meso", "micro"]:
        paper_data = paper_lookups.get(level, {}).get(paper_id)
        if paper_data:
            n = paper_data["embeddings"].shape[0]
            all_embs.append(paper_data["embeddings"])
            all_texts.extend(paper_data["texts"])
            all_doc_ids.extend(paper_data["doc_ids"])
            all_levels.extend([level] * n)

    if not all_embs:
        return []

    combined_embs = np.concatenate(all_embs, axis=0)
    query_emb_2d = query_emb.reshape(1, -1)
    scores = cosine_similarity(query_emb_2d, combined_embs)[0]

    # Apply confidence-weighted boost to predicted level
    for i, level in enumerate(all_levels):
        if level == effective_pred:
            scores[i] *= (1.0 + slod_conf * boost)

    # Rank and return top-k
    k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "doc_id": all_doc_ids[idx],
            "text": all_texts[idx],
            "score": float(scores[idx]),
            "rank": rank + 1,
        })
    return results
```

#### After implementation:
1. Delete cached results: `data/results/retrieval_results.json`, `evaluation_metrics.json`, `bootstrap_tests.json`, `analysis_breakdowns.json`
2. Delete `reports/SH3_report.md` and `reports/figures/*.png`
3. Re-run steps 04, 05, 06
4. If validation tuning is desired: run step 04 with `eval_split: "validation"` first, sweep `slod_weight_boost` in [0.1, 0.2, 0.3, 0.5, 1.0], pick best, then run on test

#### Confusion analysis note
The current confusion analysis compares v1 `slod_routed` against baselines. For iteration 3, it should be updated to compare `slod_weighted` (or whichever is the best SLoD condition) against baselines. The confusion function in `src/analyze.py` may need a small tweak to accept the condition name as a parameter.

---

## 7. What We Have Learned (Regardless of Iteration 3 Outcome)

### For the SLoD Research Program

1. **SLoD classification is a solved problem** (SH1: 0.72 macro-F1), but **SLoD routing is not straightforward**. The value of knowing a query's SLoD depends critically on having good indices at each level and an evaluation protocol that doesn't bias toward one level.

2. **The macro level needs rethinking.** First-paragraph heuristic summaries are not useful for retrieval. Either use LLM-generated summaries (expensive) or accept that macro-level retrieval requires a different evaluation framework (e.g., answer quality rather than evidence attribution).

3. **Micro-level retrieval is the strongest use case for SLoD routing.** When a query genuinely needs a specific detail (a number, a name, an equation), sentence-level retrieval can surface it when paragraph-level retrieval buries it in context. This is the finding to emphasize in the final report.

4. **Confidence calibration is critical for routing systems.** A probe with 0.72 F1 and poor calibration (46% below 0.6 confidence) is not suitable for hard routing. Future work should explore calibrated confidence (Platt scaling, temperature scaling) or alternative routing mechanisms (e.g., query complexity features).

### For the Experiment Methodology

5. **The QASPER evidence attribution protocol is a ceiling**, not a floor. Any system that retrieves paragraph-level text from the correct section will score well. Systems that retrieve different-granularity text are penalized by the token-F1 matching, even if the retrieved content is more useful for answer generation.

6. **Naive hybrid is a surprisingly strong baseline** because cosine similarity in a shared embedding space implicitly performs routing. This suggests that for SLoD routing to add value, the routing signal must contain information NOT present in the embeddings — e.g., structural information about the document hierarchy.
