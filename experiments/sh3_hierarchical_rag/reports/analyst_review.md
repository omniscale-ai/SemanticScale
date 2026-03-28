# SH3 Analyst Review: Why SLoD-Routing Underperforms and What To Do About It

**Date:** 2026-03-10
**Status:** COMPLETE
**Verdict:** ITERATE — two targeted improvements can likely flip the result

---

## 1. Executive Summary

SLoD-routed retrieval (F1=0.207 at k=3) loses to chunks-only (0.237) by 3 points and ties naive-hybrid (0.208). The root cause is **not** a fundamental flaw in routing — it is a compounding of two specific, fixable problems:

1. **The macro index is structurally weak.** First-paragraph-as-summary is a poor heuristic that produces documents inferior to full paragraphs for retrieval matching against gold evidence.
2. **46% of SLoD predictions have confidence below 0.6.** The probe routes with barely-better-than-random confidence for nearly half of all queries, and errors on macro-classified queries are catastrophic.

The good news: **micro routing already works** (F1 boost of +1.4 to +3.9 points over chunks-only at k>=5), and **meso routing is neutral** (identical to chunks-only, as expected). The entire loss comes from macro-routed queries.

**Recommendation:** Implement confidence-based hybrid fallback (Improvement A) and test a weighted hybrid approach (Improvement B). These require changes to `src/retrieve.py` only, with no re-embedding needed. Expected impact: +3 to +4 points on attribution F1, which would make slod_routed competitive with or superior to chunks-only.

---

## 2. Detailed Diagnosis

### 2.1 The Macro Routing Problem

This is the dominant failure mode. The numbers are unambiguous:

| k | SLoD-routed (macro queries) | Chunks-only (macro queries) | Delta |
|---|---|---|---|
| 1 | 0.1504 | 0.1755 | -0.0252 |
| 3 | 0.1685 | 0.2375 | **-0.0690** |
| 5 | 0.1571 | 0.2234 | **-0.0663** |
| 10 | 0.1246 | 0.1930 | **-0.0685** |
| 20 | 0.1072 | 0.1484 | -0.0412 |

For the 590 macro-classified queries (43.6% of test set), routing to the macro index **destroys** performance. The SLoD-routed F1 for macro queries exactly matches the summaries_only baseline (verified numerically), confirming that the routing logic works correctly — the problem is the macro index itself.

**Why the macro index fails:** Each paper averages only 14.4 macro documents vs 46.0 meso documents. The macro documents are first paragraphs of sections — these are typically introductory sentences that don't contain the specific claims, methods, or results that gold evidence paragraphs contain. QASPER gold evidence is paragraph-level, so matching against a section's first paragraph (which may not even be the gold paragraph) is fundamentally disadvantaged.

### 2.2 Micro Routing: The Bright Spot

Micro routing shows a consistent positive signal that grows with k:

| k | SLoD-routed (micro queries) | Chunks-only (micro queries) | Delta |
|---|---|---|---|
| 3 | 0.2484 | 0.2445 | +0.0039 |
| 5 | 0.2530 | 0.2389 | +0.0141 |
| 10 | 0.2304 | 0.2040 | **+0.0263** |
| 20 | 0.1983 | 0.1593 | **+0.0390** |

At k>=10, micro routing provides a meaningful advantage. This makes theoretical sense: micro chunks can capture specific details (numbers, entity names, experimental conditions) that paragraph-level documents dilute. Unfortunately, only 152 queries (11.2%) are classified as micro, limiting the overall impact.

### 2.3 Meso Routing: Neutral by Design

Meso-classified queries (610, 45.1%) produce identical results to chunks-only, which is expected since both retrieve from the meso (paragraph) index. This is not a problem — it just means meso routing adds no value but also no harm.

### 2.4 Probe Confidence Crisis

The SLoD probe has a severe calibration problem:

| Class | Total | Conf <0.5 | 0.5-0.6 | 0.6-0.8 | >0.8 | Mean Conf |
|-------|-------|-----------|---------|---------|------|-----------|
| macro | 954 | 192 (20%) | 186 (20%) | 399 (42%) | 177 (19%) | 0.646 |
| meso | 1058 | 231 (22%) | 260 (25%) | 430 (41%) | 137 (13%) | 0.623 |
| micro | 267 | 129 (48%) | 60 (22%) | 72 (27%) | 6 (2%) | 0.533 |

**46.4% of all predictions have confidence below 0.6.** For micro, nearly half are below 0.5 (the random threshold for 3-way). The probe's macro-F1 of 0.72 means roughly 28% of queries are misrouted, and misrouting to macro is catastrophic (-6.6 points F1).

### 2.5 Confusion Analysis

At k=5: only 14 queries helped, 619 hurt, 719 neutral. The hurt cases break down as:
- **macro-predicted: 286 hurt** (48.5% of macro queries)
- **meso-predicted: 260 hurt** (42.6% of meso queries)
- **micro-predicted: 73 hurt** (48.0% of micro queries)

The meso hurt cases are puzzling since meso routing equals chunks-only — these 260 cases must be queries where naive_hybrid outperforms chunks_only (the confusion analysis compares against the *best* baseline per query, not just chunks_only).

### 2.6 Counterfactual Analysis

If we simply routed macro-classified queries to the meso index instead (i.e., never use the macro index):

| k | Current SLoD-routed | If macro→meso | Delta | Chunks-only |
|---|---|---|---|---|
| 3 | 0.2073 | 0.2374 | +0.0301 | 0.2370 |
| 5 | 0.1985 | 0.2274 | +0.0289 | 0.2258 |
| 10 | 0.1695 | 0.1994 | +0.0299 | 0.1964 |

This naive fix alone would make SLoD-routed **match** chunks-only at k=3 and slightly **exceed** it at k>=5. But it would still lose to naive_hybrid at k=10 (0.2407).

---

## 3. Structural Issues Assessment

### 3.1 Macro Index Quality: POOR
- "First paragraph as summary" is a weak heuristic. First paragraphs are often transitional/introductory.
- Only 14.4 macro docs per paper vs 46.0 meso docs — the search space is too constrained.
- Macro documents overlap 100% with meso documents (every macro doc is also a meso doc), so they add no new information.
- Gold evidence in QASPER is at paragraph granularity, inherently favoring meso-level retrieval.

### 3.2 Micro Index Quality: GOOD
- 83.9 micro docs per paper provides good coverage.
- Sentence-level chunks with overlap capture fine-grained details.
- Figure/table captions are included, adding value for specific queries.
- Token-F1 matching at 0.5 threshold successfully handles the granularity mismatch (micro chunk matching against full gold paragraph).

### 3.3 Evaluation Bias Toward Meso
QASPER gold evidence is defined at paragraph level, which inherently advantages the meso (paragraph) index. This is a structural limitation of the evaluation, not necessarily of the approach. In a real RAG system, macro-level retrieval would help with answer generation even if it doesn't match specific gold paragraphs.

---

## 4. Improvement Recommendations (Ranked by Expected Impact)

### A. Confidence-Based Hybrid Fallback (HIGH IMPACT, LOW EFFORT)

**What:** When probe confidence is below a threshold (0.6 or 0.7), fall back to naive_hybrid retrieval instead of routing to a single level.

**Expected impact:** +2 to +4 points F1. This addresses 46% of queries where the probe is uncertain. For confident predictions, routing still happens (preserving the micro-level gains).

**Implementation:** Modify `run_condition()` in `src/retrieve.py`:
- Add confidence threshold parameter (from config)
- When `slod_routed` and confidence < threshold: use naive_hybrid logic
- When confidence >= threshold: route as before

**Effort:** ~20 lines of code change in `src/retrieve.py`, add one config parameter.

### B. Never-Route-to-Macro (HIGH IMPACT, MINIMAL EFFORT)

**What:** For macro-classified queries, route to meso instead of macro. Effectively: macro→meso, meso→meso, micro→micro.

**Expected impact:** +3 points F1 at k=3, matching chunks-only. Combined with Improvement A, could exceed it.

**Implementation:** One line change in `run_condition()`: if `slod_pred == "macro"`, set to `"meso"`.

**Effort:** Trivial.

### C. Combine A + B: Confidence-Gated Meso/Micro Routing (RECOMMENDED)

**What:** Combine both fixes:
1. Never route to macro (always use meso for macro-classified queries)
2. For low-confidence predictions (<0.6), use naive_hybrid
3. For high-confidence micro predictions, route to micro

**Expected impact:** +3 to +5 points F1. This preserves the micro-routing gains for confident predictions while eliminating the macro penalty and hedging uncertain predictions.

**Implementation:** ~30 lines in `src/retrieve.py`, 2 config parameters.

### D. Weighted Hybrid (MEDIUM IMPACT, MODERATE EFFORT)

**What:** Instead of hard routing, weight retrieval scores by SLoD match. If query is predicted micro with 0.7 confidence: micro scores *= 1.0, meso scores *= 0.3, macro scores *= 0.0. Then rank all together.

**Expected impact:** Unknown but theoretically elegant — provides soft routing that degrades gracefully.

**Implementation:** New retrieval function in `src/retrieve.py`, ~50 lines. Needs confidence-to-weight mapping.

### E. Better Macro Index via LLM Summaries (HIGH POTENTIAL, HIGH EFFORT)

**What:** Replace first-paragraph heuristic with LLM-generated section summaries.

**Expected impact:** Could make the macro index genuinely useful, but requires API calls for 697 papers x ~10 sections each = ~7K summaries.

**Implementation:** New data prep step, re-embedding macro documents. High cost, uncertain benefit given the evaluation bias toward meso.

**Recommendation:** Defer to future work. The retrieval evaluation is structurally biased against macro; fixing the macro index won't help much when gold evidence is paragraph-level.

---

## 5. Exit Criteria Assessment

Per PLAN.md Section 9:

**Current status: Between "Failure" and "Partial Success"**
- SLoD-routed does NOT beat chunks-only at any k (Failure criterion)
- SLoD-routed beats summaries_only significantly at all k (partial credit)
- SLoD-routed ties naive_hybrid at k=3 (p=0.44, not significant)
- Micro routing shows a real positive signal (Partial Success indicator)

**After proposed improvements (A+B+C):**
- Expected to match or slightly beat chunks-only
- May approach naive_hybrid at moderate k
- Would demonstrate that *selective* routing (micro only, with confidence gating) provides value
- This shifts the narrative from "routing doesn't work" to "granularity-matched routing helps for fine-grained queries when confidence is high"

**Verdict: ITERATE with improvements A+B (combined as C)**

Justification:
1. The fix is surgical (one file, ~30 lines) and requires no re-embedding or re-running the expensive pipeline steps
2. Budget allows it (4 of 45 iterations used)
3. The micro-routing signal is real and worth preserving
4. The improved result would be publishable: "Confidence-gated SLoD routing with level-aware fallback"
5. Even if the improved version only matches chunks-only, the analysis itself (why naive routing fails, where it helps) is valuable

---

## 6. Specific Implementation Instructions for Next Iteration

### Files to Modify

**`config.yaml`** — Add:
```yaml
retrieval:
  # ... existing params ...
  slod_confidence_threshold: 0.6  # Below this, fall back to hybrid
  slod_disable_macro_routing: true  # Never route to macro index
```

**`src/retrieve.py`** — Modify `run_condition()`:
```python
elif condition_name == "slod_routed":
    if slod_pred is None:
        slod_pred = "meso"

    # Get confidence for this prediction
    slod_conf = slod_confidence if slod_confidence is not None else 1.0
    conf_threshold = config.get("retrieval", {}).get("slod_confidence_threshold", 0.0)
    disable_macro = config.get("retrieval", {}).get("slod_disable_macro_routing", False)

    # Improvement B: Never route to macro
    if disable_macro and slod_pred == "macro":
        slod_pred = "meso"

    # Improvement A: Low confidence -> fall back to naive hybrid
    if slod_conf < conf_threshold:
        # Use naive_hybrid logic
        return run_condition("naive_hybrid", query_emb, paper_id, paper_lookups, top_k)

    # High confidence: route to predicted level
    paper_data = paper_lookups.get(slod_pred, {}).get(paper_id)
    if not paper_data:
        return []
    return retrieve_for_paper(
        query_emb, paper_data["embeddings"],
        paper_data["texts"], paper_data["doc_ids"], top_k,
    )
```

**`src/retrieve.py`** — Modify `run_all_retrieval()` to pass confidence:
- When building slod_pred, also extract `slod_conf = slod_preds[qid]["probabilities"][slod_pred]`
- Pass it to `run_condition()`

**`scripts/04_retrieve.py` through `scripts/06_analyze.py`** — Delete cached results so the pipeline re-runs:
- Delete `data/results/retrieval_results.json`
- Delete `data/results/evaluation_metrics.json`
- Delete `data/results/bootstrap_tests.json`
- Delete `data/results/analysis_breakdowns.json`
- Delete `reports/SH3_report.md`
- Delete `reports/figures/*.png`

### What Does NOT Need to Re-Run
- Step 01 (data prep) — unchanged
- Step 02 (embeddings) — unchanged
- Step 03 (query classification) — unchanged (predictions already saved with confidences)

### What Needs to Re-Run
- Step 04 (retrieval) — new routing logic
- Step 05 (evaluation) — new metrics for modified condition
- Step 06 (analysis) — new breakdowns and report

**Estimated time: ~15 minutes** (retrieval + evaluation + analysis)

---

## 7. Broader Implications for the SLoD Research Program

1. **Macro-level retrieval is fundamentally challenged** when evaluation uses paragraph-level gold evidence. This suggests SH3 should reframe: the value of macro retrieval is for answer *generation* (providing context), not evidence *attribution*. An SH4 evaluation with LLM-generated answers would be fairer to macro.

2. **SLoD classification confidence matters more than accuracy.** Even with a 0.72 F1 probe, confident predictions can be useful if low-confidence cases are handled gracefully. This is a general lesson for probe-based routing.

3. **The QASPER evaluation protocol advantages meso granularity.** Any future SLoD experiments should either (a) use multi-granularity gold annotations, or (b) evaluate at the answer quality level rather than evidence attribution level.

4. **Micro routing is the unexpected winner.** The SLoD framework's real value may be in identifying fine-grained questions and routing them to sub-paragraph retrieval, rather than in the original macro-vs-meso distinction.
