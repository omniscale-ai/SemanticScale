# SH5 Conclusive Report: Abstraction-Level Jump Rate as a Behavioral Signature of Reasoning Quality

**Verdict: NOT CONFIRMED**
**Date:** 2026-03-15
**Author:** Monitor Agent (Final Report)

---

## Executive Summary

SH5 tested whether the abstraction-level jump rate in chain-of-thought (CoT) reasoning steps serves as a behavioral signature of reasoning quality. The hypothesis was twofold: (1) lower jump rates should correlate with higher answer correctness, and (2) SLoD-routed retrieval conditions should naturally induce lower jump rates compared to unrouted baselines.

**Neither hypothesis was confirmed.**

- **H1 (Correlation):** No correlation between normalized jump rate and answer correctness (Spearman rho = 0.003, p = 0.90, 95% CI [-0.042, 0.047]). The confidence interval tightly straddles zero, ruling out even weak effects.
- **H2 (Condition Difference):** SLoD-routed conditions show marginally lower jump rates (slod_weighted_parent: 0.429 vs chunks_only: 0.447) but the difference is not statistically significant (Wilcoxon p = 0.25, rank-biserial r = 0.083).
- **Unexpected finding:** All six jump metrics correlate *positively* with attribution F1 (evidence quality), with normalized_jump_rate showing rho = 0.092 (p = 4.1e-05). This suggests that cross-level reasoning -- jumping between abstraction levels -- may actually be *beneficial* for identifying relevant evidence, contrary to the original hypothesis.

---

## Experimental Setup

### Design

- **500 QASPER test questions** stratified by answer type (extractive: 828 traces, abstractive: 844 traces, yes/no: 328 traces across conditions)
- **4 retrieval conditions** from SH3 at k=5: `chunks_only`, `naive_hybrid`, `slod_weighted`, `slod_weighted_parent`
- **2000 total CoT traces** (500 questions x 4 conditions) generated via Claude Haiku (`claude-3-haiku-20240307`, temperature=0.0)
- Each CoT step tagged with the SH1 SciBERT+LogReg probe (macro/meso/micro classification, macro-F1 = 0.72)
- **6 jump metrics** computed per trace: jump_count, mean_abs_delta, max_jump, slod_variance, direction_changes, normalized_jump_rate
- Correlated with **token-F1** (answer correctness vs gold) and **attribution F1** (evidence quality vs gold evidence)

### Pipeline Statistics

| Statistic | Value |
|-----------|-------|
| Total CoT traces | 2000 |
| Total reasoning steps | 6101 |
| Parse success rate | 100% |
| Mean steps per trace | 3.05 |
| Median steps per trace | 3 |
| SLoD distribution | macro 41.8%, meso 12.4%, micro 45.8% |
| Probe macro-F1 (validation) | 0.72 |
| API cost | ~$0.50 |

---

## Results

### H1: Correlation Between Jump Rate and Answer Correctness

**Primary test:** Spearman rho(normalized_jump_rate, answer_token_f1) = 0.003, p = 0.90. Criterion was rho < -0.10 with p < 0.05. **NOT PASSED.**

#### Full Correlation Table (Overall, N=2000)

| Jump Metric | vs Token-F1 (rho) | p-value | vs Attribution-F1 (rho) | p-value |
|---|---|---|---|---|
| jump_count | 0.003 | 0.903 | **0.093** | **3.1e-05** |
| mean_abs_delta | -0.015 | 0.493 | **0.071** | **0.001** |
| max_jump | -0.016 | 0.468 | **0.059** | **0.009** |
| slod_variance | -0.016 | 0.466 | **0.057** | **0.010** |
| direction_changes | -0.010 | 0.647 | **0.062** | **0.005** |
| normalized_jump_rate | 0.003 | 0.899 | **0.092** | **4.1e-05** |

No jump metric achieves even marginal significance against token-F1. All jump metrics show significant *positive* correlation with attribution F1.

#### By Answer Type (Normalized Jump Rate vs Token-F1)

| Answer Type | rho | p-value | 95% CI | n |
|---|---|---|---|---|
| Extractive | 0.063 | 0.068 | [-0.008, 0.133] | 828 |
| Abstractive | -0.059 | 0.085 | [-0.127, 0.008] | 844 |
| Yes/No | 0.045 | 0.414 | [-0.064, 0.154] | 328 |

The weak opposite-sign trends for extractive (+0.063) and abstractive (-0.059) cancel out in the aggregate, further confirming the null result.

#### By Condition (Normalized Jump Rate vs Token-F1)

| Condition | rho | p-value | n |
|---|---|---|---|
| chunks_only | -0.057 | 0.203 | 500 |
| naive_hybrid | 0.033 | 0.463 | 500 |
| slod_weighted | 0.011 | 0.804 | 500 |
| slod_weighted_parent | 0.017 | 0.713 | 500 |

No condition individually shows a significant correlation. The slight negative trend in `chunks_only` (rho = -0.057) is the closest to the hypothesized direction but is far from significant.

### H2: SLoD-Routed Conditions Have Lower Jump Rates

**Primary test:** slod_weighted_parent NJR (0.429) vs chunks_only NJR (0.447), Wilcoxon p = 0.247. Criterion was routed < unrouted with p < 0.05. **NOT PASSED.**

#### Condition Summaries (Normalized Jump Rate)

| Condition | Mean | Median | Std |
|---|---|---|---|
| chunks_only | 0.447 | 0.500 | 0.379 |
| naive_hybrid | 0.427 | 0.500 | 0.382 |
| slod_weighted | 0.434 | 0.500 | 0.385 |
| slod_weighted_parent | 0.429 | 0.500 | 0.376 |

#### Pairwise Wilcoxon Tests (Normalized Jump Rate)

| Comparison | Diff | p-value | Effect (r) |
|---|---|---|---|
| slod_weighted vs chunks_only | -0.013 | 0.317 | 0.074 |
| slod_weighted_parent vs chunks_only | -0.018 | 0.247 | 0.083 |
| slod_weighted vs naive_hybrid | +0.007 | 0.786 | 0.018 |
| slod_weighted_parent vs naive_hybrid | +0.002 | 0.972 | 0.002 |

All effect sizes are negligible (r < 0.1). The routed conditions trend marginally lower than `chunks_only` but are essentially indistinguishable from `naive_hybrid`.

### The Unexpected Positive Finding

The most notable result is the consistent **positive** correlation between jump metrics and attribution F1 across conditions:

| Condition | NJR vs Attribution-F1 (rho) | p-value |
|---|---|---|
| chunks_only | **0.159** | **0.0004** |
| slod_weighted | **0.098** | **0.029** |
| slod_weighted_parent | **0.126** | **0.005** |
| naive_hybrid | 0.012 | 0.784 |

This pattern is significant in 3 of 4 conditions and strongest in `chunks_only` (the condition with the least structured retrieval). It suggests that traces with *more* abstraction-level jumping tend to retrieve better evidence -- the opposite of what SH5 hypothesized.

### Regression Analysis

Multiple regression: answer_token_f1 ~ normalized_jump_rate + n_steps + condition + answer_type

- R-squared: 0.190
- The strongest predictor is answer type (yes_no coefficient = +0.364), not jump rate
- Normalized jump rate coefficient = +0.003 (effectively zero)
- Condition effects are negligible (-0.018 to -0.002)

---

## Interpretation

### Why SH5 Failed

1. **LLM CoT reasoning may not exhibit the same abstraction dynamics as human reasoning.** The SH5 hypothesis was grounded in an intuition about human cognition: that disciplined, level-consistent reasoning produces better answers. LLMs may not reason this way -- their "reasoning steps" are generated text, not genuine cognitive transitions.

2. **The SH1 probe faces domain shift.** The SciBERT+LogReg classifier was trained on document-level spans (sections, paragraphs) from scientific papers. CoT reasoning steps are a fundamentally different text type -- shorter, more formulaic, and task-directed. A macro-F1 of 0.72 on in-domain data may degrade substantially on out-of-domain CoT text, introducing classification noise that masks any real signal.

3. **"Jumping" between abstraction levels may be a sign of thorough reasoning.** The positive correlation with attribution F1 supports this interpretation. A model that considers both high-level concepts (macro) and specific details (micro) during reasoning may be doing a better job of synthesizing information from the retrieved evidence. Cross-level reasoning is not incoherent -- it may be integrative.

4. **The hypothesis assumes linear, level-consistent reasoning is optimal.** Scientific QA may benefit from exactly the opposite: cross-level synthesis where the model connects abstract claims to specific data points. The "jumping" that SH5 tried to penalize may be the mechanism by which models achieve good evidence attribution.

5. **Short traces limit metric sensitivity.** With a mean of 3.05 steps per trace, there are only ~2 transitions to measure. This floor effect limits the variance of jump metrics and reduces statistical power for detecting subtle correlations.

---

## Implications for the Paper

### What SH5 Means for the SLoD Thesis

SH5 is an honest negative result that **closes the experimental loop** without undermining the core findings:

- **SH1 (Mechanism) stands independently.** The probe correctly identifies abstraction levels in document spans with macro-F1 = 0.72. SH5's failure does not question this -- it questions whether the same classifier transfers to CoT text.
- **SH3 (Utility) stands independently.** SLoD-weighted retrieval improves evidence attribution. SH5's failure simply means this improvement does not manifest as measurably smoother reasoning traces.
- **SH5 narrows the claim.** We can confidently state: "Abstraction consistency in CoT steps is not a necessary condition for good reasoning in scientific QA." This is a useful boundary on the SLoD thesis.

### The Positive Correlation Is Worth Reporting

The unexpected positive correlation between jump rate and attribution F1 (rho = 0.092, p = 4.1e-05) is an interesting finding in its own right:

- It suggests that **cross-level reasoning may be beneficial** rather than harmful
- It is robust across 3 of 4 conditions
- It provides a counterpoint to the assumption that coherent = level-consistent
- Framing: "Models that traverse abstraction levels during reasoning tend to identify relevant evidence more effectively, suggesting that level diversity in reasoning may reflect integrative rather than incoherent cognition."

### Recommended Framing

In the paper, SH5 should be presented as:
1. A pre-registered hypothesis that was tested and not confirmed
2. Evidence that the SLoD structure in documents does not trivially transfer to reasoning behavior
3. A provocative hint (via the positive attribution-F1 finding) that cross-level reasoning may be adaptive
4. A clear delimiter on the SLoD claim: SLoD describes document structure, not reasoning structure

---

## Final Roadmap Summary: SH0-SH5 Status

| Sub-Hypothesis | Claim | Status | Key Result |
|---|---|---|---|
| **SH0** | Scientific documents exhibit a hierarchical structure of abstraction levels | **CONFIRMED** | Annotation study confirms macro/meso/micro stratification across 50 papers |
| **SH1** | Abstraction level can be automatically classified from text | **CONFIRMED** | SciBERT+LogReg probe achieves macro-F1 = 0.72 (3-class: macro/meso/micro) |
| **SH2** | Document sections cluster by abstraction level | **CONFIRMED** | UMAP visualization shows clear separation; section-type is predictive of SLoD level |
| **SH3** | SLoD-aware retrieval improves evidence attribution | **PARTIAL** | Soft SLoD-weighted retrieval beats chunk-only baselines on attribution F1; effect is moderate |
| **SH4** | SLoD-aware extraction improves answer quality | **PARTIAL** | Improvements on extractive questions; mixed results on abstractive |
| **SH5** | Lower abstraction jump rate in CoT correlates with answer quality | **NOT CONFIRMED** | rho = 0.003 (p = 0.90); unexpected positive correlation with attribution F1 |

### Overall Narrative

The SLoD thesis is supported at the document level (SH0-SH2) and shows practical utility in retrieval (SH3-SH4). SH5 tested whether SLoD extends from documents to reasoning behavior and found that it does not -- at least not in the direction hypothesized. The behavioral signature of good reasoning in scientific QA appears to be *more* cross-level, not less. This closes the loop honestly: SLoD is a property of documents and a useful signal for retrieval, but it is not a prescriptive template for how models should reason.

---

## Artifacts

### Data Files
- `data/selected_questions.jsonl` -- 500 sampled questions with gold answers
- `data/cot_traces/{condition}.jsonl` -- 2000 CoT traces (4 files)
- `data/cot_slod_tags.jsonl` -- SLoD labels for all 6101 reasoning steps
- `data/jump_metrics.jsonl` -- 6 jump metrics per trace
- `data/answer_scores.jsonl` -- Token-F1 and attribution F1 per trace
- `data/results/correlation_results.json` -- Full correlation analysis
- `data/results/condition_comparison.json` -- Condition comparison tests

### Figures
- `reports/figures/jump_rate_by_condition.png`
- `reports/figures/jump_vs_correctness_scatter.png`
- `reports/figures/slod_sequence_examples.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/metric_distributions.png`
- `reports/figures/answer_type_breakdown.png`

### Code
- `src/` -- 8 Python modules implementing the 7-stage pipeline
- `scripts/` -- 7 executable pipeline scripts
- `config.yaml` -- All hyperparameters
