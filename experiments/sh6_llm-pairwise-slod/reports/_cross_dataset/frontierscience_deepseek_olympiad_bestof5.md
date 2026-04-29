# FrontierScience DeepSeek Olympiad Best-of-5

Strict same-dataset reranking analysis over the five DeepSeek FrontierScience runs,
using only the Olympiad subset and strict Olympiad-only out-of-fold SLoD scores.

## Setup

- Runs: 5
- Subset: `olympiad` only
- Scorer feature set: `trajectory_full`
- Scorer target label: `is_correct`
- Headline selection model: `lightgbm`
- Pass metrics count inference/grading errors as failed attempts.
- Unscored attempts rank last during selection; if a problem has no scored attempts at all, the deterministic fallback still counts as a failure.

## Headline metrics

| Metric | Value |
|---|---|
| Average-per-attempt Pass@1 | 58.2% |
| Pass@5 (oracle across the five attempts) | 83.0% |
| SLoD-selected Pass@1 (lightgbm) | 68.0% |
| Gain over average Pass@1 (lightgbm) | +0.098 (+9.8 pts) |
| Oracle gap recovered (lightgbm) | 39.5% |

## Per-run attempt baselines

| Run | Answered | Errors | Pass@1 | logreg OOF AUC | lightgbm OOF AUC |
|---|---:|---:|---|---|---|
| `deepseek/deepseek-v3.2_reasoning-auto` | 95 | 5 | 61.0% | 0.684 | 0.741 |
| `deepseek/deepseek-v3.2_reasoning-auto_s1` | 96 | 4 | 56.0% | 0.472 | 0.546 |
| `deepseek/deepseek-v3.2_reasoning-auto_s2` | 96 | 4 | 58.0% | 0.615 | 0.721 |
| `deepseek/deepseek-v3.2_reasoning-auto_s3` | 99 | 1 | 58.0% | 0.598 | 0.608 |
| `deepseek/deepseek-v3.2_reasoning-auto_s4` | 97 | 3 | 58.0% | 0.507 | 0.709 |

## Reranking results

| Scorer | Scored attempts | Problems with 5 scores | Problems with partial scores | Problems with 0 scores | Selected Pass@1 | Delta vs avg Pass@1 | Oracle gap recovered |
|---|---:|---:|---:|---:|---|---|---|
| `logreg` | 483 | 83 | 17 | 0 | 60.0% | +0.018 | 7.3% |
| `lightgbm` | 483 | 83 | 17 | 0 | 68.0% | +0.098 | 39.5% |

## Interpretation

- This is a **useful utility evaluation**: it tests whether SLoD-derived confidence helps choose among five attempts from the same model/config family.
- It is **only partially fair as core evidence for SLoD**. The FrontierScience answer-side detector family was selected post-hoc on FrontierScience, so gains here do not by themselves prove broad SLoD validity.
- The important safeguard in this analysis is that selection uses **out-of-fold** scores on the Olympiad slice, which avoids item-level in-sample leakage.
- Stronger evidence would require transfer: e.g. freeze the scorer, then rerank a different FrontierScience generator or a different dataset entirely.

Artifacts: `frontierscience_deepseek_olympiad_bestof5_attempts.csv`, `frontierscience_deepseek_olympiad_bestof5_selected.csv`, `frontierscience_deepseek_olympiad_bestof5.json`.
