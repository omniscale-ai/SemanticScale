# FrontierScience DeepSeek Olympiad — LLM Self-Verification Baseline

Each of the five same-config DeepSeek runs is re-graded by the same model
(without access to the reference answer). The verifier sees the problem,
the reasoning trace, and the final answer, and returns a `verdict` plus a
0-100 `confidence`. Per problem, the best-of-5 attempt is the one with the
largest **signed** confidence (`+conf` if verdict=correct, `-conf` otherwise).

Compare against:

- `frontierscience_deepseek_olympiad_self_consistency.md` (no scorer, plurality vote)
- `frontierscience_deepseek_olympiad_bestof5.md` (SLoD-derived confidence)

## Setup

- Runs: 5
- Subset: `olympiad` (filter: `has_final_answer == true`)
- Verifier model: `deepseek/deepseek-v3.2`
- Verifier sees: problem + student reasoning + student final answer (no reference answer)
- Best-of-5 picks highest signed confidence; unscored attempts rank last (deterministic fallback).

## Headline metrics

| Metric | Value |
|---|---|
| Average-per-attempt Pass@1 | 58.2% |
| Pass@5 (oracle across the five attempts) | 83.0% |
| **LLM-verify Pass@1** | **67.0%** |
| Gain over avg Pass@1 | +0.088 (+8.8 pts) |
| Oracle gap recovered | 35.5% |

## Per-run verifier diagnostics

| Run | Attempts | Verified | Errors | Pass@1 | Verifier 'correct' rate | Label agreement | OOF AUC | Mean conf |
|---|---:|---:|---:|---|---|---|---|---:|
| `deepseek/deepseek-v3.2_reasoning-auto` | 100 | 100 | 0 | 61.0% | 74.0% | 75.0% | 0.734 | 84.3 |
| `deepseek/deepseek-v3.2_reasoning-auto_s1` | 100 | 100 | 0 | 56.0% | 75.0% | 67.0% | 0.767 | 85.5 |
| `deepseek/deepseek-v3.2_reasoning-auto_s2` | 100 | 100 | 0 | 58.0% | 73.0% | 75.0% | 0.761 | 86.9 |
| `deepseek/deepseek-v3.2_reasoning-auto_s3` | 100 | 100 | 0 | 58.0% | 79.0% | 75.0% | 0.803 | 85.9 |
| `deepseek/deepseek-v3.2_reasoning-auto_s4` | 100 | 100 | 0 | 58.0% | 70.0% | 68.0% | 0.729 | 85.0 |

## Per-subject breakdown

| Subject | Problems | Avg Pass@1 | Pass@5 | Verify Pass@1 | Mean selected conf |
|---|---:|---|---|---|---:|
| biology | 10 | 32.0% | 70.0% | 20.0% | 85.0 |
| chemistry | 40 | 65.0% | 87.5% | 75.0% | 94.2 |
| physics | 50 | 58.0% | 82.0% | 70.0% | 92.5 |

## Selective prediction by verifier signed-score threshold

Restrict the best-of-5 selection to problems whose chosen attempt cleared the
given **signed** confidence threshold (`+conf` if verdict=correct, `-conf` if
verdict=incorrect). A threshold of 0 keeps all problems whose selected attempt
was judged correct.

| Min signed score | Coverage | Problems | Selective Pass@1 |
|---:|---|---:|---|
| ≥0 | 96.0% | 96 | 69.8% |
| ≥25 | 96.0% | 96 | 69.8% |
| ≥50 | 96.0% | 96 | 69.8% |
| ≥75 | 96.0% | 96 | 69.8% |
| ≥90 | 81.0% | 81 | 72.8% |

## Interpretation

- **LLM self-verification** is a simple, prompt-only baseline: no SLoD signal,
  no inter-sample voting, just ask the same model whether each solution looks
  right. Performance above the average per-attempt Pass@1 means the verifier
  is calibrated enough to discriminate its own correct vs. incorrect attempts.
- Compare directly to `frontierscience_deepseek_olympiad_bestof5.md` and
  `frontierscience_deepseek_olympiad_self_consistency.md`: all three select
  one of the same five attempts per problem, so deltas are apples-to-apples.

Artifacts: `frontierscience_deepseek_olympiad_llm_verify_attempts.csv`, `frontierscience_deepseek_olympiad_llm_verify_selected.csv`, `frontierscience_deepseek_olympiad_llm_verify.json`.
