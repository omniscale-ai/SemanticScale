# FrontierScience DeepSeek Olympiad — Self-Consistency Baseline

Plurality vote (Wang et al., 2022) across five same-config DeepSeek runs on the
Olympiad subset of FrontierScience. Each per-attempt grade is reused from the
upstream `is_correct` label; the SC verdict is the majority `is_correct` among
attempts that share the winning normalized answer.

## Setup

- Runs: 5
- Subset: `olympiad` (filter: `has_final_answer == true`)
- Problems with at least one row in every run: 100
- Errors / blank `predicted_answer` abstain (don't vote); SC counts them as wrong if no plurality forms.
- Ties broken by earliest `run_order` of the tied group (label-free).

## Headline metrics

| Metric | Value |
|---|---|
| Average-per-attempt Pass@1 | 58.2% |
| Pass@5 (oracle across the five attempts) | 83.0% |
| **Self-consistency Pass@1** | **68.0%** |
| Gain over avg Pass@1 | +0.098 (+9.8 pts) |
| Oracle gap recovered | 39.5% |
| Mean consistency (winning_count / 5) | 0.478 |

## Per-run baselines

| Run | Answered | Errors | Pass@1 |
|---|---:|---:|---|
| `deepseek/deepseek-v3.2_reasoning-auto` | 85 | 15 | 61.0% |
| `deepseek/deepseek-v3.2_reasoning-auto_s1` | 87 | 13 | 56.0% |
| `deepseek/deepseek-v3.2_reasoning-auto_s2` | 87 | 13 | 58.0% |
| `deepseek/deepseek-v3.2_reasoning-auto_s3` | 89 | 11 | 58.0% |
| `deepseek/deepseek-v3.2_reasoning-auto_s4` | 83 | 17 | 58.0% |

## Per-subject breakdown

| Subject | Problems | Avg Pass@1 | Pass@5 | SC Pass@1 | Mean consistency |
|---|---:|---|---|---|---:|
| biology | 10 | 32.0% | 70.0% | 20.0% | 0.760 |
| chemistry | 40 | 65.0% | 87.5% | 80.0% | 0.605 |
| physics | 50 | 58.0% | 82.0% | 68.0% | 0.320 |

## Selective prediction by agreement threshold

Restrict SC predictions to problems whose winning answer received at least N of 5 votes.

| Min winning votes | Coverage | Problems | SC Pass@1 |
|---:|---|---:|---|
| ≥1/5 | 100.0% | 100 | 68.0% |
| ≥2/5 | 60.0% | 60 | 73.3% |
| ≥3/5 | 36.0% | 36 | 75.0% |
| ≥4/5 | 26.0% | 26 | 76.9% |
| ≥5/5 | 17.0% | 17 | 70.6% |

## Interpretation

- This is the **textbook self-consistency baseline**: choose the answer most
  models agree on across independent samples. It uses no SLoD signal and no
  external scorer; the only inputs are the five generations and the upstream
  per-sample grading already in `traces.jsonl`.
- Compare directly against `frontierscience_deepseek_olympiad_bestof5.md`,
  which selects among the same five attempts using SLoD-derived confidence.
  Differences in `Pass@1` between the two reflect whether SLoD provides signal
  beyond simple inter-sample agreement.
- Coverage / SC-Pass@1 along the agreement threshold is a selective-prediction
  curve: high-consistency problems should be high-accuracy if self-consistency
  is well-calibrated.

Artifacts: `frontierscience_deepseek_olympiad_self_consistency_attempts.csv`, `frontierscience_deepseek_olympiad_self_consistency_votes.csv`, `frontierscience_deepseek_olympiad_self_consistency.json`.
