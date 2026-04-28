# SH6 Stage-5 — Cross-Dataset Model Comparison

**Incumbent:** `logreg` &nbsp;&nbsp; **Challenger:** `lightgbm`

**Decision rule:** Δ-AUC ≥ +0.03 AND CI-lower > 0 (per run); ≥3 of 5 protocol runs to carry overall.

| Run                                                                     |   N |   Pos |   AUC logreg |   AUC lightgbm |   Δ-AUC |   CI low |   CI high | Verdict      | In protocol set   |
|:------------------------------------------------------------------------|----:|------:|-------------:|---------------:|--------:|---------:|----------:|:-------------|:------------------|
| agenthallu/framework-all                                                | 693 |   250 |        0.503 |          0.515 |   0.012 |   -0.039 |     0.065 | inconclusive | True              |
| frontierscience/R1-Distill-32B-cloudjudge_reasoning-auto                | 147 |    27 |        0.566 |          0.522 |  -0.044 |   -0.138 |     0.058 | regress      | False             |
| frontierscience/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B_reasoning-auto | 145 |    31 |        0.61  |          0.647 |   0.037 |   -0.081 |     0.158 | inconclusive | False             |
| frontierscience/deepseek/deepseek-v3.2_reasoning-auto                   | 153 |    61 |        0.837 |          0.863 |   0.026 |   -0.025 |     0.078 | inconclusive | True              |
| gpqa-diamond/deepseek/deepseek-v3.2_reasoning-auto                      | 192 |   147 |        0.696 |          0.663 |  -0.033 |   -0.115 |     0.053 | regress      | True              |
| processbench/gsm8k                                                      | 400 |   200 |        0.507 |          0.469 |  -0.038 |   -0.099 |     0.024 | regress      | True              |
| processbench/olympiadbench                                              | 400 |   200 |        0.474 |          0.522 |   0.048 |   -0.019 |     0.116 | inconclusive | False             |
| processbench/omnimath                                                   | 400 |   200 |        0.632 |          0.555 |  -0.077 |   -0.134 |    -0.021 | regress      | False             |
| swe-agent-trajectories/model-all                                        | 645 |   202 |        0.858 |          0.863 |   0.004 |    0     |     0.009 | inconclusive | True              |
| swe-agent-trajectories/model-all_steps-50                               | 200 |    17 |        0.913 |          0.913 |   0     |   -0     |     0     | inconclusive | False             |

## Protocol verdict

- Protocol runs evaluated: **5 / 5**
- Challenger wins: **0**
- Significant regressions: **2**
- ≥3-of-5 carry rule: **FAIL — challenger does NOT carry the comparison**

Per the stop conditions in `DESIGN-stage5-models.md`, this implies no broadly applicable interaction signal in the current Stage-5 feature set. The right next step is feature engineering (TA-pack, multi-scale, cross-trajectory), not a stronger classifier.
