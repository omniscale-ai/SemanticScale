# SH6 — Follow-up experiments (deferred from workshop submission)

These three experiments were designed during the ICML 2026 *Agentic Uncertainty*
workshop revision pass but **not executed** to keep the submission scope tight.
Each is self-contained and can be run independently against the data already on
disk. References below use repo-relative paths.

---

## B. Self-consistency / majority-vote baseline (Wang 2022)

**Goal.** Add a self-consistency row to Table 3 (`tab:rerank`) of
`SLoD-ICML-2026-Workshop/main.tex`, comparing the SLoD-LightGBM scorer against
the canonical no-trace baseline.

**Inputs.** Five `traces.jsonl` files at
`data/sh6/frontierscience/deepseek/deepseek-v3.2_reasoning-auto{,_s1..s4}/`.
Key fields per record: `id` (problem UUID), `predicted_answer` (extracted
free-text answer), `is_correct` (boolean grading result).

**Logic.**
1. Group attempts by problem `id` across the five seeds.
2. Normalize each `predicted_answer`: lowercase, strip whitespace, collapse
   internal whitespace, strip trailing punctuation.
3. Pick the modal normalized answer per problem.
4. Pass@1 = mean over problems of (modal answer's `is_correct`).

**Tie-break.** Deterministic first-of-modes (matches Wang 2022 default).
A SLoD-tiebreak would mix two signals and muddy the comparison.

**Critical decision: do not invoke the FrontierScience grader's normalization.**
That would conflate "what the model said" with "what the grader accepts" and
inflate the baseline. Exact-match-modulo-whitespace is the canonical
self-consistency comparison.

**Pre-commit framing decision** (resolve *before* writing the new row into the
paper, as it can change the abstract):

| Self-consistency Pass@1 | Action |
|---|---|
| < 63% | Keep current headline (+9.8 pp lift). Add one row "Self-consistency (majority vote)" between "Per-attempt average" and "LightGBM scorer". |
| ≥ 65% | SLoD vs. self-consistency becomes the new headline. Add a third row "SLoD-weighted majority": among problems where majority vote is non-unanimous, pick the modal answer's attempt with the highest SLoD score. Reframe abstract numerically. |

**Suggested script.** `experiments/sh6_llm-pairwise-slod/scripts/11_self_consistency_baseline.py`.

---

## C. Cross-generator transfer for SWE-agent

**Goal.** Convert the "in-dataset evaluation" limitation (paper §5,
`Limitations`) into a positive distribution-shift result by training the SLoD
failure scorer on llama-70b traces and evaluating on llama-{8b, 405b}.

**Inputs.** The per-trace feature table at
`experiments/sh6_llm-pairwise-slod/reports/swe-agent-trajectories/model-all/trajectory_features.csv`
with `generator` column populated as `swe-agent-llama-{8b,70b,405b}`
(n = 549 / 1232 / 236).

**Reuses.**
- `build_feature_table()` — `src/semanticscale/sh6/failure_analysis.py:519`
- `choose_feature_sets()` — same module; produces `mode_stack`, `trajectory_full`
- `compute_mode_scores()` — `src/semanticscale/sh6/failure_modes.py`
- `get_spec("lightgbm")` — `src/semanticscale/sh6/failure_models.py:500`

**Logic.**
1. Filter feature table to llama-70b rows; train LightGBM on `mode_stack` and
   `trajectory_full` separately (full fit, no CV on the train side).
2. Evaluate ROC-AUC on the held-out llama-8b and llama-405b rows.
3. Repeat with `random_state` ∈ {0..4}; report mean ± std.

**Output.** Six-row table — 2 feature sets × {70b CV diagonal from existing
reports, 70b → 8b, 70b → 405b}. Place in §4.4.1 of the paper after
`tab:swe_per_gen` (around L512).

**Direction.** Train on 70b (largest, AUC 0.91 on the diagonal — clean source).
The reverse direction (e.g., 8b → 70b) is uninformative because the source AUC
is itself only 0.85.

**Pre-commit framing.** If 70b → {8b, 405b} transfer is at-or-below the
in-distribution length baseline on the target generator, frame this as a
**confirmation** of the generator-conditioning hypothesis from §4.4.1, not as
failed transfer. The negative is on-theme for the workshop's
"uncertainty under distribution shift" topic. Don't bury it.

**Suggested script.** `experiments/sh6_llm-pairwise-slod/scripts/12_cross_generator_transfer.py`.

---

## D. Prefix-length ablation (anytime-valid evidence)

**Goal.** Show that the SLoD selective-sampling signal is detectable on
*partial* trajectories — i.e. usable as an online stopping criterion. Directly
addresses the workshop's anytime-valid sequential inference theme.

**Inputs.** `data/sh6/frontierscience/deepseek/deepseek-v3.2_reasoning-auto*/chunk_rankings.jsonl`
for the five seeds. Per-trace fields `reasoning_chunks` (list of paragraph
strings) and `reasoning_params` (real-valued SLoD coords) are aligned in trace
order.

**Logic.** For each prefix fraction p ∈ {0.25, 0.5, 0.75, 1.0}:
1. Truncate each trace's `reasoning_params` and `reasoning_chunks` to the first
   ⌊pN⌋ entries.
2. Rebuild trajectory features via `build_feature_table()` — it accepts the
   truncated lists directly (see `failure_analysis.py:538`).
3. 5-fold stratified CV on `trajectory_full` LightGBM.
4. Report ROC-AUC ± std.

**Unit choice.** Chunk fractions, not token/character — chunks are the natural
quantum of SLoD pairwise judging, and "p of the reasoning seen" is the
interpretable framing for an online deployment.

**Output.** Four-row inline `tabular` in §4.4.3 of the paper, *or* one-sentence
summary in Future Work depending on remaining page budget.

**Suggested script.** `experiments/sh6_llm-pairwise-slod/scripts/13_prefix_length_ablation.py`.

---

## Suggested execution order

If revisiting after the workshop deadline: **B → C → D**. B is cheapest and
most likely to change the paper's headline; C is the "limitation → contribution"
conversion most directly on workshop theme; D is the most novel framing and
worth the extra effort once B and C are landed.
