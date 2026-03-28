#!/usr/bin/env python3
"""Stage 6d: Evaluate SLoD shift, surface metrics, and quality for summaries.

H1: SLoD shift (micro vs baseline) — paired t-test, Cohen's d > 0.5, p < 0.05
H2: Surface metrics (entity/citation/numeric/sentence length) — 2+ of 4 significant
H3: ROUGE-L drop of micro summary vs micro_reference < rouge_drop_threshold

Output: data/summarization/evaluation_results.json
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_jsonl, load_json, save_json
from src.embedding import embed_texts
from src.evaluate import (
    compute_slod_shift,
    compute_surface_metrics,
    compare_surface_metrics,
)

try:
    from rouge_score import rouge_scorer as rouge_scorer_module
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    from src.utils import token_f1


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F-measure between prediction and reference."""
    if HAS_ROUGE:
        scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    else:
        return token_f1(prediction, reference)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6d: Evaluate summarization experiment"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    summ_cfg = config["summarization"]

    out_path = Path(summ_cfg["evaluation_results_path"])

    if out_path.exists() and not args.force:
        print(f"Output exists: {out_path}. Use --force to rerun.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load summaries
    summaries_path = Path(summ_cfg.get("summaries_path", "../../data/sh2/summarization/summaries.jsonl"))
    print(f"Loading summaries from {summaries_path}...")
    all_summaries = load_jsonl(str(summaries_path))

    # Separate by condition
    baseline_records = {r["paper_id"]: r for r in all_summaries if r["condition"] == "baseline"}
    micro_records = {r["paper_id"]: r for r in all_summaries if r["condition"] == "micro"}
    macro_records = {r["paper_id"]: r for r in all_summaries if r["condition"] == "macro"}

    # Find papers present in all conditions
    common_ids = sorted(
        set(baseline_records) & set(micro_records) & set(macro_records)
    )
    print(f"Papers with all 3 conditions: {len(common_ids)}")

    if len(common_ids) == 0:
        print("ERROR: No papers found with all three conditions. Run 04d first.")
        return

    baseline_texts = [baseline_records[pid]["summary"] for pid in common_ids]
    micro_texts = [micro_records[pid]["summary"] for pid in common_ids]
    macro_texts = [macro_records[pid]["summary"] for pid in common_ids]

    # 2. Load micro references from papers
    papers_path = Path(summ_cfg["papers_path"])
    split_path = papers_path.parent / "split.json"

    print("Loading papers for micro references...")
    papers = load_jsonl(str(papers_path))
    papers_by_id = {p["paper_id"]: p for p in papers}

    micro_references = [
        papers_by_id[pid]["micro_reference"]
        for pid in common_ids
        if pid in papers_by_id
    ]
    # Ensure alignment — only keep ids that have references
    aligned_ids = [pid for pid in common_ids if pid in papers_by_id]
    if len(aligned_ids) < len(common_ids):
        print(f"Warning: {len(common_ids) - len(aligned_ids)} papers missing from papers.jsonl")
        baseline_texts = [baseline_records[pid]["summary"] for pid in aligned_ids]
        micro_texts = [micro_records[pid]["summary"] for pid in aligned_ids]
        macro_texts = [macro_records[pid]["summary"] for pid in aligned_ids]
        common_ids = aligned_ids

    print(f"Evaluating {len(common_ids)} papers")

    # 3. Embed all summaries with SciBERT
    print("\nLoading SLoD evaluation axis...")
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    slod_axis = axis_data["centroid_axis"]

    scibert_model = config["scibert_model"]
    scibert_batch = config["scibert_batch_size"]
    scibert_maxlen = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch, scibert_maxlen)

    print("\nEmbedding baseline summaries...")
    baseline_embs = embed_fn(baseline_texts)
    baseline_scores = baseline_embs @ slod_axis

    print("Embedding micro-steered summaries...")
    micro_embs = embed_fn(micro_texts)
    micro_scores = micro_embs @ slod_axis

    print("Embedding macro-steered summaries...")
    macro_embs = embed_fn(macro_texts)
    macro_scores = macro_embs @ slod_axis

    # 4. H1: SLoD shift
    print("\nH1: Computing SLoD shift (micro vs baseline)...")
    h1_micro = compute_slod_shift(baseline_scores, micro_scores)
    h1_macro = compute_slod_shift(baseline_scores, macro_scores)

    thresholds = config.get("thresholds", {})
    h1_p_thresh = thresholds.get("h1_shift_p", 0.05)
    h1_d_thresh = thresholds.get("h1_shift_d", 0.5)

    h1_passed = (
        h1_micro["p_value"] < h1_p_thresh
        and abs(h1_micro["cohens_d"]) >= h1_d_thresh
    )
    print(f"  Micro: mean_delta={h1_micro['mean_delta']:.4f}, "
          f"p={h1_micro['p_value']:.4e}, d={h1_micro['cohens_d']:.4f}")
    print(f"  Macro: mean_delta={h1_macro['mean_delta']:.4f}, "
          f"p={h1_macro['p_value']:.4e}, d={h1_macro['cohens_d']:.4f}")
    print(f"  H1 passed: {h1_passed} (threshold: |d| >= {h1_d_thresh}, p < {h1_p_thresh})")

    # 5. H2: Surface metrics
    print("\nH2: Computing surface metrics...")
    baseline_surface = [compute_surface_metrics(t) for t in baseline_texts]
    micro_surface = [compute_surface_metrics(t) for t in micro_texts]

    h2_comparison = compare_surface_metrics(baseline_surface, micro_surface)
    h2_min_sig = thresholds.get("h2_min_metrics", 2)
    h2_passed = h2_comparison["n_significant"] >= h2_min_sig
    print(f"  Significant metrics: {h2_comparison['n_significant']}/4")
    print(f"  H2 passed: {h2_passed}")

    # 6. H3: ROUGE-L quality
    print("\nH3: Computing ROUGE-L vs micro reference...")
    baseline_rouge_scores = [
        compute_rouge_l(pred, ref)
        for pred, ref in zip(baseline_texts, micro_references)
    ]
    micro_rouge_scores = [
        compute_rouge_l(pred, ref)
        for pred, ref in zip(micro_texts, micro_references)
    ]

    baseline_rouge_mean = float(np.mean(baseline_rouge_scores))
    micro_rouge_mean = float(np.mean(micro_rouge_scores))
    rouge_drop = baseline_rouge_mean - micro_rouge_mean

    rouge_drop_threshold = summ_cfg.get("rouge_drop_threshold", 0.05)
    h3_passed = rouge_drop < rouge_drop_threshold

    print(f"  Baseline mean ROUGE-L: {baseline_rouge_mean:.4f}")
    print(f"  Micro-steered mean ROUGE-L: {micro_rouge_mean:.4f}")
    print(f"  Drop: {rouge_drop:.4f} (threshold: {rouge_drop_threshold})")
    print(f"  H3 passed: {h3_passed}")

    # 7. Verdict
    if h1_passed and h3_passed:
        verdict = "CONFIRMED"
    elif h1_passed and not h3_passed:
        verdict = "PARTIAL"
    else:
        verdict = "NOT CONFIRMED"
    print(f"\nVERDICT: {verdict}")

    # 8. Load alpha sweep for metadata
    alpha_sweep_path = Path("../../data/sh2/summarization/alpha_sweep.json")
    alpha_sweep_data = {}
    selected_alpha = None
    if alpha_sweep_path.exists():
        alpha_sweep_data = load_json(str(alpha_sweep_path))
        selected_alpha = alpha_sweep_data.get("best_alpha")

    layer_sel_path = Path(config["data_dir"]) / "results" / "layer_selection_summ.json"
    layer_selection = {}
    if layer_sel_path.exists():
        layer_selection = load_json(str(layer_sel_path))

    # 9. Save results
    results = {
        "verdict": verdict,
        "n_papers": len(common_ids),
        "h1_passed": h1_passed,
        "h2_passed": h2_passed,
        "h3_passed": h3_passed,
        "h1_slod_shift": {
            "micro": h1_micro,
            "macro": h1_macro,
        },
        "h2_surface": h2_comparison,
        "h3_quality": {
            "baseline_rouge_l": baseline_rouge_mean,
            "micro_rouge_l": micro_rouge_mean,
            "drop": float(rouge_drop),
            "threshold": float(rouge_drop_threshold),
            "pass": h3_passed,
            "baseline_scores_per_paper": [float(x) for x in baseline_rouge_scores],
            "micro_scores_per_paper": [float(x) for x in micro_rouge_scores],
        },
        "slod_scores": {
            "baseline_mean": float(baseline_scores.mean()),
            "micro_steered_mean": float(micro_scores.mean()),
            "macro_steered_mean": float(macro_scores.mean()),
        },
        "layer_selection": layer_selection,
        "selected_alpha": selected_alpha,
        "paper_ids": common_ids,
    }

    save_json(results, str(out_path))
    print(f"\nSaved evaluation results to {out_path}")

    # Also save per-paper SLoD scores for visualization
    scores_path = out_path.parent / "slod_scores.npz"
    np.savez(
        str(scores_path),
        baseline=baseline_scores,
        steered_micro=micro_scores,
        steered_macro=macro_scores,
        deltas_micro=micro_scores - baseline_scores,
        deltas_macro=macro_scores - baseline_scores,
    )
    print(f"Saved SLoD scores to {scores_path}")
    print("\nStage 6d complete.")


if __name__ == "__main__":
    main()
