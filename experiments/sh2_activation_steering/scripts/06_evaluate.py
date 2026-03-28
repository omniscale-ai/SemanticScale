#!/usr/bin/env python3
"""Stage F: Evaluate baseline vs steered answers.

Measures SLoD shift, surface metrics, and factuality preservation.

Output: data/results/evaluation_results.json
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_jsonl, save_json, token_f1
from src.embedding import embed_texts
from src.evaluate import (
    compute_slod_scores,
    compute_slod_shift,
    compute_surface_metrics,
    compare_surface_metrics,
    evaluate_factuality,
)


def main():
    parser = argparse.ArgumentParser(description="Stage F: Evaluate")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "results" / "evaluation_results.json"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading baseline answers...")
    baseline_records = load_jsonl(str(Path(config["data_dir"]) / "baseline_answers.jsonl"))

    print("Loading steered answers...")
    steered_records = load_jsonl(str(Path(config["data_dir"]) / "steered_answers.jsonl"))

    print("Loading SLoD evaluation axis...")
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    slod_axis = axis_data["centroid_axis"]

    # Split steered records by direction
    steered_micro = [r for r in steered_records if r.get("direction") == "micro"]
    steered_macro = [r for r in steered_records if r.get("direction") == "macro"]

    # Align by question_id (baseline is the reference)
    baseline_by_id = {r["question_id"]: r for r in baseline_records}
    question_ids = [r["question_id"] for r in baseline_records]

    # Build aligned lists
    baseline_answers = [baseline_by_id[qid]["answer"] for qid in question_ids]
    gold_answers = [baseline_by_id[qid].get("gold_answer_text", "") for qid in question_ids]

    steered_micro_by_id = {r["question_id"]: r for r in steered_micro}
    steered_macro_by_id = {r["question_id"]: r for r in steered_macro}

    # Keep only questions present in all conditions
    common_ids = [
        qid for qid in question_ids
        if qid in steered_micro_by_id and qid in steered_macro_by_id
    ]
    print(f"Common questions across all conditions: {len(common_ids)}")

    baseline_answers_aligned = [baseline_by_id[qid]["answer"] for qid in common_ids]
    gold_answers_aligned = [baseline_by_id[qid].get("gold_answer_text", "") for qid in common_ids]
    steered_micro_answers = [steered_micro_by_id[qid]["answer"] for qid in common_ids]
    steered_macro_answers = [steered_macro_by_id[qid]["answer"] for qid in common_ids]

    # --- Embed all answers ---
    scibert_model = config["scibert_model"]
    scibert_batch_size = config["scibert_batch_size"]
    scibert_max_length = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch_size, scibert_max_length)

    print("\nEmbedding baseline answers...")
    baseline_embs = embed_fn(baseline_answers_aligned)
    baseline_scores = baseline_embs @ slod_axis

    print("Embedding steered_micro answers...")
    micro_embs = embed_fn(steered_micro_answers)
    micro_scores = micro_embs @ slod_axis

    print("Embedding steered_macro answers...")
    macro_embs = embed_fn(steered_macro_answers)
    macro_scores = macro_embs @ slod_axis

    # --- H1: SLoD shift ---
    print("\nH1: Computing SLoD shift...")
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
    print(f"  H1 passed: {h1_passed}")

    # --- H2: Surface metrics ---
    print("\nH2: Computing surface metrics...")
    baseline_surface = [compute_surface_metrics(t) for t in baseline_answers_aligned]
    micro_surface = [compute_surface_metrics(t) for t in steered_micro_answers]

    h2_comparison = compare_surface_metrics(baseline_surface, micro_surface)
    h2_min_sig = thresholds.get("h2_min_metrics", 2)
    h2_passed = h2_comparison["n_significant"] >= h2_min_sig
    print(f"  Significant metrics: {h2_comparison['n_significant']}/4")
    print(f"  H2 passed: {h2_passed}")

    # --- H3: Factuality ---
    print("\nH3: Computing factuality (token-F1)...")
    baseline_factuality = evaluate_factuality(baseline_answers_aligned, gold_answers_aligned, token_f1)
    micro_factuality = evaluate_factuality(steered_micro_answers, gold_answers_aligned, token_f1)
    macro_factuality = evaluate_factuality(steered_macro_answers, gold_answers_aligned, token_f1)

    max_quality_drop = thresholds.get("h3_max_quality_drop", 0.05)
    f1_drop_micro = baseline_factuality["mean_f1"] - micro_factuality["mean_f1"]
    h3_passed = f1_drop_micro <= max_quality_drop

    print(f"  Baseline token-F1: {baseline_factuality['mean_f1']:.4f}")
    print(f"  Steered micro token-F1: {micro_factuality['mean_f1']:.4f}")
    print(f"  Drop: {f1_drop_micro:.4f} (threshold: {max_quality_drop})")
    print(f"  H3 passed: {h3_passed}")

    # --- Verdict ---
    if h1_passed and h3_passed:
        verdict = "CONFIRMED"
    elif h1_passed and not h3_passed:
        verdict = "PARTIAL"
    else:
        verdict = "NOT CONFIRMED"
    print(f"\nVERDICT: {verdict}")

    # --- Load layer selection for results ---
    layer_selection_path = Path(config["data_dir"]) / "results" / "layer_selection.json"
    layer_selection = {}
    if layer_selection_path.exists():
        import json
        with open(str(layer_selection_path)) as f:
            layer_selection = json.load(f)

    alpha_sweep_path = Path(config["data_dir"]) / "results" / "alpha_sweep.json"
    selected_alpha = None
    if alpha_sweep_path.exists():
        import json
        with open(str(alpha_sweep_path)) as f:
            alpha_data = json.load(f)
        selected_alpha = alpha_data.get("best_alpha")

    # --- Save results ---
    results = {
        "verdict": verdict,
        "n_questions": len(common_ids),
        "h1_passed": h1_passed,
        "h2_passed": h2_passed,
        "h3_passed": h3_passed,
        "h1_micro": h1_micro,
        "h1_macro": h1_macro,
        "h2_surface": h2_comparison,
        "h3_factuality": {
            "baseline_mean_f1": float(baseline_factuality["mean_f1"]),
            "steered_micro_mean_f1": float(micro_factuality["mean_f1"]),
            "steered_macro_mean_f1": float(macro_factuality["mean_f1"]),
            "f1_drop_micro": float(f1_drop_micro),
            "baseline_scores_per_question": baseline_factuality["scores"],
            "micro_scores_per_question": micro_factuality["scores"],
            "macro_scores_per_question": macro_factuality["scores"],
        },
        "slod_scores": {
            "baseline_mean": float(baseline_scores.mean()),
            "micro_steered_mean": float(micro_scores.mean()),
            "macro_steered_mean": float(macro_scores.mean()),
        },
        "layer_selection": layer_selection,
        "selected_alpha": selected_alpha,
        "question_ids": common_ids,
    }

    save_json(results, str(output_path))
    print(f"\nSaved evaluation results to {output_path}")

    # Also save per-answer SLoD scores for visualization
    slod_scores_path = Path(config["data_dir"]) / "results" / "slod_scores.npz"
    np.savez(
        str(slod_scores_path),
        baseline=baseline_scores,
        steered_micro=micro_scores,
        steered_macro=macro_scores,
        deltas_micro=micro_scores - baseline_scores,
        deltas_macro=macro_scores - baseline_scores,
    )
    print(f"Saved SLoD scores to {slod_scores_path}")


if __name__ == "__main__":
    main()
