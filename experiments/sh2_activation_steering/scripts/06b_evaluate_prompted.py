#!/usr/bin/env python3
"""Stage F-alt: Evaluate prompt-based SLoD control (SH2a).

Uses same evaluation pipeline as 06_evaluate.py but on prompted_answers.jsonl.
Compares: baseline vs micro-instructed and baseline vs macro-instructed.

Output: data/results/evaluation_results_prompted.json
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, load_jsonl, load_json, save_json, token_f1
from src.embedding import embed_texts
from src.evaluate import (
    compute_slod_scores, compute_slod_shift,
    compute_surface_metrics, compare_surface_metrics,
    evaluate_factuality
)


def main():
    parser = argparse.ArgumentParser(description="Stage F-alt: Evaluate prompted answers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "results" / "evaluation_results_prompted.json"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading answers...")
    baseline = load_jsonl(config["data_dir"] + "/baseline_answers.jsonl")
    prompted = load_jsonl(config["data_dir"] + "/prompted_answers.jsonl")

    # Index by question_id
    baseline_by_id = {r["question_id"]: r for r in baseline}
    prompted_micro = {r["question_id"]: r for r in prompted if r["direction"] == "micro"}
    prompted_macro = {r["question_id"]: r for r in prompted if r["direction"] == "macro"}

    # Find common questions
    common_ids = sorted(
        set(baseline_by_id.keys()) & set(prompted_micro.keys()) & set(prompted_macro.keys())
    )
    print(f"Common questions: {len(common_ids)}")

    # baseline uses "answer" field (not "answer_text")
    baseline_texts = [baseline_by_id[qid].get("answer_text", baseline_by_id[qid].get("answer", "")) for qid in common_ids]
    micro_texts = [prompted_micro[qid]["answer_text"] for qid in common_ids]
    macro_texts = [prompted_macro[qid]["answer_text"] for qid in common_ids]

    # Load SLoD evaluation axis
    print("Loading SLoD axis...")
    import numpy as np
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    slod_axis = axis_data["centroid_axis"]

    # Embed all answers
    scibert_model = config["scibert_model"]
    batch_size = config["scibert_batch_size"]
    max_length = config["scibert_max_length"]

    print("Embedding baseline answers...")
    def embed_fn(texts):
        return embed_texts(texts, scibert_model, batch_size, max_length)

    baseline_scores = compute_slod_scores(baseline_texts, embed_fn, slod_axis)
    print("Embedding micro-instructed answers...")
    micro_scores = compute_slod_scores(micro_texts, embed_fn, slod_axis)
    print("Embedding macro-instructed answers...")
    macro_scores = compute_slod_scores(macro_texts, embed_fn, slod_axis)

    # Save scores for later analysis
    np.savez(
        config["data_dir"] + "/results/slod_scores_prompted.npz",
        baseline=baseline_scores,
        prompted_micro=micro_scores,
        prompted_macro=macro_scores,
    )

    # H1: SLoD shift
    print("\nComputing H1 (SLoD shift)...")
    micro_shift = compute_slod_shift(baseline_scores, micro_scores)
    macro_shift = compute_slod_shift(baseline_scores, macro_scores)

    h1_p = config["thresholds"]["h1_shift_p"]
    h1_d = config["thresholds"]["h1_shift_d"]
    h1_pass = (
        micro_shift["p_value"] < h1_p and abs(micro_shift["cohens_d"]) > h1_d
        and micro_shift["mean_delta"] > 0  # must be in the right direction
    )

    print(f"  Micro: delta={micro_shift['mean_delta']:.4f}, d={micro_shift['cohens_d']:.4f}, p={micro_shift['p_value']:.4e}")
    print(f"  Macro: delta={macro_shift['mean_delta']:.4f}, d={macro_shift['cohens_d']:.4f}, p={macro_shift['p_value']:.4e}")

    # H2: Surface metrics
    print("\nComputing H2 (surface metrics)...")
    baseline_surface = [compute_surface_metrics(t) for t in baseline_texts]
    micro_surface = [compute_surface_metrics(t) for t in micro_texts]
    macro_surface = [compute_surface_metrics(t) for t in macro_texts]

    micro_surface_cmp = compare_surface_metrics(baseline_surface, micro_surface)
    h2_pass = micro_surface_cmp["n_significant"] >= config["thresholds"]["h2_min_metrics"]
    print(f"  Micro significant: {micro_surface_cmp['n_significant']}/4")

    # H3: Factuality
    print("\nComputing H3 (factuality)...")
    questions = load_jsonl(config["sh5_questions"])
    gold_by_id = {q["question_id"]: q.get("gold_answer_text", "") for q in questions}
    gold_texts = [gold_by_id.get(qid, "") for qid in common_ids]

    baseline_f1 = evaluate_factuality(baseline_texts, gold_texts, token_f1)
    micro_f1 = evaluate_factuality(micro_texts, gold_texts, token_f1)
    macro_f1 = evaluate_factuality(macro_texts, gold_texts, token_f1)

    f1_drop = baseline_f1["mean_f1"] - micro_f1["mean_f1"]
    h3_pass = f1_drop < config["thresholds"]["h3_max_quality_drop"]
    print(f"  Baseline F1: {baseline_f1['mean_f1']:.4f}")
    print(f"  Micro F1:    {micro_f1['mean_f1']:.4f} (drop: {f1_drop:.4f})")
    print(f"  Macro F1:    {macro_f1['mean_f1']:.4f}")

    # Verdict
    if h1_pass and h3_pass:
        verdict = "CONFIRMED"
    elif h1_pass and not h3_pass:
        verdict = "PARTIAL"
    else:
        verdict = "NOT CONFIRMED"

    print(f"\nVerdict: {verdict}")

    # Save results
    results = {
        "method": "prompt_instruction",
        "n_questions": len(common_ids),
        "h1_slod_shift": {
            "steered_micro": micro_shift,
            "steered_macro": macro_shift,
            "pass": h1_pass,
        },
        "h2_surface": {
            "steered_micro": micro_surface_cmp,
            "n_significant": micro_surface_cmp["n_significant"],
            "pass": h2_pass,
        },
        "h3_factuality": {
            "baseline_mean_f1": float(baseline_f1["mean_f1"]),
            "steered_micro_mean_f1": float(micro_f1["mean_f1"]),
            "steered_macro_mean_f1": float(macro_f1["mean_f1"]),
            "max_drop": float(f1_drop),
            "pass": h3_pass,
        },
        "overall_verdict": verdict,
        "micro_instruction": "Be detailed, include numbers/methods/findings",
        "macro_instruction": "Be brief and high-level, no specific details",
    }

    save_json(results, str(output_path))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
