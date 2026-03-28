#!/usr/bin/env python3
"""Re-evaluate SH2-summ with LLM-derived SLoD axis.

Same evaluation as 06d but uses a new centroid axis computed from
LLM-labeled spans (section-blind) instead of SH0 section-derived labels.

No GPU needed — reuses cached summaries, only re-embeds + re-scores.

Usage:
    cd playground/SLoD-SH2
    python scripts/06e_evaluate_summaries_llm_axis.py
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_jsonl, load_json, save_json
from src.embedding import embed_texts
from src.slod_axis import compute_centroid_axis, compute_lda_axis, validate_axis
from src.evaluate import compute_slod_shift, compute_surface_metrics, compare_surface_metrics

try:
    from rouge_score import rouge_scorer as rouge_scorer_module
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    from src.utils import token_f1

SH1_ROOT = Path(__file__).resolve().parent.parent.parent / "sh1_linear_probe"


def compute_rouge_l(prediction: str, reference: str) -> float:
    if HAS_ROUGE:
        scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    else:
        return token_f1(prediction, reference)


def build_llm_axis(config):
    """Compute new centroid axis from LLM-labeled spans."""
    print("=" * 60)
    print("STEP 1: Computing LLM-derived SLoD axis")
    print("=" * 60)

    # Load SH1 embeddings
    sh1_data = np.load(config["sh1_embeddings"])
    embeddings = sh1_data["embeddings"]
    original_labels = sh1_data["labels"]

    # Load LLM labels from SH1 relabeling
    relabel_dir = SH1_ROOT / "data" / "relabel"
    llm_labels = {}
    for p in sorted(relabel_dir.glob("result_*.json")):
        with open(p) as f:
            batch = json.load(f)
        for item in batch:
            if isinstance(item, dict) and "id" in item and "label" in item:
                label_map = {"macro": 0, "meso": 1, "micro": 2}
                label = item["label"].lower().strip()
                if label in label_map:
                    llm_labels[int(item["id"])] = label_map[label]

    print(f"  LLM labels loaded: {len(llm_labels)} spans")

    # Get indices and labels for LLM-labeled spans
    llm_indices = sorted(llm_labels.keys())
    llm_label_arr = np.array([llm_labels[i] for i in llm_indices])
    llm_emb = embeddings[llm_indices]

    # Compute centroid axis from LLM labels
    print("\n  Computing centroid axis from LLM labels...")
    llm_centroid_axis = compute_centroid_axis(llm_emb, llm_label_arr)

    # Also compute LDA axis
    print("  Computing LDA axis from LLM labels...")
    llm_lda_axis = compute_lda_axis(llm_emb, llm_label_arr)

    # Compare with original axis
    original_axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    original_centroid = original_axis_data["centroid_axis"]

    cosine_sim = float(np.dot(llm_centroid_axis, original_centroid))
    print(f"\n  Cosine similarity (LLM vs SH0 centroid axis): {cosine_sim:.4f}")

    # Validate on SH1 test set (with original labels for comparability)
    with open(config["sh1_splits"]) as f:
        splits = json.load(f)
    test_idx = np.array(splits["test"])
    test_emb = embeddings[test_idx]
    test_labels = original_labels[test_idx]

    print("\n  Validating LLM centroid axis on SH1 test set...")
    llm_stats = validate_axis(test_emb, test_labels, llm_centroid_axis)

    print("\n  Validating original centroid axis on SH1 test set...")
    orig_stats = validate_axis(test_emb, test_labels, original_centroid)

    return llm_centroid_axis, llm_lda_axis, {
        "cosine_similarity": cosine_sim,
        "llm_axis_validation": llm_stats,
        "original_axis_validation": orig_stats,
    }


def main():
    config = load_config()
    summ_cfg = config["summarization"]

    # Step 1: Build LLM-derived axis
    llm_axis, llm_lda_axis, axis_comparison = build_llm_axis(config)

    # Save new axis
    new_axis_path = Path(config["data_dir"]) / "eval_slod_axis_llm.npz"
    np.savez(
        str(new_axis_path),
        centroid_axis=llm_axis,
        lda_axis=llm_lda_axis,
        source="LLM-labeled spans (section-blind, Gemini-2.5-pro)",
    )
    print(f"\n  Saved LLM axis to {new_axis_path}")

    # Step 2: Load cached summaries
    print("\n" + "=" * 60)
    print("STEP 2: Re-evaluating summaries with LLM axis")
    print("=" * 60)

    summaries_path = Path(summ_cfg.get("summaries_path", "../../data/sh2/summarization/summaries.jsonl"))
    all_summaries = load_jsonl(str(summaries_path))

    baseline_records = {r["paper_id"]: r for r in all_summaries if r["condition"] == "baseline"}
    micro_records = {r["paper_id"]: r for r in all_summaries if r["condition"] == "micro"}
    macro_records = {r["paper_id"]: r for r in all_summaries if r["condition"] == "macro"}

    common_ids = sorted(set(baseline_records) & set(micro_records) & set(macro_records))
    print(f"  Papers with all 3 conditions: {len(common_ids)}")

    baseline_texts = [baseline_records[pid]["summary"] for pid in common_ids]
    micro_texts = [micro_records[pid]["summary"] for pid in common_ids]
    macro_texts = [macro_records[pid]["summary"] for pid in common_ids]

    # Load micro references
    papers = load_jsonl(str(Path(summ_cfg["papers_path"])))
    papers_by_id = {p["paper_id"]: p for p in papers}
    aligned_ids = [pid for pid in common_ids if pid in papers_by_id]
    micro_references = [papers_by_id[pid]["micro_reference"] for pid in aligned_ids]

    if len(aligned_ids) < len(common_ids):
        baseline_texts = [baseline_records[pid]["summary"] for pid in aligned_ids]
        micro_texts = [micro_records[pid]["summary"] for pid in aligned_ids]
        macro_texts = [macro_records[pid]["summary"] for pid in aligned_ids]
        common_ids = aligned_ids

    # Step 3: Embed and project with BOTH axes
    scibert_model = config["scibert_model"]
    scibert_batch = config["scibert_batch_size"]
    scibert_maxlen = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch, scibert_maxlen)

    print("\n  Embedding summaries with SciBERT...")
    baseline_embs = embed_fn(baseline_texts)
    micro_embs = embed_fn(micro_texts)
    macro_embs = embed_fn(macro_texts)

    # Original axis scores
    original_axis = np.load(config["data_dir"] + "/eval_slod_axis.npz")["centroid_axis"]
    orig_baseline_scores = baseline_embs @ original_axis
    orig_micro_scores = micro_embs @ original_axis
    orig_macro_scores = macro_embs @ original_axis

    # LLM axis scores
    llm_baseline_scores = baseline_embs @ llm_axis
    llm_micro_scores = micro_embs @ llm_axis
    llm_macro_scores = macro_embs @ llm_axis

    # Step 4: Evaluate H1, H2, H3 with both axes
    thresholds = config.get("thresholds", {})
    h1_d_thresh = thresholds.get("h1_shift_d", 0.5)
    h1_p_thresh = thresholds.get("h1_shift_p", 0.05)

    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    # H1 with original axis
    h1_orig = compute_slod_shift(orig_baseline_scores, orig_micro_scores)
    h1_orig_macro = compute_slod_shift(orig_baseline_scores, orig_macro_scores)

    # H1 with LLM axis
    h1_llm = compute_slod_shift(llm_baseline_scores, llm_micro_scores)
    h1_llm_macro = compute_slod_shift(llm_baseline_scores, llm_macro_scores)

    print(f"\n  H1: SLoD Shift (micro steering)")
    print(f"  {'Axis':<15} {'Cohen d':>10} {'p-value':>12} {'mean Δ':>10} {'Pass':>6}")
    print(f"  {'-'*55}")
    h1_orig_pass = h1_orig["p_value"] < h1_p_thresh and abs(h1_orig["cohens_d"]) >= h1_d_thresh
    h1_llm_pass = h1_llm["p_value"] < h1_p_thresh and abs(h1_llm["cohens_d"]) >= h1_d_thresh
    print(f"  {'SH0 (original)':<15} {h1_orig['cohens_d']:>10.4f} {h1_orig['p_value']:>12.2e} {h1_orig['mean_delta']:>10.4f} {'✓' if h1_orig_pass else '✗':>6}")
    print(f"  {'LLM (new)':<15} {h1_llm['cohens_d']:>10.4f} {h1_llm['p_value']:>12.2e} {h1_llm['mean_delta']:>10.4f} {'✓' if h1_llm_pass else '✗':>6}")

    # H2: Surface metrics (axis-independent, same as before)
    baseline_surface = [compute_surface_metrics(t) for t in baseline_texts]
    micro_surface = [compute_surface_metrics(t) for t in micro_texts]
    h2_comparison = compare_surface_metrics(baseline_surface, micro_surface)
    h2_min_sig = thresholds.get("h2_min_metrics", 2)
    h2_passed = h2_comparison["n_significant"] >= h2_min_sig
    print(f"\n  H2: Surface metrics: {h2_comparison['n_significant']}/4 significant → {'PASS' if h2_passed else 'FAIL'}")

    # H3: ROUGE-L (axis-independent, same as before)
    baseline_rouge = [compute_rouge_l(p, r) for p, r in zip(baseline_texts, micro_references)]
    micro_rouge = [compute_rouge_l(p, r) for p, r in zip(micro_texts, micro_references)]
    baseline_rouge_mean = float(np.mean(baseline_rouge))
    micro_rouge_mean = float(np.mean(micro_rouge))
    rouge_drop = baseline_rouge_mean - micro_rouge_mean
    rouge_thresh = summ_cfg.get("rouge_drop_threshold", 0.05)
    h3_passed = rouge_drop < rouge_thresh
    print(f"\n  H3: ROUGE-L drop = {rouge_drop:.4f} (threshold {rouge_thresh}) → {'PASS' if h3_passed else 'FAIL'}")

    # Verdicts
    orig_verdict = "CONFIRMED" if h1_orig_pass and h3_passed else ("PARTIAL" if h1_orig_pass else "NOT CONFIRMED")
    llm_verdict = "CONFIRMED" if h1_llm_pass and h3_passed else ("PARTIAL" if h1_llm_pass else "NOT CONFIRMED")

    print(f"\n  {'='*55}")
    print(f"  VERDICT (SH0 axis): {orig_verdict}")
    print(f"  VERDICT (LLM axis): {llm_verdict}")
    print(f"  {'='*55}")

    # Save results
    results = {
        "original_axis": {
            "verdict": orig_verdict,
            "h1_micro": h1_orig,
            "h1_macro": h1_orig_macro,
            "h1_passed": h1_orig_pass,
        },
        "llm_axis": {
            "verdict": llm_verdict,
            "h1_micro": h1_llm,
            "h1_macro": h1_llm_macro,
            "h1_passed": h1_llm_pass,
        },
        "h2": {"passed": h2_passed, "comparison": h2_comparison},
        "h3": {
            "passed": h3_passed,
            "baseline_rouge": baseline_rouge_mean,
            "micro_rouge": micro_rouge_mean,
            "drop": float(rouge_drop),
        },
        "axis_comparison": axis_comparison,
        "n_papers": len(common_ids),
    }

    out_path = Path(summ_cfg["evaluation_results_path"]).parent / "evaluation_results_llm_axis.json"
    save_json(results, str(out_path))
    print(f"\n  Results saved to {out_path}")

    # Report
    report_lines = [
        "# SH2-summ Re-evaluation with LLM-derived SLoD Axis",
        "",
        f"> Axis cosine similarity (LLM vs SH0): {axis_comparison['cosine_similarity']:.4f}",
        "",
        "## H1: SLoD Shift Comparison",
        "",
        "| Axis | Cohen's d | p-value | Mean Δ | Pass |",
        "|---|---|---|---|---|",
        f"| SH0 (original) | {h1_orig['cohens_d']:.4f} | {h1_orig['p_value']:.2e} | {h1_orig['mean_delta']:.4f} | {'PASS' if h1_orig_pass else 'FAIL'} |",
        f"| **LLM (new)** | **{h1_llm['cohens_d']:.4f}** | {h1_llm['p_value']:.2e} | {h1_llm['mean_delta']:.4f} | {'PASS' if h1_llm_pass else 'FAIL'} |",
        "",
        f"## H2: Surface Metrics → {'PASS' if h2_passed else 'FAIL'} ({h2_comparison['n_significant']}/4 significant)",
        "",
        f"## H3: ROUGE-L → {'PASS' if h3_passed else 'FAIL'} (drop = {rouge_drop:.4f})",
        "",
        f"## Verdict",
        "",
        f"- SH0 axis: **{orig_verdict}**",
        f"- LLM axis: **{llm_verdict}**",
    ]

    report_path = Path(__file__).resolve().parent.parent / "reports" / "SH2_summ_LLM_AXIS_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    print(f"  Report saved to {report_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
