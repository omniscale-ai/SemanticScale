#!/usr/bin/env python3
"""Stage 4d: Generate baseline and steered summaries for summarization experiment.

1. Alpha sweep on validation split (50 papers): select alpha maximizing SLoD shift
   subject to ROUGE-L drop < rouge_drop_threshold vs baseline.
2. Generate baseline, micro-steered, and macro-steered summaries for evaluation split.

GPU required. Expected runtime: ~60 minutes on L40S.

Output: data/summarization/summaries.jsonl
        data/summarization/alpha_sweep.json
"""
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import load_config, load_jsonl, save_jsonl, save_json
from src.steering import generate_with_steering
from src.embedding import embed_texts

try:
    from rouge_score import rouge_scorer as rouge_scorer_module
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: rouge_score not available, falling back to token-F1 for ROUGE-L approximation")
    from src.utils import token_f1


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F-measure between prediction and reference."""
    if HAS_ROUGE:
        scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    else:
        # Fallback: token-F1 as approximation
        return token_f1(prediction, reference)


def build_paper_prompt(paper, config, max_source_tokens=None, tokenizer=None):
    """Format a paper text into a summarization prompt (no instruction suffix)."""
    summ_cfg = config["summarization"]
    template = summ_cfg["prompt_template"]
    text = paper["text"]

    if max_source_tokens is not None and tokenizer is not None:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_source_tokens)
        text = tokenizer.decode(tokens, skip_special_tokens=True)

    return template.format(text=text)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4d: Generate baseline and steered summaries"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    summ_cfg = config["summarization"]

    out_path = Path(summ_cfg.get("summaries_path", "../../data/sh2/summarization/summaries.jsonl"))
    alpha_sweep_path = Path("../../data/sh2/summarization/alpha_sweep.json")

    if out_path.exists() and not args.force:
        print(f"Output exists: {out_path}. Use --force to rerun.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load papers + split
    papers_path = Path(summ_cfg["papers_path"])
    split_path = papers_path.parent / "split.json"

    print("Loading papers and split...")
    papers = load_jsonl(str(papers_path))

    with open(str(split_path)) as f:
        split = json.load(f)

    validation_idx = split["validation"]
    evaluation_idx = split["evaluation"]

    validation_papers = [papers[i] for i in validation_idx]
    evaluation_papers = [papers[i] for i in evaluation_idx]

    print(f"Validation: {len(validation_papers)} papers")
    print(f"Evaluation: {len(evaluation_papers)} papers")

    # 2. Load steering vectors
    sv_path = Path(summ_cfg["steering_vectors_path"])
    print(f"\nLoading steering vectors from {sv_path}")
    sv_data = np.load(str(sv_path), allow_pickle=True)
    vectors_array = sv_data["vectors"]            # (n_layers, hidden_dim)
    selected_layer = int(sv_data["selected_layer"])
    direction_flip = bool(sv_data["direction_flip"])
    print(f"Selected layer: {selected_layer}, direction_flip: {direction_flip}")

    steering_vector = vectors_array[selected_layer]   # (hidden_dim,)

    # 3. Load generative model
    model_name = config["generative_model"]["name"]
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = config["generative_model"].get("quantization")
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
    model.eval()

    max_new_tokens = config["generative_model"]["max_new_tokens"]
    temperature = config["generative_model"]["temperature"]
    max_source_tokens = summ_cfg["max_source_tokens"]
    rouge_drop_threshold = summ_cfg.get("rouge_drop_threshold", 0.05)

    # 4. Load SLoD eval axis for alpha selection
    print("Loading SLoD eval axis...")
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    eval_axis = axis_data["centroid_axis"]

    scibert_model = config["scibert_model"]
    scibert_batch = config["scibert_batch_size"]
    scibert_maxlen = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch, scibert_maxlen)

    # 5. Alpha sweep on validation set
    alpha_values = summ_cfg.get("alpha_values", [0.1, 0.2, 0.5, 1.0, 2.0])
    print(f"\nAlpha sweep on {len(validation_papers)} validation papers...")
    print(f"Testing alphas: {alpha_values}")

    val_prompts = [
        build_paper_prompt(p, config, max_source_tokens, tokenizer)
        for p in validation_papers
    ]
    val_references = [p["micro_reference"] for p in validation_papers]

    # Generate baseline for validation set
    print("Generating baseline summaries for validation set...")
    baseline_val_outputs = generate_with_steering(
        model, tokenizer, val_prompts,
        steering_vector=None, alpha=0.0,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    baseline_val_texts = [r["text"] for r in baseline_val_outputs]
    baseline_val_embs = embed_fn(baseline_val_texts)
    baseline_val_proj = baseline_val_embs @ eval_axis

    # Compute baseline ROUGE-L for reference
    baseline_rouge_vals = [
        compute_rouge_l(pred, ref)
        for pred, ref in zip(baseline_val_texts, val_references)
    ]
    baseline_rouge_mean = float(np.mean(baseline_rouge_vals))
    print(f"Baseline mean ROUGE-L on val: {baseline_rouge_mean:.4f}")

    torch.cuda.empty_cache()

    # Determine hook direction for "micro intent"
    # direction_flip=True means micro instruction -> macro hook (subtract vector)
    # direction_flip=False means micro instruction -> micro hook (add vector)
    micro_hook_direction = "macro" if direction_flip else "micro"

    alpha_sweep_results = {}
    for alpha in alpha_values:
        print(f"  Testing alpha={alpha}...")
        steered_val_outputs = generate_with_steering(
            model, tokenizer, val_prompts,
            steering_vector=steering_vector, alpha=alpha,
            direction=micro_hook_direction, layer_idx=selected_layer,
            max_new_tokens=max_new_tokens, temperature=temperature
        )
        steered_val_texts = [r["text"] for r in steered_val_outputs]
        steered_val_embs = embed_fn(steered_val_texts)
        steered_val_proj = steered_val_embs @ eval_axis

        mean_shift = float((steered_val_proj - baseline_val_proj).mean())

        # Compute ROUGE-L drop
        steered_rouge_vals = [
            compute_rouge_l(pred, ref)
            for pred, ref in zip(steered_val_texts, val_references)
        ]
        steered_rouge_mean = float(np.mean(steered_rouge_vals))
        rouge_drop = baseline_rouge_mean - steered_rouge_mean

        alpha_sweep_results[alpha] = {
            "shift": mean_shift,
            "rouge_l": steered_rouge_mean,
            "rouge_drop": rouge_drop,
        }
        print(f"    alpha={alpha}: SLoD shift={mean_shift:.4f}, ROUGE-L={steered_rouge_mean:.4f}, drop={rouge_drop:.4f}")
        torch.cuda.empty_cache()

    # Select best alpha: maximize SLoD shift subject to ROUGE-L drop < threshold
    valid_alphas = [
        a for a in alpha_values
        if alpha_sweep_results[a]["rouge_drop"] < rouge_drop_threshold
    ]
    if valid_alphas:
        # Among valid alphas, pick the one with largest SLoD shift
        best_alpha = max(valid_alphas, key=lambda a: alpha_sweep_results[a]["shift"])
    else:
        # If none pass the constraint, pick the one with smallest drop
        best_alpha = min(alpha_values, key=lambda a: alpha_sweep_results[a]["rouge_drop"])
        print(f"Warning: no alpha satisfies ROUGE-L drop < {rouge_drop_threshold}. "
              f"Selecting alpha with smallest drop: {best_alpha}")

    print(f"\nSelected alpha: {best_alpha} "
          f"(shift={alpha_sweep_results[best_alpha]['shift']:.4f}, "
          f"rouge_drop={alpha_sweep_results[best_alpha]['rouge_drop']:.4f})")

    # Save alpha sweep
    alpha_sweep_json = {
        "alpha_values": alpha_values,
        "results": {str(a): v for a, v in alpha_sweep_results.items()},
        "best_alpha": float(best_alpha),
        "selected_layer": selected_layer,
        "direction_flip": direction_flip,
        "micro_hook_direction": micro_hook_direction,
        "baseline_rouge_l": baseline_rouge_mean,
        "rouge_drop_threshold": rouge_drop_threshold,
        "validation_n": len(validation_papers),
    }
    save_json(alpha_sweep_json, str(alpha_sweep_path))
    print(f"Saved alpha sweep to {alpha_sweep_path}")

    # 6. Generate summaries for evaluation set
    print(f"\nGenerating summaries for {len(evaluation_papers)} evaluation papers...")
    all_records = []

    eval_prompts = [
        build_paper_prompt(p, config, max_source_tokens, tokenizer)
        for p in evaluation_papers
    ]

    # Baseline (no steering)
    print("  Generating baseline summaries...")
    for i, (paper, prompt) in enumerate(zip(evaluation_papers, eval_prompts)):
        if i % 20 == 0:
            print(f"    Baseline: {i}/{len(evaluation_papers)}")
        outputs = generate_with_steering(
            model, tokenizer, [prompt],
            steering_vector=None, alpha=0.0,
            max_new_tokens=max_new_tokens, temperature=temperature
        )
        all_records.append({
            "paper_id": paper["paper_id"],
            "condition": "baseline",
            "summary": outputs[0]["text"],
            "n_tokens": outputs[0]["n_tokens"],
            "alpha": None,
        })
    torch.cuda.empty_cache()

    # Micro-steered
    print("  Generating micro-steered summaries...")
    for i, (paper, prompt) in enumerate(zip(evaluation_papers, eval_prompts)):
        if i % 20 == 0:
            print(f"    Micro: {i}/{len(evaluation_papers)}")
        outputs = generate_with_steering(
            model, tokenizer, [prompt],
            steering_vector=steering_vector, alpha=best_alpha,
            direction=micro_hook_direction, layer_idx=selected_layer,
            max_new_tokens=max_new_tokens, temperature=temperature
        )
        all_records.append({
            "paper_id": paper["paper_id"],
            "condition": "micro",
            "summary": outputs[0]["text"],
            "n_tokens": outputs[0]["n_tokens"],
            "alpha": float(best_alpha),
        })
    torch.cuda.empty_cache()

    # Macro-steered (opposite direction)
    macro_hook_direction = "micro" if direction_flip else "macro"
    print("  Generating macro-steered summaries...")
    for i, (paper, prompt) in enumerate(zip(evaluation_papers, eval_prompts)):
        if i % 20 == 0:
            print(f"    Macro: {i}/{len(evaluation_papers)}")
        outputs = generate_with_steering(
            model, tokenizer, [prompt],
            steering_vector=steering_vector, alpha=best_alpha,
            direction=macro_hook_direction, layer_idx=selected_layer,
            max_new_tokens=max_new_tokens, temperature=temperature
        )
        all_records.append({
            "paper_id": paper["paper_id"],
            "condition": "macro",
            "summary": outputs[0]["text"],
            "n_tokens": outputs[0]["n_tokens"],
            "alpha": float(best_alpha),
        })
    torch.cuda.empty_cache()

    # 7. Save summaries
    save_jsonl(all_records, str(out_path))
    n_eval = len(evaluation_papers)
    print(f"\nSaved {len(all_records)} records to {out_path}")
    print(f"  {n_eval} baseline + {n_eval} micro + {n_eval} macro = {3 * n_eval} total")
    print("\nStage 4d complete.")


if __name__ == "__main__":
    main()
