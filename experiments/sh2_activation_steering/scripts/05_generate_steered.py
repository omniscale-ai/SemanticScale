#!/usr/bin/env python3
"""Stage E: Generate steered answers (micro and macro directions).

Uses the steering vector from Stage C to generate answers at different
abstraction levels. Includes alpha sweep for strength selection.

GPU required. Expected runtime: 30-60 minutes on A100.

Output: data/steered_answers.jsonl, data/results/alpha_sweep.json
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


def format_prompt(question: dict, config: dict) -> str:
    """Format a question dict into a prompt string using the config template."""
    context = " ".join(question.get("gold_evidence", []))
    template = config["generation"]["prompt_template"]
    return template.format(context=context, question=question["question_text"])


def main():
    parser = argparse.ArgumentParser(description="Stage E: Generate steered answers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "steered_answers.jsonl"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["data_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load steering vectors
    sv_path = Path(config["data_dir"]) / "steering_vectors.npz"
    print(f"Loading steering vectors from {sv_path}")
    sv_data = np.load(str(sv_path), allow_pickle=True)
    vectors_array = sv_data["vectors"]           # (n_layers, hidden_dim)
    selected_layer = int(sv_data["selected_layer"])
    candidate_layers = sv_data["candidate_layers"].tolist()
    print(f"Selected layer: {selected_layer}")

    steering_vector = vectors_array[selected_layer]  # (hidden_dim,)

    # 2. Load generative model
    model_name = config["generative_model"]["name"]
    print(f"Loading model: {model_name}")
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
    batch_size = config["generative_model"]["batch_size"]

    # 3. Load questions
    print("Loading questions...")
    questions = load_jsonl(config["sh5_questions"])
    n_questions = config["generation"]["n_questions"]
    questions = questions[:n_questions]
    print(f"Total questions: {len(questions)}")

    # 4. Load SLoD evaluation axis for alpha selection
    print("Loading SLoD eval axis...")
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    eval_axis = axis_data["centroid_axis"]

    scibert_model = config["scibert_model"]
    scibert_batch_size = config["scibert_batch_size"]
    scibert_max_length = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch_size, scibert_max_length)

    # Read direction_flip flag
    direction_flip = config["steering"].get("direction_flip", False)
    if direction_flip:
        print("direction_flip=True: micro<->macro hook directions swapped")

    # 5. Alpha sweep on first 50 questions (validation subset)
    val_n = config["steering"]["validation_n"]
    val_questions = questions[:val_n]
    val_prompts = [format_prompt(q, config) for q in val_questions]
    alpha_values = config["steering"]["alpha_values"]

    print(f"\nAlpha sweep on {len(val_prompts)} validation questions...")
    print(f"Testing alphas: {alpha_values}")

    # Generate baseline for validation set once
    print("Generating baseline for validation set...")
    baseline_outputs = generate_with_steering(
        model, tokenizer, val_prompts,
        steering_vector=None, alpha=0.0,
        max_new_tokens=min(max_new_tokens, 256), temperature=temperature
    )
    baseline_texts = [r["text"] for r in baseline_outputs]
    baseline_embs = embed_fn(baseline_texts)
    baseline_proj = baseline_embs @ eval_axis

    torch.cuda.empty_cache()

    alpha_sweep_results = {}
    for alpha in alpha_values:
        print(f"  Testing alpha={alpha}...")
        # Use flipped hook direction for the sweep (sweep always probes "micro" intent)
        sweep_hook_direction = "macro" if direction_flip else "micro"
        steered_outputs = generate_with_steering(
            model, tokenizer, val_prompts,
            steering_vector=steering_vector, alpha=alpha,
            direction=sweep_hook_direction, layer_idx=selected_layer,
            max_new_tokens=min(max_new_tokens, 256), temperature=temperature
        )
        steered_texts = [r["text"] for r in steered_outputs]
        steered_embs = embed_fn(steered_texts)
        steered_proj = steered_embs @ eval_axis
        mean_shift = float((steered_proj - baseline_proj).mean())
        alpha_sweep_results[alpha] = mean_shift
        print(f"    alpha={alpha}: mean SLoD shift = {mean_shift:.4f}")
        torch.cuda.empty_cache()

    # 6. Select best alpha (largest absolute shift)
    best_alpha = max(alpha_values, key=lambda a: abs(alpha_sweep_results[a]))
    print(f"\nSelected alpha: {best_alpha} (shift={alpha_sweep_results[best_alpha]:.4f})")

    alpha_sweep_json = {
        "alpha_values": alpha_values,
        "shifts": {str(a): float(s) for a, s in alpha_sweep_results.items()},
        "best_alpha": float(best_alpha),
        "selected_layer": selected_layer,
        "validation_n": val_n,
    }
    save_json(alpha_sweep_json, str(results_dir / "alpha_sweep.json"))
    print(f"Saved alpha sweep results to {results_dir / 'alpha_sweep.json'}")

    # 7. Generate steered answers for all questions × 2 directions at best_alpha
    directions = config["steering"]["directions"]
    all_records = []

    for direction in directions:
        # Flip hook direction if needed: "micro" intent uses "macro" hook and vice-versa
        if direction_flip:
            hook_direction = "macro" if direction == "micro" else "micro"
        else:
            hook_direction = direction
        print(f"\nGenerating {direction} steered answers for {len(questions)} questions "
              f"(hook_direction={hook_direction})...")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i: i + batch_size]
            prompts = [format_prompt(q, config) for q in batch_questions]

            outputs = generate_with_steering(
                model, tokenizer, prompts,
                steering_vector=steering_vector, alpha=best_alpha,
                direction=hook_direction, layer_idx=selected_layer,
                max_new_tokens=max_new_tokens, temperature=temperature
            )

            for q, out in zip(batch_questions, outputs):
                record = {
                    "question_id": q["question_id"],
                    "question_text": q["question_text"],
                    "gold_answer_text": q.get("gold_answer_text", ""),
                    "answer": out["text"],
                    "n_tokens": out["n_tokens"],
                    "condition": f"steered_{direction}",
                    "direction": direction,
                    "alpha": best_alpha,
                    "layer_idx": selected_layer,
                }
                all_records.append(record)

            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(questions):
                print(f"  {direction}: {min(i + batch_size, len(questions))}/{len(questions)}")
            torch.cuda.empty_cache()

    # 8. Save steered answers
    save_jsonl(all_records, str(output_path))
    print(f"\nSaved {len(all_records)} steered answers to {output_path}")
    print(f"  ({len(all_records) // len(directions)} per direction × {len(directions)} directions)")


if __name__ == "__main__":
    main()
