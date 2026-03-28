#!/usr/bin/env python3
"""Stage D: Generate baseline answers (no steering).

Generates answers to 500 QASPER questions using the generative model
without any activation steering. These serve as the control condition.

GPU required. Expected runtime: ~15 minutes on A100.

Output: data/baseline_answers.jsonl
"""
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import load_config, load_jsonl, save_jsonl
from src.steering import generate_with_steering


def format_prompt(question: dict, config: dict) -> str:
    """Format a question dict into a prompt string using the config template."""
    context = " ".join(question.get("gold_evidence", []))
    template = config["generation"]["prompt_template"]
    return template.format(context=context, question=question["question_text"])


def main():
    parser = argparse.ArgumentParser(description="Stage D: Generate baseline answers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "baseline_answers.jsonl"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load questions
    print("Loading questions...")
    questions = load_jsonl(config["sh5_questions"])
    n_questions = config["generation"]["n_questions"]
    questions = questions[:n_questions]
    print(f"Loaded {len(questions)} questions")

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

    # 3. Generate baseline answers (no steering)
    print(f"Generating baseline answers for {len(questions)} questions...")
    records = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i: i + batch_size]
        prompts = [format_prompt(q, config) for q in batch_questions]

        outputs = generate_with_steering(
            model, tokenizer, prompts,
            steering_vector=None, alpha=0.0,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        for q, out in zip(batch_questions, outputs):
            record = {
                "question_id": q["question_id"],
                "question_text": q["question_text"],
                "gold_answer_text": q.get("gold_answer_text", ""),
                "answer": out["text"],
                "n_tokens": out["n_tokens"],
                "condition": "baseline",
            }
            records.append(record)

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(questions):
            print(f"  Generated {min(i + batch_size, len(questions))}/{len(questions)}")
        torch.cuda.empty_cache()

    # 4. Save
    save_jsonl(records, str(output_path))
    print(f"Saved {len(records)} baseline answers to {output_path}")


if __name__ == "__main__":
    main()
