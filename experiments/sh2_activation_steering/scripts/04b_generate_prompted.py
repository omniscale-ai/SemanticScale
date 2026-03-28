#!/usr/bin/env python3
"""Stage D-alt: Generate prompted answers (SH2a: prompt-based SLoD control).

Uses explicit SLoD-level instructions in the prompt instead of activation steering.
This is the upper-bound experiment: how much SLoD shift can explicit prompting achieve?

Output: data/prompted_answers.jsonl
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import load_config, load_jsonl, save_jsonl


MACRO_INSTRUCTION = (
    "Provide a brief, high-level answer in one or two sentences. "
    "Focus on the main idea only. Do not include specific numbers, "
    "technical details, or citations."
)

MICRO_INSTRUCTION = (
    "Provide a detailed technical answer. Include specific numbers, "
    "methods, experimental findings, and cite evidence from the context "
    "where relevant. Be as specific as possible."
)


def format_prompted_qa(question: dict, instruction: str, config: dict) -> str:
    """Format a QA prompt with an explicit SLoD-level instruction."""
    q_text = question.get("question_text", "")
    evidence = question.get("gold_evidence", [])
    if isinstance(evidence, list):
        context = " ".join(str(e) for e in evidence)
    else:
        context = str(evidence)
    # Inject instruction into the prompt template
    template = config["generation"]["prompt_template"]
    base = template.format(context=context, question=q_text)
    return base + f"\n\nInstruction: {instruction}"


def main():
    parser = argparse.ArgumentParser(description="Stage D-alt: Generate prompted answers (SH2a)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "prompted_answers.jsonl"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load questions
    print("Loading questions...")
    questions = load_jsonl(config["sh5_questions"])
    n_questions = config["generation"]["n_questions"]
    questions = questions[:n_questions]
    print(f"Loaded {len(questions)} questions")

    # Load model
    model_name = config["generative_model"]["name"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    device = next(model.parameters()).device

    max_new_tokens = config["generative_model"]["max_new_tokens"]

    results = []

    for direction, instruction in [("micro", MICRO_INSTRUCTION), ("macro", MACRO_INSTRUCTION)]:
        print(f"\nGenerating {direction}-instructed answers for {len(questions)} questions...")
        for i, q in enumerate(questions):
            if i % 10 == 0:
                print(f"  {direction}: {i}/{len(questions)}")

            prompt = format_prompted_qa(q, instruction, config)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0, input_len:]
            answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            n_tokens = len(new_tokens)

            results.append({
                "question_id": q.get("question_id", f"q{i}"),
                "direction": direction,
                "answer_text": answer_text,
                "n_tokens": int(n_tokens),
                "answer_type": q.get("answer_type", "abstractive"),
                "method": "prompt_instruction",
            })

    save_jsonl(results, str(output_path))
    print(f"\nSaved {len(results)} prompted answers to {output_path}")
    print(f"  ({len(questions)} per direction × 2 directions)")


if __name__ == "__main__":
    main()
