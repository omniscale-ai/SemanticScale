#!/usr/bin/env python3
"""Stage C (SH2b): Compute QA-context contrastive steering vector.

Instead of using SH0 document spans, this version computes the steering vector
from contrastive QA-context hidden states (macro vs micro instruction pairs).
This fixes the task-domain mismatch hypothesis.

GPU required. Expected runtime: ~30 minutes on L40S.

Output: data/steering_vectors.npz
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_jsonl, save_json
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.steering import compute_steering_vectors, select_best_layer
from src.embedding import embed_texts


def extract_qa_hidden_states(model, tokenizer, prompts, layers, device):
    """Extract hidden states from the answer portion of QA generation.

    For each prompt, do a forward pass and extract hidden states at each layer,
    mean-pooled over the non-prompt tokens (i.e. we first tokenize prompt-only,
    then full prompt, and extract hidden states from the suffix).

    Since we want the model's QA representation, we extract from the last
    token of the prompt (the token right before it would generate), which
    captures the model's state after processing the full QA context.
    Actually simpler and more robust: extract from the LAST TOKEN POSITION
    of the input, which is the model's summary state of the entire input.
    """
    hidden_by_layer = {l: [] for l in layers}

    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                print(f"  Extracting hidden states: {i}/{len(prompts)}")

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            # hidden_states is a tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
            # Use the last token position as the summary state
            for l in layers:
                hs = outputs.hidden_states[l + 1]  # +1 because index 0 is embedding
                last_token = hs[0, -1, :].float().cpu().numpy()  # (hidden_dim,)
                hidden_by_layer[l].append(last_token)

    # Convert lists to arrays
    return {l: np.array(hidden_by_layer[l]) for l in layers}


def format_qa_prompt(question, config):
    """Format a question into a QA prompt."""
    q_text = question.get("question_text", "")
    evidence = question.get("gold_evidence", [])
    if isinstance(evidence, list):
        context = " ".join(str(e) for e in evidence)
    else:
        context = str(evidence)
    template = config["generation"]["prompt_template"]
    return template.format(context=context[:1000], question=q_text)  # truncate context


def main():
    parser = argparse.ArgumentParser(description="Stage C (SH2b): Compute QA-context steering vector")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "steering_vectors.npz"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config["data_dir"]) / "results"
    Path(config["data_dir"] + "/results").mkdir(parents=True, exist_ok=True)

    # 1. Load questions — use first qa_vector_n for construction, rest for eval
    print("Loading questions...")
    questions = load_jsonl(config["sh5_questions"])
    n_vector = config["steering"].get("qa_vector_n", 100)
    construction_questions = questions[:n_vector]
    print(f"Using {len(construction_questions)} questions for steering vector construction")
    print(f"Remaining {len(questions) - n_vector} questions for evaluation")

    # 2. Build contrastive prompt pairs
    macro_instruction = config["steering"].get(
        "macro_instruction",
        "Answer briefly in one or two sentences at a high level, without specific technical details, numbers, or citations."
    )
    micro_instruction = config["steering"].get(
        "micro_instruction",
        "Answer with full technical detail: include specific numbers, methods, experimental findings, and cite evidence where relevant."
    )

    macro_prompts = []
    micro_prompts = []
    for q in construction_questions:
        base = format_qa_prompt(q, config)
        macro_prompts.append(base + f"\n\nInstruction: {macro_instruction}")
        micro_prompts.append(base + f"\n\nInstruction: {micro_instruction}")

    print(f"Built {len(macro_prompts)} macro + {len(micro_prompts)} micro prompts")

    # 3. Load generative model
    model_name = config["generative_model"]["name"]
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # 4. Determine candidate layers
    n_layers = model.config.num_hidden_layers
    candidate_layers = [int(f * n_layers) for f in config["steering"]["layer_fractions"]]
    print(f"Model has {n_layers} layers. Candidate layers: {candidate_layers}")

    # 5. Extract hidden states for macro and micro prompts
    print("\nExtracting macro hidden states...")
    macro_states = extract_qa_hidden_states(model, tokenizer, macro_prompts, candidate_layers, device)

    print("\nExtracting micro hidden states...")
    micro_states = extract_qa_hidden_states(model, tokenizer, micro_prompts, candidate_layers, device)

    for l in candidate_layers:
        print(f"  Layer {l}: macro={macro_states[l].shape}, micro={micro_states[l].shape}")

    # 6. Compute per-layer steering vectors
    print("\nComputing steering vectors...")
    from src.steering import compute_steering_vectors
    sv_result = compute_steering_vectors(macro_states, micro_states)
    print(f"Steering vectors computed. Norms: {sv_result['norms']}")

    # 7. Load SLoD eval axis for layer validation
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    eval_axis = axis_data["centroid_axis"]

    # 8. Use remaining questions for layer/alpha selection
    val_questions = questions[n_vector:n_vector + config["steering"]["validation_n"]]
    val_prompts = [format_qa_prompt(q, config) for q in val_questions]
    print(f"\nUsing {len(val_prompts)} questions for layer validation")

    # 9. Select best layer (signed selection, fixed)
    print("\nSelecting best layer...")
    scibert_model = config["scibert_model"]
    scibert_batch = config["scibert_batch_size"]
    scibert_maxlen = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch, scibert_maxlen)

    from src.steering import select_best_layer
    layer_result = select_best_layer(
        model, tokenizer, sv_result["vectors"],
        val_prompts, eval_axis, embed_fn,
        candidate_layers, alpha=1.0
    )
    print(f"Selected layer: {layer_result['selected_layer']}")
    print(f"Layer shifts: {layer_result['layer_shifts']}")

    # 10. Save steering vectors and metadata
    vectors_array = np.zeros((n_layers, model.config.hidden_size))
    for l, v in sv_result["vectors"].items():
        vectors_array[l] = v

    np.savez(
        str(output_path),
        vectors=vectors_array,
        selected_layer=np.array(layer_result["selected_layer"]),
        candidate_layers=np.array(candidate_layers),
        layer_shifts=np.array([layer_result["layer_shifts"].get(l, 0.0) for l in candidate_layers]),
        model_name=np.array(model_name),
        n_macro_spans=np.array(len(macro_prompts)),
        n_micro_spans=np.array(len(micro_prompts)),
        method=np.array("qa_context_contrastive")
    )
    print(f"\nSaved steering vectors to {output_path}")

    # 11. Save layer selection results
    layer_selection = {
        "selected_layer": int(layer_result["selected_layer"]),
        "candidate_layers": candidate_layers,
        "layer_shifts": {str(l): float(layer_result["layer_shifts"].get(l, 0.0)) for l in candidate_layers},
        "method": "qa_context_contrastive",
        "n_construction_questions": n_vector,
        "macro_instruction": macro_instruction,
        "micro_instruction": micro_instruction,
        "justification": f"Layer {layer_result['selected_layer']} produced largest positive SLoD shift on validation set (signed selection)"
    }
    save_json(layer_selection, config["data_dir"] + "/results/layer_selection.json")
    print(f"Saved layer selection to data/results/layer_selection.json")
    print("\nStage C (SH2b) complete.")


if __name__ == "__main__":
    main()
