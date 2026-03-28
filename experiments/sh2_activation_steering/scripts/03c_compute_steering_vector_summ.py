#!/usr/bin/env python3
"""Stage 3c: Compute steering vector for summarization task.

Uses QA-context contrastive approach (same as SH2b) adapted to summarization:
- For each construction paper, build macro_prompt and micro_prompt
- Extract last-token hidden states at candidate layers
- Compute per-layer steering vectors (micro - macro)
- Select best layer on validation split (signed, not abs)
- Check direction: if selected layer shift is negative, set direction_flip=True
- Save data/summarization/steering_vectors.npz

GPU required. Expected runtime: ~45 minutes on L40S.

Output: data/summarization/steering_vectors.npz
        data/results/layer_selection_summ.json
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


def extract_last_token_hidden_states(model, tokenizer, prompts, layers, device,
                                     max_length=1536):
    """Extract last-token hidden states from model for given prompts.

    For each prompt, do a forward pass and extract hidden states at each
    candidate layer at the last token position (model's summary state of
    the entire input after processing the full context).

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        prompts: list of prompt strings
        layers: list of layer indices to extract
        device: torch device
        max_length: max tokenization length

    Returns:
        dict mapping layer_index -> np.ndarray of shape (n_prompts, hidden_dim)
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
                max_length=max_length
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            # hidden_states is a tuple of (n_layers+1) tensors: (1, seq_len, hidden_dim)
            # index 0 = embedding layer, index L+1 = output of transformer layer L
            # Use last token position as the summary state
            for l in layers:
                hs = outputs.hidden_states[l + 1]  # (1, seq_len, hidden_dim)
                last_token = hs[0, -1, :].float().cpu().numpy()  # (hidden_dim,)
                hidden_by_layer[l].append(last_token)

            del outputs, inputs

    # Convert lists to arrays
    return {l: np.array(hidden_by_layer[l]) for l in layers}


def build_paper_prompt(paper, config, truncate_to_tokens=None, tokenizer=None):
    """Format a paper text into a summarization prompt."""
    summ_cfg = config["summarization"]
    template = summ_cfg["prompt_template"]
    text = paper["text"]

    # Truncate text to max_source_tokens if tokenizer is provided
    if truncate_to_tokens is not None and tokenizer is not None:
        tokens = tokenizer.encode(text, truncation=True, max_length=truncate_to_tokens)
        text = tokenizer.decode(tokens, skip_special_tokens=True)

    return template.format(text=text)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3c: Compute summarization steering vector"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    summ_cfg = config["summarization"]

    out_path = Path(summ_cfg["steering_vectors_path"])
    layer_sel_path = Path(config["data_dir"]) / "results" / "layer_selection_summ.json"

    if out_path.exists() and not args.force:
        print(f"Output exists: {out_path}. Use --force to rerun.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config["data_dir"] + "/results").mkdir(parents=True, exist_ok=True)

    # 1. Load papers + split
    papers_path = Path(summ_cfg["papers_path"])
    split_path = papers_path.parent / "split.json"

    print("Loading papers and split...")
    papers = load_jsonl(str(papers_path))

    import json
    with open(str(split_path)) as f:
        split = json.load(f)

    construction_idx = split["construction"]
    validation_idx = split["validation"]

    construction_papers = [papers[i] for i in construction_idx]
    validation_papers = [papers[i] for i in validation_idx]

    print(f"Construction: {len(construction_papers)} papers")
    print(f"Validation: {len(validation_papers)} papers")

    # 2. Load generative model
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

    max_source_tokens = summ_cfg["max_source_tokens"]
    macro_instruction = summ_cfg["macro_instruction"]
    micro_instruction = summ_cfg["micro_instruction"]

    # 3. Determine candidate layers
    n_layers = model.config.num_hidden_layers
    candidate_layers = [int(f * n_layers) for f in config["steering"]["layer_fractions"]]
    print(f"Model has {n_layers} layers. Candidate layers: {candidate_layers}")

    # 4. Build contrastive prompt pairs for construction set
    print("\nBuilding contrastive prompts for construction set...")
    macro_prompts = []
    micro_prompts = []

    for paper in construction_papers:
        base = build_paper_prompt(paper, config, max_source_tokens, tokenizer)
        macro_prompts.append(base + f"\n\nInstruction: {macro_instruction}")
        micro_prompts.append(base + f"\n\nInstruction: {micro_instruction}")

    print(f"Built {len(macro_prompts)} macro + {len(micro_prompts)} micro prompts")

    # 5. Extract hidden states (last-token)
    # Use max_length = max_source_tokens + some headroom for the instruction text
    extract_max_length = max_source_tokens + 256

    print("\nExtracting macro hidden states...")
    macro_states = extract_last_token_hidden_states(
        model, tokenizer, macro_prompts, candidate_layers, device,
        max_length=extract_max_length
    )

    print("\nExtracting micro hidden states...")
    micro_states = extract_last_token_hidden_states(
        model, tokenizer, micro_prompts, candidate_layers, device,
        max_length=extract_max_length
    )

    for l in candidate_layers:
        print(f"  Layer {l}: macro={macro_states[l].shape}, micro={micro_states[l].shape}")

    # 6. Compute per-layer steering vectors
    print("\nComputing steering vectors...")
    sv_result = compute_steering_vectors(macro_states, micro_states)
    print(f"Steering vectors computed. Norms: {sv_result['norms']}")

    # 7. Load SLoD eval axis
    axis_data = np.load(config["data_dir"] + "/eval_slod_axis.npz")
    eval_axis = axis_data["centroid_axis"]

    # 8. Build validation prompts (no instruction — just base prompt for generation)
    print(f"\nBuilding {len(validation_papers)} validation prompts...")
    val_prompts = []
    for paper in validation_papers:
        val_prompts.append(build_paper_prompt(paper, config, max_source_tokens, tokenizer))

    # 9. Select best layer (signed selection)
    print("\nSelecting best layer...")
    scibert_model = config["scibert_model"]
    scibert_batch = config["scibert_batch_size"]
    scibert_maxlen = config["scibert_max_length"]

    def embed_fn(texts):
        return embed_texts(texts, scibert_model, scibert_batch, scibert_maxlen)

    layer_result = select_best_layer(
        model, tokenizer, sv_result["vectors"],
        val_prompts, eval_axis, embed_fn,
        candidate_layers, alpha=1.0
    )
    selected_layer = layer_result["selected_layer"]
    selected_shift = layer_result["layer_shifts"][selected_layer]

    print(f"Selected layer: {selected_layer} (shift={selected_shift:.4f})")
    print(f"Layer shifts: {layer_result['layer_shifts']}")

    # 10. Check direction: if selected shift is NEGATIVE, set direction_flip=True
    direction_flip = bool(selected_shift < 0.0)
    print(f"Direction flip: {direction_flip} (shift={'negative' if direction_flip else 'positive'})")

    # 11. Save steering vectors
    vectors_array = np.zeros((n_layers, model.config.hidden_size), dtype=np.float32)
    for l, v in sv_result["vectors"].items():
        vectors_array[l] = v

    np.savez(
        str(out_path),
        vectors=vectors_array,
        selected_layer=np.array(selected_layer),
        candidate_layers=np.array(candidate_layers),
        layer_shifts=np.array([layer_result["layer_shifts"].get(l, 0.0) for l in candidate_layers]),
        direction_flip=np.array(direction_flip),
        model_name=np.array(model_name),
        n_construction=np.array(len(construction_papers)),
        method=np.array("summarization_contrastive_last_token")
    )
    print(f"\nSaved steering vectors to {out_path}")

    # 12. Save layer selection results
    layer_selection = {
        "selected_layer": int(selected_layer),
        "candidate_layers": candidate_layers,
        "layer_shifts": {str(l): float(layer_result["layer_shifts"].get(l, 0.0)) for l in candidate_layers},
        "direction_flip": direction_flip,
        "method": "summarization_contrastive_last_token",
        "n_construction_papers": len(construction_papers),
        "macro_instruction": macro_instruction,
        "micro_instruction": micro_instruction,
        "justification": (
            f"Layer {selected_layer} produced largest positive SLoD shift on validation set "
            f"(signed selection). Shift={selected_shift:.4f}. "
            f"direction_flip={direction_flip} because shift was {'negative' if direction_flip else 'positive'}."
        )
    }
    save_json(layer_selection, str(layer_sel_path))
    print(f"Saved layer selection to {layer_sel_path}")
    print("\nStage 3c complete.")


if __name__ == "__main__":
    main()
