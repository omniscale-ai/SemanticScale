"""Activation steering: compute steering vectors and generate steered text.

This module implements the core SH2 functionality:
1. Extract per-layer hidden states from a generative model for labeled spans
2. Compute per-layer steering vectors (difference-of-means: micro - macro)
3. Apply steering via forward hooks during generation
4. Layer and alpha selection via validation

References:
- Zou et al. (2023) "Representation Engineering" — difference-of-means steering
- Turner et al. (2023) "Activation Addition" — residual stream injection

Implementation Notes:
- The Engineer agent should implement all functions below.
- Use `model.model.layers[L]` for Mistral/Llama to access transformer layers.
- Hidden states: use `output_hidden_states=True` or register forward hooks.
- For memory efficiency: process spans in batches, accumulate running sums for centroids.
- Use mean-pooling across token positions (or last-token for causal LMs).
"""
import numpy as np
import torch


def extract_hidden_states(model, tokenizer, texts: list, batch_size: int = 16,
                          layers: list = None) -> dict:
    """Extract hidden states from generative model for given texts.

    Args:
        model: HuggingFace causal LM (e.g., MistralForCausalLM)
        tokenizer: corresponding tokenizer
        texts: list of text strings
        batch_size: processing batch size
        layers: list of layer indices to extract (None = all layers)

    Returns:
        dict mapping layer_index -> np.ndarray of shape (n_texts, hidden_dim)
        Each entry is the mean-pooled hidden state across tokens for that layer.
    """
    if layers is None:
        n_layers = model.config.num_hidden_layers
        layers = list(range(n_layers))

    # Initialize accumulators: layer -> list of (n_texts_in_batch, hidden_dim)
    layer_results = {l: [] for l in layers}

    n_texts = len(texts)
    n_batches = (n_texts + batch_size - 1) // batch_size

    print(f"Extracting hidden states for {n_texts} texts in {n_batches} batches at layers {layers}")

    for batch_idx in range(0, n_texts, batch_size):
        batch_texts = texts[batch_idx: batch_idx + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Support device_map="auto" (multi-device): send to embedding layer's device
        first_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(first_device)
        attention_mask = inputs["attention_mask"].to(first_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim)
        # index 0 = embedding layer, index L+1 = output of transformer layer L
        hidden_states = outputs.hidden_states

        for layer_idx in layers:
            # Layer L corresponds to hidden_states[L+1]
            hs = hidden_states[layer_idx + 1]  # (batch, seq_len, hidden_dim)

            # Mean pool across seq_len, masking out padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            sum_hs = (hs * mask).sum(dim=1)              # (batch, hidden_dim)
            count = mask.sum(dim=1)                       # (batch, 1)
            mean_hs = sum_hs / count                      # (batch, hidden_dim)

            layer_results[layer_idx].append(mean_hs.cpu().float().numpy())

        # Free memory for this batch
        del outputs, hidden_states, input_ids, attention_mask
        torch.cuda.empty_cache()

    # Concatenate across batches
    result = {}
    for layer_idx in layers:
        result[layer_idx] = np.concatenate(layer_results[layer_idx], axis=0)

    return result


def compute_steering_vectors(macro_states: dict, micro_states: dict) -> dict:
    """Compute per-layer steering vectors from macro/micro hidden states.

    Args:
        macro_states: dict[layer_idx] -> (n_macro, hidden_dim) array
        micro_states: dict[layer_idx] -> (n_micro, hidden_dim) array

    Returns:
        dict with:
        - "vectors": dict[layer_idx] -> (hidden_dim,) unit vector (micro - macro direction)
        - "norms": dict[layer_idx] -> float (pre-normalization L2 norm)
    """
    vectors = {}
    norms = {}

    layers = list(macro_states.keys())
    for layer_idx in layers:
        macro_centroid = macro_states[layer_idx].mean(axis=0)
        micro_centroid = micro_states[layer_idx].mean(axis=0)
        raw_vector = micro_centroid - macro_centroid
        norm = float(np.linalg.norm(raw_vector))
        if norm < 1e-10:
            print(f"Warning: layer {layer_idx} has near-zero steering vector norm.")
            unit_vector = raw_vector
        else:
            unit_vector = raw_vector / norm
        vectors[layer_idx] = unit_vector.astype(np.float32)
        norms[layer_idx] = norm

    return {"vectors": vectors, "norms": norms}


def select_best_layer(model, tokenizer, steering_vectors: dict,
                      eval_texts: list, eval_axis: np.ndarray,
                      embed_fn, candidate_layers: list,
                      alpha: float = 1.0) -> dict:
    """Select the best layer for steering by measuring SLoD shift on validation set.

    Args:
        model: generative model
        tokenizer: tokenizer
        steering_vectors: dict[layer_idx] -> unit vector
        eval_texts: list of question prompts for validation
        eval_axis: (768,) SciBERT SLoD axis for measuring shift
        embed_fn: function to embed text with SciBERT -> (N, 768)
        candidate_layers: list of layer indices to try
        alpha: steering strength

    Returns:
        dict with:
        - "selected_layer": int
        - "layer_shifts": dict[layer_idx] -> mean SLoD shift
        - "layer_details": per-layer statistics
    """
    print(f"Selecting best layer from candidates: {candidate_layers}")

    # Generate baseline outputs once (no steering)
    print("Generating baseline outputs...")
    baseline_outputs = generate_with_steering(
        model, tokenizer, eval_texts,
        steering_vector=None, alpha=0.0,
        max_new_tokens=256, temperature=0.0
    )
    baseline_texts = [r["text"] for r in baseline_outputs]
    baseline_embs = embed_fn(baseline_texts)          # (N, 768)
    baseline_proj = baseline_embs @ eval_axis          # (N,)

    torch.cuda.empty_cache()

    layer_shifts = {}
    layer_details = {}

    for layer_idx in candidate_layers:
        if layer_idx not in steering_vectors:
            print(f"  Layer {layer_idx}: no steering vector, skipping.")
            layer_shifts[layer_idx] = 0.0
            continue

        sv = steering_vectors[layer_idx]
        print(f"  Testing layer {layer_idx}...")

        steered_outputs = generate_with_steering(
            model, tokenizer, eval_texts,
            steering_vector=sv, alpha=alpha,
            direction="micro", layer_idx=layer_idx,
            max_new_tokens=256, temperature=0.0
        )
        steered_texts = [r["text"] for r in steered_outputs]
        steered_embs = embed_fn(steered_texts)
        steered_proj = steered_embs @ eval_axis

        mean_shift = float((steered_proj - baseline_proj).mean())
        layer_shifts[layer_idx] = mean_shift
        layer_details[layer_idx] = {
            "mean_shift": mean_shift,
            "baseline_mean_proj": float(baseline_proj.mean()),
            "steered_mean_proj": float(steered_proj.mean()),
        }
        print(f"    Layer {layer_idx}: mean shift = {mean_shift:.4f}")

        torch.cuda.empty_cache()

    # Select layer with largest positive shift in micro direction
    selected_layer = max(candidate_layers, key=lambda l: layer_shifts.get(l, 0.0))
    print(f"Selected layer: {selected_layer} (shift={layer_shifts[selected_layer]:.4f})")

    return {
        "selected_layer": selected_layer,
        "layer_shifts": layer_shifts,
        "layer_details": layer_details,
    }


def create_steering_hook(steering_vector: np.ndarray, alpha: float = 1.0,
                         direction: str = "micro"):
    """Create a forward hook that injects the steering vector.

    Args:
        steering_vector: (hidden_dim,) unit vector (micro direction)
        alpha: steering strength multiplier
        direction: "micro" (add vector) or "macro" (subtract vector)

    Returns:
        hook function compatible with PyTorch register_forward_hook
    """
    sv_tensor = torch.tensor(steering_vector, dtype=torch.bfloat16)
    sign = 1.0 if direction == "micro" else -1.0

    def hook(module, input, output):
        # output may be a tensor or a tuple; handle both cases
        hidden_states = output[0] if isinstance(output, tuple) else output
        sv = sv_tensor.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states + sign * alpha * sv
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    return hook


def generate_with_steering(model, tokenizer, prompts: list,
                           steering_vector: np.ndarray = None,
                           alpha: float = 0.0, direction: str = "micro",
                           layer_idx: int = None,
                           max_new_tokens: int = 512,
                           temperature: float = 0.0) -> list:
    """Generate text with optional activation steering.

    Args:
        model: generative model
        tokenizer: tokenizer
        prompts: list of prompt strings
        steering_vector: (hidden_dim,) unit vector (None = no steering)
        alpha: steering strength (0.0 = no steering)
        direction: "micro" or "macro"
        layer_idx: which layer to steer
        max_new_tokens: max generation length
        temperature: sampling temperature (0.0 = greedy)

    Returns:
        list of dicts: [{"text": "...", "n_tokens": 128}, ...]
    """
    hook_handle = None

    if steering_vector is not None and alpha > 0.0 and layer_idx is not None:
        hook_fn = create_steering_hook(steering_vector, alpha=alpha, direction=direction)
        hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    results = []
    try:
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            # Support device_map="auto": send to embedding layer's device
            first_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(first_device)
            input_len = input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = model.generate(input_ids, **gen_kwargs)

            new_tokens = output_ids[0][input_len:]
            n_new_tokens = len(new_tokens)
            decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            results.append({"text": decoded_text, "n_tokens": n_new_tokens})

            del input_ids, output_ids
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return results
