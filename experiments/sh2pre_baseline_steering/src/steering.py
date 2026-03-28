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
    # TODO: Engineer implements this
    # Key steps:
    # 1. Tokenize texts in batches (padding=True, truncation=True, max_length=512)
    # 2. Forward pass with output_hidden_states=True
    # 3. For each layer, extract hidden_states[layer] -> (batch, seq_len, hidden_dim)
    # 4. Mean-pool across seq_len dimension (or use last non-padding token)
    # 5. Accumulate across batches
    raise NotImplementedError("Engineer agent should implement this")


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
    # TODO: Engineer implements this
    # Key steps:
    # 1. For each layer: macro_centroid = mean(macro_states[layer])
    # 2. micro_centroid = mean(micro_states[layer])
    # 3. raw_vector = micro_centroid - macro_centroid
    # 4. norm = ||raw_vector||
    # 5. steering_vector = raw_vector / norm
    raise NotImplementedError("Engineer agent should implement this")


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
    # TODO: Engineer implements this
    # Key steps:
    # 1. For each candidate layer:
    #    a. Generate steered outputs (micro direction) for eval_texts
    #    b. Generate baseline outputs for eval_texts
    #    c. Embed both with SciBERT
    #    d. Project onto eval_axis
    #    e. Compute mean shift = mean(steered_proj - baseline_proj)
    # 2. Select layer with largest absolute shift
    raise NotImplementedError("Engineer agent should implement this")


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
    # TODO: Engineer implements this
    # Key pattern:
    #
    # import torch
    # sv_tensor = torch.tensor(steering_vector, dtype=torch.float16).cuda()
    # sign = 1.0 if direction == "micro" else -1.0
    #
    # def hook(module, input, output):
    #     hidden_states = output[0]  # (batch, seq_len, hidden_dim)
    #     hidden_states = hidden_states + sign * alpha * sv_tensor
    #     return (hidden_states,) + output[1:]
    #
    # return hook
    raise NotImplementedError("Engineer agent should implement this")


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
    # TODO: Engineer implements this
    # Key steps:
    # 1. If steering_vector is not None and alpha > 0:
    #    a. Create hook with create_steering_hook()
    #    b. Register on model.model.layers[layer_idx]
    # 2. For each prompt:
    #    a. Tokenize
    #    b. model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    #    c. Decode output tokens
    #    d. Store text and token count
    # 3. Remove hook if registered
    raise NotImplementedError("Engineer agent should implement this")
