"""Embedding generation for SH1: MiniLM, SciBERT, Specter2."""

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils import LABEL_MAP, load_config, load_spans


def embed_with_sentence_transformer(
    texts: list[str],
    model_name: str,
    batch_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """Embed texts using a SentenceTransformer model (e.g., MiniLM).

    Returns:
        np.ndarray of shape (len(texts), dim).
    """
    from sentence_transformers import SentenceTransformer

    logging.info(f"Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)

    logging.info(f"Encoding {len(texts)} texts (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_with_transformers(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Embed texts using a transformers AutoModel with [CLS] pooling (e.g., SciBERT).

    Sorts texts by word count before batching to minimize padding waste,
    then restores original order.

    Returns:
        np.ndarray of shape (len(texts), hidden_dim).
    """
    from transformers import AutoModel, AutoTokenizer

    logging.info(f"Loading transformers model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Sort by approximate length (word count) to minimize padding waste
    sort_idx = np.argsort([len(t.split()) for t in texts])
    sorted_texts = [texts[i] for i in sort_idx]

    all_embeddings = []
    for i in tqdm(range(0, len(sorted_texts), batch_size), desc=f"Embedding ({model_name})"):
        batch_texts = sorted_texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        # Extract [CLS] token embedding (first token of last hidden state)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    sorted_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Restore original order
    unsort_idx = np.argsort(sort_idx)
    return sorted_embeddings[unsort_idx]


def embed_with_adapter(
    texts: list[str],
    base_model: str,
    adapter_name: str,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Embed texts using a transformers model with an adapter (e.g., Specter2).

    Tries to load the adapter; if it fails, falls back to the base model with [CLS] pooling.

    Returns:
        np.ndarray of shape (len(texts), hidden_dim).
    """
    from transformers import AutoTokenizer

    logging.info(f"Loading base model: {base_model} with adapter: {adapter_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = None
    # Try loading with adapters library
    try:
        from adapters import AutoAdapterModel

        logging.info("Trying adapters library for adapter loading...")
        model = AutoAdapterModel.from_pretrained(base_model)
        model.load_adapter(adapter_name, source="hf", set_active=True)
        logging.info(f"Adapter '{adapter_name}' loaded successfully via adapters library.")
    except Exception as e:
        logging.warning(f"Adapter loading via adapters library failed: {e}")
        model = None

    # Fallback: try model with PEFT
    if model is None:
        try:
            from transformers import AutoModel

            logging.info("Trying AutoModel with PEFT adapter...")
            model = AutoModel.from_pretrained(base_model)
            model.load_adapter(adapter_name)
            logging.info(f"Adapter '{adapter_name}' loaded via AutoModel.load_adapter().")
        except Exception as e:
            logging.warning(f"PEFT adapter loading failed: {e}")
            model = None

    # Final fallback: base model only
    if model is None:
        from transformers import AutoModel

        logging.warning(f"All adapter methods failed. Using base model only: {base_model}")
        model = AutoModel.from_pretrained(base_model)

    model.eval()

    # Sort by approximate length (word count) to minimize padding waste
    sort_idx = np.argsort([len(t.split()) for t in texts])
    sorted_texts = [texts[i] for i in sort_idx]

    all_embeddings = []
    for i in tqdm(range(0, len(sorted_texts), batch_size), desc=f"Embedding ({base_model}+adapter)"):
        batch_texts = sorted_texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    sorted_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Restore original order
    unsort_idx = np.argsort(sort_idx)
    return sorted_embeddings[unsort_idx]


def embed_dataset(
    config: dict,
    model_key: str,
    dataset_key: str = "primary",
    project_root: Path | None = None,
) -> Path:
    """Orchestrate embedding: load spans, embed with the specified model, save .npz.

    Args:
        config: Loaded config dict.
        model_key: Key in config['models'] (e.g., 'minilm', 'scibert', 'specter2').
        dataset_key: 'primary' for length-matched, 'full' for the full dataset.
        project_root: Project root directory (for resolving paths).

    Returns:
        Path to the saved .npz file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    model_cfg = config["models"][model_key]

    # Determine dataset path
    if dataset_key == "primary":
        data_path = project_root / config["sh0_data_dir"] / config["primary_dataset"]
        suffix = "length_matched"
    elif dataset_key == "full":
        data_path = project_root / config["sh0_data_dir"] / config["full_dataset"]
        suffix = "full"
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    # Output path
    embeddings_dir = project_root / config["data_dir"] / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_path = embeddings_dir / f"{model_key}_{suffix}.npz"

    # Check if already computed
    if output_path.exists():
        logging.info(f"Embeddings already exist at {output_path}, skipping.")
        return output_path

    # Load spans
    logging.info(f"Loading spans from {data_path}")
    spans = load_spans(data_path)
    texts = [s["text"] for s in spans]
    labels = np.array([LABEL_MAP[s["label"]] for s in spans], dtype=np.int32)
    span_ids = np.array([s["span_id"] for s in spans])

    # Embed based on model type
    model_type = model_cfg["type"]
    if model_type == "sentence-transformer":
        embeddings = embed_with_sentence_transformer(
            texts,
            model_name=model_cfg["name"],
            batch_size=model_cfg.get("batch_size", 128),
            normalize=model_cfg.get("normalize", True),
        )
    elif model_type == "transformers":
        embeddings = embed_with_transformers(
            texts,
            model_name=model_cfg["name"],
            batch_size=model_cfg.get("batch_size", 64),
            max_length=model_cfg.get("max_length", 512),
        )
    elif model_type == "transformers-adapter":
        embeddings = embed_with_adapter(
            texts,
            base_model=model_cfg["name"],
            adapter_name=model_cfg["adapter"],
            batch_size=model_cfg.get("batch_size", 64),
            max_length=model_cfg.get("max_length", 512),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save
    logging.info(f"Saving embeddings to {output_path} (shape: {embeddings.shape})")
    np.savez(output_path, embeddings=embeddings, labels=labels, span_ids=span_ids)

    return output_path
