"""SciBERT embedding of CoT reasoning steps."""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def embed_texts(texts: list, model_name: str = "allenai/scibert_scivocab_uncased",
                batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """Embed texts using SciBERT [CLS] token.

    Args:
        texts: list of strings to embed
        model_name: HuggingFace model name
        batch_size: batch size for inference
        max_length: max token length

    Returns:
        embeddings: (N, 768) numpy array of [CLS] embeddings
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"Embedding {len(texts)} texts in {n_batches} batches (batch_size={batch_size})")
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract [CLS] token embedding (index 0)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings
