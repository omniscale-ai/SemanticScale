"""Embedding generation for SH3: documents and queries with MiniLM or Specter2."""

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils import load_config, load_jsonl


def load_embedding_model(model_name: str):
    """Load a SentenceTransformer model.

    Returns:
        SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer

    logging.info(f"Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def embed_texts(
    model,
    texts: list[str],
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts using a SentenceTransformer model.

    Args:
        model: SentenceTransformer model.
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        np.ndarray of shape (len(texts), dim).
    """
    logging.info(f"Encoding {len(texts)} texts (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_texts_transformers(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Embed texts using a transformers AutoModel with [CLS] pooling (e.g., Specter2 base).

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


def get_retrieval_model_config(config: dict) -> dict:
    """Get the active retrieval model configuration.

    Returns:
        Dict with model config (name, type, dim, batch_size, etc.).
    """
    embedding_cfg = config["embedding"]
    retrieval_model = embedding_cfg.get("retrieval_model", "minilm")
    models = embedding_cfg.get("models", {})

    if retrieval_model in models:
        return models[retrieval_model]

    # Fallback to legacy config
    logging.warning(f"Retrieval model '{retrieval_model}' not found in models config, using legacy MiniLM config")
    return {
        "name": embedding_cfg.get("model_name", "all-MiniLM-L6-v2"),
        "type": "sentence-transformer",
        "dim": embedding_cfg.get("dimension", 384),
        "batch_size": embedding_cfg.get("batch_size", 256),
    }


def get_embedding_file_prefix(config: dict) -> str:
    """Get the file prefix for embedding files based on active retrieval model.

    Returns:
        Prefix string like 'specter2' or '' (empty for minilm/default).
    """
    retrieval_model = config["embedding"].get("retrieval_model", "minilm")
    if retrieval_model == "minilm":
        return ""  # Legacy: no prefix, files are macro_embeddings.npz etc.
    return f"{retrieval_model}_"


def embed_all_documents(config: dict, project_root: Path | None = None) -> None:
    """Embed all documents at all 3 levels and save to .npz files.

    Each .npz contains: embeddings, paper_ids, doc_ids, texts.
    Uses the legacy MiniLM config (embedding.model_name).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    emb_dir = data_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["embedding"]["model_name"]
    batch_size = config["embedding"]["batch_size"]
    normalize = config["embedding"]["normalize"]

    # Check for existing output
    level_files = {
        "macro": emb_dir / "macro_embeddings.npz",
        "meso": emb_dir / "meso_embeddings.npz",
        "micro": emb_dir / "micro_embeddings.npz",
    }

    all_exist = all(f.exists() for f in level_files.values())
    if all_exist:
        logging.info("All document embeddings already exist, skipping.")
        return

    # Load paper index
    papers_path = data_dir / "papers_index.jsonl"
    papers = load_jsonl(papers_path)

    # Collect documents by level
    docs_by_level = {"macro": [], "meso": [], "micro": []}
    for paper in papers:
        for level in ["macro", "meso", "micro"]:
            key = f"{level}_docs"
            for doc in paper.get(key, []):
                docs_by_level[level].append(doc)

    # Load model once
    model = load_embedding_model(model_name)

    for level, docs in docs_by_level.items():
        out_path = level_files[level]
        if out_path.exists():
            logging.info(f"{level} embeddings already exist, skipping.")
            continue

        if not docs:
            logging.warning(f"No {level} documents found, skipping.")
            continue

        texts = [d["text"] for d in docs]
        paper_ids = [d["doc_id"].split("__")[0] for d in docs]
        doc_ids = [d["doc_id"] for d in docs]

        logging.info(f"Embedding {len(texts)} {level} documents...")
        embeddings = embed_texts(model, texts, batch_size=batch_size, normalize=normalize)

        logging.info(f"Saving {level} embeddings to {out_path} (shape: {embeddings.shape})")
        np.savez(
            out_path,
            embeddings=embeddings,
            paper_ids=np.array(paper_ids, dtype=object),
            doc_ids=np.array(doc_ids, dtype=object),
            texts=np.array(texts, dtype=object),
        )

    logging.info("Document embedding complete.")


def embed_queries(config: dict, project_root: Path | None = None) -> None:
    """Embed all question texts with MiniLM and save to .npz."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    emb_dir = data_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    out_path = emb_dir / "query_embeddings.npz"
    if out_path.exists():
        logging.info("Query embeddings already exist, skipping.")
        return

    model_name = config["embedding"]["model_name"]
    batch_size = config["embedding"]["batch_size"]
    normalize = config["embedding"]["normalize"]

    # Load questions
    questions_path = data_dir / "questions.jsonl"
    questions = load_jsonl(questions_path)

    texts = [q["question_text"] for q in questions]
    question_ids = [q["question_id"] for q in questions]
    paper_ids = [q["paper_id"] for q in questions]

    # Load model and embed
    model = load_embedding_model(model_name)
    logging.info(f"Embedding {len(texts)} queries...")
    embeddings = embed_texts(model, texts, batch_size=batch_size, normalize=normalize)

    logging.info(f"Saving query embeddings to {out_path} (shape: {embeddings.shape})")
    np.savez(
        out_path,
        embeddings=embeddings,
        question_ids=np.array(question_ids, dtype=object),
        paper_ids=np.array(paper_ids, dtype=object),
        texts=np.array(texts, dtype=object),
    )

    logging.info("Query embedding complete.")


def embed_all_specter2(config: dict, project_root: Path | None = None) -> None:
    """Embed all documents and queries with Specter2 and save to .npz files.

    Output files: specter2_macro.npz, specter2_meso.npz, specter2_micro.npz, specter2_queries.npz
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    emb_dir = data_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Get Specter2 config
    specter2_cfg = config["embedding"]["models"]["specter2"]
    model_name = specter2_cfg["name"]
    batch_size = specter2_cfg.get("batch_size", 64)
    max_length = specter2_cfg.get("max_length", 512)

    # Define output files
    level_files = {
        "macro": emb_dir / "specter2_macro.npz",
        "meso": emb_dir / "specter2_meso.npz",
        "micro": emb_dir / "specter2_micro.npz",
    }
    query_file = emb_dir / "specter2_queries.npz"

    all_exist = all(f.exists() for f in level_files.values()) and query_file.exists()
    if all_exist:
        logging.info("All Specter2 embeddings already exist, skipping.")
        return

    # Load paper index
    papers_path = data_dir / "papers_index.jsonl"
    papers = load_jsonl(papers_path)

    # Collect documents by level
    docs_by_level = {"macro": [], "meso": [], "micro": []}
    for paper in papers:
        for level in ["macro", "meso", "micro"]:
            key = f"{level}_docs"
            for doc in paper.get(key, []):
                docs_by_level[level].append(doc)

    # Embed each level
    for level, docs in docs_by_level.items():
        out_path = level_files[level]
        if out_path.exists():
            logging.info(f"Specter2 {level} embeddings already exist, skipping.")
            continue

        if not docs:
            logging.warning(f"No {level} documents found, skipping.")
            continue

        texts = [d["text"] for d in docs]
        paper_ids = [d["doc_id"].split("__")[0] for d in docs]
        doc_ids = [d["doc_id"] for d in docs]

        logging.info(f"Embedding {len(texts)} {level} documents with Specter2...")
        embeddings = embed_texts_transformers(
            texts, model_name, batch_size=batch_size, max_length=max_length,
        )

        logging.info(f"Saving Specter2 {level} embeddings to {out_path} (shape: {embeddings.shape})")
        np.savez(
            out_path,
            embeddings=embeddings,
            paper_ids=np.array(paper_ids, dtype=object),
            doc_ids=np.array(doc_ids, dtype=object),
            texts=np.array(texts, dtype=object),
        )

    # Embed queries
    if not query_file.exists():
        questions_path = data_dir / "questions.jsonl"
        questions = load_jsonl(questions_path)

        texts = [q["question_text"] for q in questions]
        question_ids = [q["question_id"] for q in questions]
        paper_ids = [q["paper_id"] for q in questions]

        logging.info(f"Embedding {len(texts)} queries with Specter2...")
        embeddings = embed_texts_transformers(
            texts, model_name, batch_size=batch_size, max_length=max_length,
        )

        logging.info(f"Saving Specter2 query embeddings to {query_file} (shape: {embeddings.shape})")
        np.savez(
            query_file,
            embeddings=embeddings,
            question_ids=np.array(question_ids, dtype=object),
            paper_ids=np.array(paper_ids, dtype=object),
            texts=np.array(texts, dtype=object),
        )

    logging.info("Specter2 embedding complete.")
