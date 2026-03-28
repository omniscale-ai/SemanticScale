"""Stage D: Feature engineering — SLoD drift + surface features."""

import logging
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils import (
    get_project_root,
    load_config,
    read_json,
    read_jsonl,
    write_json,
)


# --- SH1 Probe Wrapper ---

class SLoDProbe:
    """Wrapper around the SH1 SciBERT + LogReg probe.

    Retrains from saved embeddings and provides classification of new text.
    """

    LABEL_MAP = {"macro": 0, "meso": 1, "micro": 2}
    LABEL_NAMES = ["macro", "meso", "micro"]

    def __init__(self, config: dict, project_root: Path = None):
        if project_root is None:
            project_root = get_project_root()

        self.project_root = project_root
        self.probe_cfg = config["features"]["slod_probe"]
        self.sh1_data_dir = project_root / config["sh1_data_dir"]

        self.scaler = None
        self.clf = None
        self.tokenizer = None
        self.model = None

        self._train_probe()

    def _train_probe(self):
        """Retrain the LogReg probe from SH1 embeddings."""
        logging.info("Training SLoD probe from SH1 data...")

        # Load SH1 embeddings
        emb_path = self.sh1_data_dir / "embeddings" / "scibert_length_matched.npz"
        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]
        logging.info(f"Loaded SH1 embeddings: {embeddings.shape}")

        # Load SH1 splits
        splits = read_json(self.sh1_data_dir / "splits.json")
        train_idx = np.array(splits["train"])
        val_idx = np.array(splits["val"])

        X_train = embeddings[train_idx]
        y_train = labels[train_idx]

        # Scale
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)

        # Train LogReg
        C = self.probe_cfg.get("logreg_C", 1.0)
        max_iter = self.probe_cfg.get("logreg_max_iter", 2000)
        solver = self.probe_cfg.get("logreg_solver", "lbfgs")

        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42,
        )
        self.clf.fit(X_train_s, y_train)

        # Quick val check
        X_val = embeddings[val_idx]
        y_val = labels[val_idx]
        X_val_s = self.scaler.transform(X_val)
        val_acc = self.clf.score(X_val_s, y_val)
        logging.info(f"SLoD probe trained: val accuracy = {val_acc:.4f}")

    def _load_transformer(self):
        """Lazy-load the SciBERT model and tokenizer."""
        if self.tokenizer is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        model_name = self.probe_cfg["model_name"]
        logging.info(f"Loading SciBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts using SciBERT [CLS] token.

        Returns array of shape (len(texts), hidden_dim).
        """
        self._load_transformer()

        batch_size = self.probe_cfg.get("batch_size", 32)
        max_length = self.probe_cfg.get("max_length", 512)

        # Sort by length for efficient batching
        sort_idx = np.argsort([len(t.split()) for t in texts])
        sorted_texts = [texts[i] for i in sort_idx]

        all_embeddings = []
        for i in tqdm(range(0, len(sorted_texts), batch_size),
                      desc="Embedding spans"):
            batch = sorted_texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

        sorted_emb = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        # Restore original order
        unsort_idx = np.argsort(sort_idx)
        return sorted_emb[unsort_idx]

    def classify(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Classify texts into SLoD levels.

        Returns:
            labels: array of predicted labels (0=macro, 1=meso, 2=micro)
            probabilities: array of shape (n, 3) with class probabilities
        """
        embeddings = self.embed_texts(texts)
        X_scaled = self.scaler.transform(embeddings)
        labels = self.clf.predict(X_scaled)
        probabilities = self.clf.predict_proba(X_scaled)
        return labels, probabilities


# --- Surface Features ---

def compute_surface_features(
    source_span: str, surface_cfg: dict
) -> dict:
    """Compute surface features for a single source_span."""
    if not source_span:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "entity_density": 0.0,
            "citation_density": 0.0,
            "temporal_density": 0.0,
            "numeric_density": 0.0,
            "has_table_context": 0,
        }

    words = source_span.split()
    n_words = len(words)
    if n_words == 0:
        n_words = 1  # avoid division by zero

    # Sentence count (approximate: split by '.', '!', '?')
    sentences = re.split(r"[.!?]+", source_span)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    # Entity density: capitalized multi-word patterns
    entity_min_words = surface_cfg.get("entity_min_words", 2)
    # Match sequences of capitalized words (2+ words)
    entity_pattern = r"\b(?:[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){" + str(entity_min_words - 1) + r",})\b"
    entities = re.findall(entity_pattern, source_span)
    entity_density = len(entities) / n_words

    # Citation density
    citation_pattern = surface_cfg.get("citation_pattern", r"(\[\d+\]|\([^)]*\d{4}[^)]*\))")
    citations = re.findall(citation_pattern, source_span)
    citation_density = len(citations) / n_words

    # Temporal density
    temporal_pattern = surface_cfg.get("temporal_pattern", r"\b(19|20)\d{2}\b")
    temporals = re.findall(temporal_pattern, source_span)
    temporal_density = len(temporals) / n_words

    # Numeric density
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", source_span)
    numeric_density = len(numbers) / n_words

    # Table context
    table_keywords = ["table", "tab.", "results in table", "shown in table"]
    has_table = int(any(kw in source_span.lower() for kw in table_keywords))

    return {
        "word_count": len(words),
        "sentence_count": n_sentences,
        "entity_density": entity_density,
        "citation_density": citation_density,
        "temporal_density": temporal_density,
        "numeric_density": numeric_density,
        "has_table_context": has_table,
    }


# --- Drift Features ---

def compute_drift_features(
    realized_label: int,
    probabilities: np.ndarray,
    expected_slod_cfg: dict,
) -> dict:
    """Compute drift features for a single extraction.

    The source_span is shared across all fields (method, dataset, metric, value, year).
    We compute drift as |expected_SLoD - realized_SLoD| for each field.
    """
    expected = {
        "method": expected_slod_cfg["method"],
        "dataset": expected_slod_cfg["dataset"],
        "metric": expected_slod_cfg["metric"],
        "value": expected_slod_cfg["value"],
        "year": expected_slod_cfg["year"],
    }

    # Drift per field
    drifts = {}
    for field, exp_slod in expected.items():
        drifts[field] = abs(exp_slod - realized_label)

    drift_values = list(drifts.values())

    # Entropy of probe probability distribution
    prob_entropy = float(scipy_entropy(probabilities + 1e-10))

    return {
        "drift_mean": float(np.mean(drift_values)),
        "drift_max": float(np.max(drift_values)),
        "drift_value": float(drifts["value"]),
        "drift_method": float(drifts["method"]),
        "realized_slod_raw": int(realized_label),
        "realized_slod_prob_macro": float(probabilities[0]),
        "realized_slod_prob_meso": float(probabilities[1]),
        "realized_slod_prob_micro": float(probabilities[2]),
        "slod_entropy": prob_entropy,
    }


# --- Feature Matrix Assembly ---

# All feature names in canonical order
DRIFT_FEATURES = [
    "drift_mean", "drift_max", "drift_value", "drift_method",
    "realized_slod_raw",
    "realized_slod_prob_macro", "realized_slod_prob_meso", "realized_slod_prob_micro",
    "slod_entropy",
]
SURFACE_FEATURES = [
    "word_count", "sentence_count",
    "entity_density", "citation_density", "temporal_density", "numeric_density",
    "has_table_context",
]
CONFIDENCE_FEATURES = ["llm_confidence"]
ALL_FEATURES = DRIFT_FEATURES + SURFACE_FEATURES + CONFIDENCE_FEATURES


def run_feature_engineering(config: dict = None, project_root: Path = None) -> Path:
    """Run the full feature engineering pipeline.

    Returns path to the feature matrix .npz file.
    """
    if config is None:
        config = load_config()
    if project_root is None:
        project_root = get_project_root()

    feat_cfg = config["features"]
    data_dir = project_root / config["data_dir"]

    # Output path
    output_path = data_dir / feat_cfg["output"]
    if output_path.exists():
        logging.info(f"Feature matrix already exists at {output_path}, skipping.")
        return output_path

    # Load labeled extractions
    labeled_path = data_dir / config["labeling"]["output"]
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled extractions not found at {labeled_path}. Run Stage C first.")
    extractions = read_jsonl(labeled_path)
    logging.info(f"Loaded {len(extractions)} labeled extractions")

    # Filter out extractions with empty source_span
    valid_extractions = [e for e in extractions if e.get("source_span", "").strip()]
    logging.info(f"Valid extractions (non-empty source_span): {len(valid_extractions)}")

    if not valid_extractions:
        raise ValueError("No valid extractions with source_spans found")

    # Initialize SLoD probe
    probe = SLoDProbe(config, project_root)

    # Embed and classify all source_spans
    source_spans = [e["source_span"] for e in valid_extractions]
    logging.info(f"Classifying {len(source_spans)} source spans with SLoD probe...")
    labels, probabilities = probe.classify(source_spans)

    # Compute features for each extraction
    feature_rows = []
    extraction_ids = []
    paper_ids = []
    correctness_labels = []

    expected_slod_cfg = feat_cfg["expected_slod"]
    surface_cfg = feat_cfg.get("surface", {})

    for i, ext in enumerate(valid_extractions):
        # Drift features
        drift_feats = compute_drift_features(
            realized_label=int(labels[i]),
            probabilities=probabilities[i],
            expected_slod_cfg=expected_slod_cfg,
        )

        # Surface features
        surface_feats = compute_surface_features(
            ext["source_span"], surface_cfg
        )

        # Confidence feature
        confidence = ext.get("confidence", 0.5)

        # Assemble feature vector in canonical order
        row = []
        for fname in DRIFT_FEATURES:
            row.append(drift_feats[fname])
        for fname in SURFACE_FEATURES:
            row.append(surface_feats[fname])
        row.append(float(confidence))

        feature_rows.append(row)
        extraction_ids.append(ext.get("extraction_id", f"ext_{i:04d}"))
        paper_ids.append(ext.get("paper_id", ""))
        correctness_labels.append(ext.get("correct", 0))

    # Build matrices
    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(correctness_labels, dtype=np.int32)
    extraction_ids = np.array(extraction_ids)
    paper_ids = np.array(paper_ids)

    # Handle NaN/inf values — replace with 0.0
    n_nan = np.sum(np.isnan(X))
    n_inf = np.sum(np.isinf(X))
    if n_nan > 0 or n_inf > 0:
        logging.warning(f"Feature matrix has {n_nan} NaN and {n_inf} inf values — replacing with 0.0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Label distribution: correct={np.sum(y)}, incorrect={np.sum(1 - y)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X=X,
        y=y,
        feature_names=np.array(ALL_FEATURES),
        extraction_ids=extraction_ids,
        paper_ids=paper_ids,
    )
    logging.info(f"Feature matrix saved to {output_path}")

    # Also save feature names mapping
    feature_info = {
        "feature_names": ALL_FEATURES,
        "drift_features": DRIFT_FEATURES,
        "surface_features": SURFACE_FEATURES,
        "confidence_features": CONFIDENCE_FEATURES,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_correct": int(np.sum(y)),
        "n_incorrect": int(np.sum(1 - y)),
    }
    write_json(feature_info, output_path.parent / "feature_info.json")

    return output_path
