#!/usr/bin/env python3
"""Stage A: Environment setup and validation.

Verifies GPU, downloads models, validates all data files.
Output: data/environment_check.json
"""
import sys
import json
import argparse
from pathlib import Path

# Add parent dir to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, save_json, load_jsonl

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Stage A: Environment setup")
    parser.add_argument("--force", action="store_true", help="Force rerun")
    args = parser.parse_args()

    config = load_config()
    output_path = Path(config["data_dir"]) / "environment_check.json"

    if output_path.exists() and not args.force:
        print(f"Output exists: {output_path}. Use --force to rerun.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Check GPU
    import torch
    results["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        results["cuda_version"] = torch.version.cuda
    else:
        print("WARNING: No GPU detected. Steering will be very slow on CPU.")
    results["torch_version"] = torch.__version__
    results["python_version"] = sys.version

    # 2. Download models
    print("Downloading generative model (this may take a while)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = config["generative_model"]["name"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Just verify it's downloadable; don't load full model yet
        print(f"  Generative model tokenizer OK: {model_name}")
        results["generative_model"] = model_name
        results["generative_model_ok"] = True
    except Exception as e:
        print(f"  ERROR downloading {model_name}: {e}")
        results["generative_model_ok"] = False

    print("Downloading SciBERT...")
    from transformers import AutoModel
    scibert_name = config["scibert_model"]
    try:
        AutoTokenizer.from_pretrained(scibert_name)
        # Use safetensors to avoid torch.load vulnerability (CVE-2025-32434)
        try:
            AutoModel.from_pretrained(scibert_name, use_safetensors=True)
        except Exception:
            AutoModel.from_pretrained(scibert_name, weights_only=False)
        print(f"  SciBERT OK: {scibert_name}")
        results["scibert_ok"] = True
    except Exception as e:
        print(f"  ERROR downloading {scibert_name}: {e}")
        results["scibert_ok"] = False

    # 3. Validate data files
    print("Validating data files...")
    data_checks = {}

    # SH0 spans
    sh0_path = config["sh0_spans"]
    if Path(sh0_path).exists():
        spans = load_jsonl(sh0_path)
        data_checks["sh0_spans"] = {"exists": True, "n_records": len(spans)}
        print(f"  SH0 spans: {len(spans)} records")
    else:
        data_checks["sh0_spans"] = {"exists": False}
        print(f"  ERROR: {sh0_path} not found")

    # SH1 embeddings
    sh1_emb_path = config["sh1_embeddings"]
    if Path(sh1_emb_path).exists():
        d = np.load(sh1_emb_path)
        data_checks["sh1_embeddings"] = {
            "exists": True,
            "shape": list(d["embeddings"].shape),
            "labels_shape": list(d["labels"].shape),
        }
        print(f"  SH1 embeddings: {d['embeddings'].shape}")
    else:
        data_checks["sh1_embeddings"] = {"exists": False}
        print(f"  ERROR: {sh1_emb_path} not found")

    # SH1 splits
    sh1_splits_path = config["sh1_splits"]
    if Path(sh1_splits_path).exists():
        with open(sh1_splits_path) as f:
            splits = json.load(f)
        data_checks["sh1_splits"] = {
            "exists": True,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }
        print(f"  SH1 splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    else:
        data_checks["sh1_splits"] = {"exists": False}

    # SH5 questions
    sh5_q_path = config["sh5_questions"]
    if Path(sh5_q_path).exists():
        questions = load_jsonl(sh5_q_path)
        data_checks["sh5_questions"] = {"exists": True, "n_records": len(questions)}
        print(f"  SH5 questions: {len(questions)} records")
    else:
        data_checks["sh5_questions"] = {"exists": False}

    # SH5 answer scores
    sh5_scores_path = config["sh5_answer_scores"]
    if Path(sh5_scores_path).exists():
        scores = load_jsonl(sh5_scores_path)
        data_checks["sh5_answer_scores"] = {"exists": True, "n_records": len(scores)}
        print(f"  SH5 answer scores: {len(scores)} records")
    else:
        data_checks["sh5_answer_scores"] = {"exists": False}

    results["data_checks"] = data_checks

    # 4. Overall status
    all_ok = (
        results.get("cuda_available", False)
        and results.get("generative_model_ok", False)
        and results.get("scibert_ok", False)
        and all(v.get("exists", False) for v in data_checks.values())
    )
    results["all_ok"] = all_ok
    print(f"\nOverall status: {'ALL OK' if all_ok else 'ISSUES DETECTED'}")

    save_json(results, str(output_path))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
