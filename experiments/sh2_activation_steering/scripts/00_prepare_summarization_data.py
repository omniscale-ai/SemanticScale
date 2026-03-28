#!/usr/bin/env python3
"""Stage 0: Prepare summarization dataset from SH0 spans.

Reconstructs paper texts by concatenating spans in section/paragraph order.
Filters to papers suitable for summarization (sufficient length, both macro+micro).

Output: data/summarization/papers.jsonl
        data/summarization/split.json  (construction/validation/evaluation indices)
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, load_jsonl, save_jsonl, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    out_dir = Path("data/summarization")
    out_path = out_dir / "papers.jsonl"
    split_path = out_dir / "split.json"

    if out_path.exists() and not args.force:
        print(f"Output exists: {out_path}. Use --force to rerun.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    summ_cfg = config["summarization"]
    min_spans = summ_cfg["min_spans"]
    min_words = summ_cfg["min_words"]
    max_words = summ_cfg["max_words"]

    # Load spans
    print("Loading SH0 spans...")
    spans = load_jsonl(config["sh0_spans"])
    print(f"Loaded {len(spans)} spans")

    # Group by paper
    papers = defaultdict(list)
    for s in spans:
        papers[s["paper_id"]].append(s)

    # Reconstruct and filter
    print("Reconstructing paper texts...")
    records = []
    for paper_id, pspans in papers.items():
        # Sort by section then paragraph
        pspans.sort(key=lambda s: (s["section_index"], s["paragraph_index"]))

        has_macro = any(s["label"] == "macro" for s in pspans)
        has_micro = any(s["label"] == "micro" for s in pspans)
        total_words = sum(s["word_count"] for s in pspans)
        n_spans = len(pspans)

        if not (has_macro and has_micro):
            continue
        if n_spans < min_spans:
            continue
        if not (min_words <= total_words <= max_words):
            continue

        # Build text with section headers
        text_parts = []
        current_section = None
        for s in pspans:
            if s["section_name"] != current_section:
                current_section = s["section_name"]
                if current_section and current_section not in ("Title",):
                    text_parts.append(f"\n{current_section}\n")
            text_parts.append(s["text"])
        text = " ".join(text_parts).strip()

        # Build micro reference (concatenate micro spans)
        micro_ref = " ".join(s["text"] for s in pspans if s["label"] == "micro")
        macro_ref = " ".join(s["text"] for s in pspans if s["label"] == "macro")

        n_macro = sum(1 for s in pspans if s["label"] == "macro")
        n_micro = sum(1 for s in pspans if s["label"] == "micro")

        records.append({
            "paper_id": paper_id,
            "text": text,
            "micro_reference": micro_ref,
            "macro_reference": macro_ref,
            "n_spans": n_spans,
            "n_macro": n_macro,
            "n_micro": n_micro,
            "word_count": total_words,
        })

    print(f"Papers passing filters: {len(records)}")

    # Sort by paper_id for reproducibility
    records.sort(key=lambda r: r["paper_id"])

    # Split
    n_construction = summ_cfg["n_construction"]
    n_validation = summ_cfg["n_validation"]
    construction = records[:n_construction]
    validation = records[n_construction:n_construction + n_validation]
    evaluation = records[n_construction + n_validation:]

    print(f"Split: {len(construction)} construction, {len(validation)} validation, {len(evaluation)} evaluation")

    save_jsonl(records, str(out_path))
    save_json({
        "construction": list(range(n_construction)),
        "validation": list(range(n_construction, n_construction + n_validation)),
        "evaluation": list(range(n_construction + n_validation, len(records))),
        "total": len(records),
    }, str(split_path))

    print(f"Saved {len(records)} papers to {out_path}")
    print(f"Saved split to {split_path}")


if __name__ == "__main__":
    main()
