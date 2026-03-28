"""Step 1: Load QASPER dataset from HuggingFace and flatten into raw JSONL.

Each record in the output contains:
  paper_id, title, abstract, section_name, paragraphs (list[str]),
  section_index (int)

Usage:
    python src/load_qasper.py [--output data/qasper_raw.jsonl]
"""

import argparse
from datasets import load_dataset
from tqdm import tqdm

from src.utils import load_config, resolve_path, write_jsonl, setup_logging

logger = setup_logging("load_qasper")


def flatten_paper(paper: dict) -> list[dict]:
    """Flatten a single QASPER paper into section-level records.

    Returns one record per section, plus one for the abstract and one for the title.
    """
    paper_id = paper["id"]
    title = paper["title"]
    abstract = paper["abstract"]
    full_text = paper["full_text"]

    records = []

    # Title as its own record (section_index = -2)
    if title and title.strip():
        records.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "section_name": "Title",
            "paragraphs": [title.strip()],
            "section_index": -2,
        })

    # Abstract as its own record (section_index = -1)
    if abstract and abstract.strip():
        records.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "section_name": "Abstract",
            "paragraphs": [abstract.strip()],
            "section_index": -1,
        })

    # Body sections
    section_names = full_text["section_name"]
    paragraphs_per_section = full_text["paragraphs"]

    for section_idx, (sec_name, paras) in enumerate(
        zip(section_names, paragraphs_per_section)
    ):
        # Filter out empty paragraphs
        paras = [p.strip() for p in paras if p and p.strip()]
        if not paras:
            continue

        records.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "section_name": sec_name,
            "paragraphs": paras,
            "section_index": section_idx,
        })

    return records


def load_and_flatten(config: dict) -> list[dict]:
    """Load QASPER from HuggingFace and flatten all papers."""
    ds_name = config["dataset"]["name"]
    splits = config["dataset"]["splits"]

    logger.info(f"Loading dataset: {ds_name}")
    try:
        ds = load_dataset(ds_name, trust_remote_code=True)
    except RuntimeError:
        # Newer datasets library doesn't support loading scripts;
        # fall back to the pre-converted parquet revision on HF Hub.
        logger.info("Falling back to refs/convert/parquet revision")
        ds = load_dataset(ds_name, revision="refs/convert/parquet")

    all_records = []
    total_papers = 0

    for split in splits:
        if split not in ds:
            logger.warning(f"Split '{split}' not found in dataset, skipping.")
            continue

        logger.info(f"Processing split: {split} ({len(ds[split])} papers)")
        for paper in tqdm(ds[split], desc=f"Flattening {split}"):
            records = flatten_paper(paper)
            all_records.extend(records)
            total_papers += 1

    logger.info(
        f"Total: {total_papers} papers -> {len(all_records)} section records"
    )
    return all_records


def main():
    parser = argparse.ArgumentParser(description="Load and flatten QASPER dataset")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: from config.yaml)"
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = resolve_path(args.output or config["paths"]["raw_data"])

    records = load_and_flatten(config)
    write_jsonl(records, output_path)
    logger.info(f"Wrote {len(records)} records to {output_path}")

    # Summary statistics
    paper_ids = set(r["paper_id"] for r in records)
    section_names = set(r["section_name"] for r in records)
    total_paragraphs = sum(len(r["paragraphs"]) for r in records)
    logger.info(f"Summary: {len(paper_ids)} papers, {len(section_names)} unique section names, {total_paragraphs} total paragraphs")


if __name__ == "__main__":
    main()
