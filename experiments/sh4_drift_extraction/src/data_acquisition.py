"""Stage A: Data acquisition from HuggingFace Papers with Code datasets.

Supports two modes:
1. Abstract-only (iter 1): PwC papers + abstracts
2. Full-text (iter 2): QASPER full text joined with PwC eval tables
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.utils import extract_arxiv_id, load_config, get_project_root


def load_pwc_eval_tables(dataset_name: str) -> pd.DataFrame:
    """Load PwC evaluation tables from HuggingFace.

    Returns DataFrame with columns including:
    task, dataset, model_name, paper_url, metric_name, metric_value, etc.
    """
    logging.info(f"Loading PwC evaluation tables from {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()
    logging.info(f"Loaded {len(df)} rows from PwC evaluation tables")
    logging.info(f"Columns: {list(df.columns)}")
    return df


def load_pwc_papers(dataset_name: str) -> pd.DataFrame:
    """Load PwC papers metadata from HuggingFace.

    Returns DataFrame with columns including:
    arxiv_id, title, abstract, published, etc.
    """
    logging.info(f"Loading PwC papers metadata from {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()
    logging.info(f"Loaded {len(df)} papers from PwC metadata")
    logging.info(f"Columns: {list(df.columns)}")
    return df


def filter_and_extract_arxiv_ids(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Filter evaluation table to rows with extractable arxiv IDs.

    Adds 'arxiv_id' column.
    """
    # Try to extract arxiv_id from paper_url (or paper_url-like columns)
    url_col = None
    for candidate in ["paper_url", "paper_link", "url"]:
        if candidate in eval_df.columns:
            url_col = candidate
            break

    if url_col is None:
        logging.warning("No URL column found. Trying all string columns...")
        # Fall back: look for any column containing arxiv URLs
        for col in eval_df.columns:
            if eval_df[col].dtype == object:
                sample = eval_df[col].dropna().head(100)
                if sample.str.contains("arxiv", case=False).any():
                    url_col = col
                    logging.info(f"Found arxiv URLs in column: {col}")
                    break

    if url_col is None:
        raise ValueError("Could not find a column with arxiv URLs in evaluation table")

    logging.info(f"Extracting arxiv IDs from column: {url_col}")
    eval_df = eval_df.copy()
    eval_df["arxiv_id"] = eval_df[url_col].apply(extract_arxiv_id)

    n_before = len(eval_df)
    eval_df = eval_df.dropna(subset=["arxiv_id"])
    logging.info(f"Filtered to {len(eval_df)} rows with arxiv IDs (from {n_before})")

    return eval_df


def filter_papers_with_min_tuples(
    eval_df: pd.DataFrame, min_tuples: int
) -> pd.DataFrame:
    """Keep only papers that have >= min_tuples evaluation rows."""
    paper_counts = eval_df.groupby("arxiv_id").size()
    valid_papers = paper_counts[paper_counts >= min_tuples].index
    filtered = eval_df[eval_df["arxiv_id"].isin(valid_papers)]
    logging.info(
        f"Filtered to {len(valid_papers)} papers with >= {min_tuples} tuples "
        f"({len(filtered)} rows total)"
    )
    return filtered


def sample_papers(
    eval_df: pd.DataFrame, n_papers: int, seed: int
) -> pd.DataFrame:
    """Sample n_papers unique papers and return all their evaluation rows."""
    unique_papers = eval_df["arxiv_id"].unique()
    logging.info(f"Available papers: {len(unique_papers)}, sampling {n_papers}")

    n_sample = min(n_papers, len(unique_papers))
    rng = np.random.RandomState(seed)
    sampled_papers = rng.choice(unique_papers, size=n_sample, replace=False)
    sampled = eval_df[eval_df["arxiv_id"].isin(sampled_papers)]
    logging.info(
        f"Sampled {n_sample} papers with {len(sampled)} evaluation rows"
    )
    return sampled


def join_with_abstracts(
    eval_df: pd.DataFrame, papers_df: pd.DataFrame
) -> pd.DataFrame:
    """Join evaluation table with paper metadata to get abstracts.

    Tries to join on arxiv_id. Falls back to fuzzy title matching if needed.
    """
    # Identify the arxiv_id column in papers_df
    arxiv_col = None
    for candidate in ["arxiv_id", "arxiv", "paper_arxiv_id"]:
        if candidate in papers_df.columns:
            arxiv_col = candidate
            break

    if arxiv_col is None:
        # Try to extract from URLs or other columns
        logging.warning("No direct arxiv_id column in papers. Trying URL extraction...")
        for col in papers_df.columns:
            if papers_df[col].dtype == object:
                sample = papers_df[col].dropna().head(100)
                if sample.str.contains("arxiv", case=False).any():
                    papers_df = papers_df.copy()
                    papers_df["arxiv_id_extracted"] = papers_df[col].apply(extract_arxiv_id)
                    arxiv_col = "arxiv_id_extracted"
                    papers_df = papers_df.dropna(subset=[arxiv_col])
                    logging.info(f"Extracted arxiv IDs from papers column: {col}")
                    break

    if arxiv_col is None:
        raise ValueError("Could not find arxiv_id in papers dataset")

    # Rename for join
    if arxiv_col != "arxiv_id":
        papers_df = papers_df.rename(columns={arxiv_col: "arxiv_id"})

    # Deduplicate papers by arxiv_id (keep first)
    papers_dedup = papers_df.drop_duplicates(subset="arxiv_id", keep="first")
    logging.info(f"Papers dataset: {len(papers_dedup)} unique arxiv IDs")

    # Find abstract column
    abstract_col = None
    for candidate in ["abstract", "paper_abstract", "Abstract"]:
        if candidate in papers_dedup.columns:
            abstract_col = candidate
            break

    if abstract_col is None:
        raise ValueError("Could not find abstract column in papers dataset")

    # Find title column
    title_col = None
    for candidate in ["title", "paper_title", "Title"]:
        if candidate in papers_dedup.columns:
            title_col = candidate
            break

    # Select relevant columns from papers
    keep_cols = ["arxiv_id"]
    if abstract_col:
        keep_cols.append(abstract_col)
    if title_col:
        keep_cols.append(title_col)

    # Add any date/year columns
    for col in papers_dedup.columns:
        if col.lower() in ["published", "date", "year", "paper_date"]:
            keep_cols.append(col)

    papers_subset = papers_dedup[keep_cols].copy()

    # Standardize column names
    rename_map = {}
    if abstract_col and abstract_col != "abstract":
        rename_map[abstract_col] = "abstract"
    if title_col and title_col != "title":
        rename_map[title_col] = "title"
    if rename_map:
        papers_subset = papers_subset.rename(columns=rename_map)

    # Join
    joined = eval_df.merge(papers_subset, on="arxiv_id", how="inner")
    n_papers_joined = joined["arxiv_id"].nunique()
    logging.info(
        f"Joined: {len(joined)} rows, {n_papers_joined} papers with abstracts "
        f"(from {eval_df['arxiv_id'].nunique()} eval papers)"
    )

    # Drop rows with missing abstracts
    if "abstract" in joined.columns:
        joined = joined.dropna(subset=["abstract"])
        joined = joined[joined["abstract"].str.strip().str.len() > 0]
        logging.info(f"After dropping empty abstracts: {len(joined)} rows, "
                     f"{joined['arxiv_id'].nunique()} papers")

    return joined


def run_acquisition(config: dict = None, project_root: Path = None) -> Path:
    """Run the full data acquisition pipeline.

    Returns path to the joined parquet file.
    """
    if config is None:
        config = load_config()
    if project_root is None:
        project_root = get_project_root()

    acq_cfg = config["data_acquisition"]
    data_dir = project_root / config["data_dir"]

    # Output paths
    eval_path = data_dir / acq_cfg["output"]["eval_tables"]
    papers_path = data_dir / acq_cfg["output"]["papers"]
    joined_path = data_dir / acq_cfg["output"]["joined"]

    # Check if already done
    if joined_path.exists():
        logging.info(f"Joined data already exists at {joined_path}, skipping acquisition.")
        return joined_path

    # Step 1: Load PwC evaluation tables
    if eval_path.exists():
        logging.info(f"Loading cached eval tables from {eval_path}")
        eval_df = pd.read_parquet(eval_path)
    else:
        eval_df = load_pwc_eval_tables(acq_cfg["pwc_eval_dataset"])
        eval_df = filter_and_extract_arxiv_ids(eval_df)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_parquet(eval_path, index=False)
        logging.info(f"Saved eval tables to {eval_path}")

    # Step 2: Filter to papers with enough tuples
    eval_df = filter_papers_with_min_tuples(eval_df, acq_cfg["min_tuples_per_paper"])

    # Step 3: Load PwC papers metadata
    if papers_path.exists():
        logging.info(f"Loading cached papers from {papers_path}")
        papers_df = pd.read_parquet(papers_path)
    else:
        papers_df = load_pwc_papers(acq_cfg["pwc_papers_dataset"])
        papers_path.parent.mkdir(parents=True, exist_ok=True)
        papers_df.to_parquet(papers_path, index=False)
        logging.info(f"Saved papers to {papers_path}")

    # Step 4: Join to get abstracts
    joined_df = join_with_abstracts(eval_df, papers_df)

    # Step 5: Sample papers
    joined_df = sample_papers(joined_df, acq_cfg["n_papers"], acq_cfg["seed"])

    # Step 6: Save
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(joined_path, index=False)
    logging.info(f"Saved joined data to {joined_path}")

    # Summary stats
    n_papers = joined_df["arxiv_id"].nunique()
    n_rows = len(joined_df)
    logging.info(f"=== Acquisition complete: {n_papers} papers, {n_rows} eval rows ===")

    return joined_path


# --- QASPER full-text acquisition (iteration 2) ---

# Section names that likely contain results/methods with numeric values
RESULT_SECTION_KEYWORDS = [
    "result", "experiment", "evaluation", "performance", "comparison",
    "ablation", "analysis", "benchmark", "discussion", "method",
    "approach", "model", "baseline", "setup", "implementation",
]


def _is_relevant_section(section_name: str) -> bool:
    """Check if a section name is likely to contain extractable results."""
    name_lower = section_name.lower()
    # Always include abstract
    if name_lower in ("abstract", "title"):
        return True
    return any(kw in name_lower for kw in RESULT_SECTION_KEYWORDS)


def load_qasper_fulltext(qasper_path: Path) -> dict[str, dict]:
    """Load QASPER full text from data_sh0/qasper_raw.jsonl.

    Returns dict: paper_id -> {title, abstract, full_text, relevant_text, sections}
    where relevant_text is the concatenation of result/experiment/method sections.
    """
    logging.info(f"Loading QASPER full text from {qasper_path}...")

    papers = {}
    with open(qasper_path) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["paper_id"]
            section_name = rec.get("section_name", "") or ""
            paragraphs = rec.get("paragraphs", [])
            text = "\n".join(str(p) for p in paragraphs) if isinstance(paragraphs, list) else str(paragraphs)

            if pid not in papers:
                papers[pid] = {
                    "title": rec.get("title", ""),
                    "abstract": "",
                    "sections": [],
                    "all_text_parts": [],
                    "relevant_text_parts": [],
                }

            # Store abstract separately
            if section_name.lower() == "abstract":
                papers[pid]["abstract"] = text

            # Store section info
            papers[pid]["sections"].append(section_name)
            papers[pid]["all_text_parts"].append(text)

            # Store relevant sections
            if _is_relevant_section(section_name):
                papers[pid]["relevant_text_parts"].append(
                    f"## {section_name}\n{text}"
                )

    # Build final text fields
    for pid, paper in papers.items():
        paper["full_text"] = "\n\n".join(paper["all_text_parts"])
        paper["relevant_text"] = "\n\n".join(paper["relevant_text_parts"])
        # Clean up temporary fields
        del paper["all_text_parts"]
        del paper["relevant_text_parts"]

    logging.info(f"Loaded {len(papers)} QASPER papers with full text")

    # Log some stats
    text_lengths = [len(p["relevant_text"].split()) for p in papers.values()]
    logging.info(f"Relevant text word counts: "
                 f"median={np.median(text_lengths):.0f}, "
                 f"mean={np.mean(text_lengths):.0f}, "
                 f"p95={np.percentile(text_lengths, 95):.0f}")

    return papers


def acquire_qasper_pwc_data(config: dict, project_root: Path) -> Path:
    """Acquire QASPER full-text papers joined with PwC evaluation tables.

    This is the iteration 2 data source, replacing abstract-only extraction.

    Returns path to the joined parquet file (same format as iter 1 but with
    an additional 'relevant_text' column containing key paper sections).
    """
    acq_cfg = config["data_acquisition"]
    data_dir = project_root / config["data_dir"]
    sh0_data_dir = project_root / config["sh0_data_dir"]

    # Output path (same as iter 1 — we overwrite)
    joined_path = data_dir / acq_cfg["output"]["joined"]

    # Check if already done
    if joined_path.exists():
        logging.info(f"Joined data already exists at {joined_path}, skipping.")
        return joined_path

    # Step 1: Load PwC eval tables (reuse cached from iter 1)
    eval_path = data_dir / acq_cfg["output"]["eval_tables"]
    if eval_path.exists():
        logging.info(f"Loading cached eval tables from {eval_path}")
        eval_df = pd.read_parquet(eval_path)
    else:
        eval_df = load_pwc_eval_tables(acq_cfg["pwc_eval_dataset"])
        eval_df = filter_and_extract_arxiv_ids(eval_df)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_parquet(eval_path, index=False)

    # Step 2: Load QASPER full text
    qasper_path = sh0_data_dir / "qasper_raw.jsonl"
    if not qasper_path.exists():
        raise FileNotFoundError(f"QASPER data not found at {qasper_path}")

    qasper_papers = load_qasper_fulltext(qasper_path)
    qasper_ids = set(qasper_papers.keys())

    # Step 3: Find overlap with PwC
    pwc_ids = set(eval_df["arxiv_id"].dropna().unique())
    overlap_ids = qasper_ids & pwc_ids
    logging.info(f"QASPER papers: {len(qasper_ids)}, PwC papers: {len(pwc_ids)}, "
                 f"Overlap: {len(overlap_ids)}")

    if len(overlap_ids) == 0:
        raise ValueError("No overlap between QASPER and PwC papers!")

    # Step 4: Filter eval_df to overlap papers (use min_tuples=1 for max coverage)
    min_tuples = max(1, acq_cfg.get("min_tuples_per_paper", 1))
    overlap_eval = eval_df[eval_df["arxiv_id"].isin(overlap_ids)]

    if min_tuples > 1:
        paper_counts = overlap_eval.groupby("arxiv_id").size()
        valid_papers = paper_counts[paper_counts >= min_tuples].index
        overlap_eval = overlap_eval[overlap_eval["arxiv_id"].isin(valid_papers)]

    logging.info(f"Overlap eval rows: {len(overlap_eval)}, "
                 f"papers: {overlap_eval['arxiv_id'].nunique()}")

    # Step 5: Build joined DataFrame with full text
    rows = []
    for _, eval_row in overlap_eval.iterrows():
        pid = eval_row["arxiv_id"]
        paper = qasper_papers.get(pid, {})
        row = eval_row.to_dict()
        row["title"] = paper.get("title", "")
        row["abstract"] = paper.get("abstract", "")
        row["relevant_text"] = paper.get("relevant_text", "")
        rows.append(row)

    joined_df = pd.DataFrame(rows)

    # Drop rows with empty relevant_text
    joined_df = joined_df[joined_df["relevant_text"].str.strip().str.len() > 0]
    logging.info(f"After filtering empty text: {len(joined_df)} rows, "
                 f"{joined_df['arxiv_id'].nunique()} papers")

    # Step 6: Sample if requested (use all overlap papers if n_papers > available)
    n_papers = acq_cfg.get("n_papers", 300)
    n_available = joined_df["arxiv_id"].nunique()
    if n_available > n_papers:
        joined_df = sample_papers(joined_df, n_papers, acq_cfg["seed"])
    else:
        logging.info(f"Using all {n_available} overlap papers (requested {n_papers})")

    # Step 7: Save
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(joined_path, index=False)

    n_papers_final = joined_df["arxiv_id"].nunique()
    n_rows_final = len(joined_df)
    logging.info(f"=== QASPER+PwC acquisition complete: {n_papers_final} papers, "
                 f"{n_rows_final} eval rows ===")

    return joined_path
