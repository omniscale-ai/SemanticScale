"""Stage B: LLM extraction pipeline using Claude Haiku via Anthropic SDK."""

import json
import logging
import time
from pathlib import Path

import anthropic
import pandas as pd

from src.utils import (
    append_jsonl,
    get_project_root,
    load_config,
    read_jsonl,
    write_jsonl,
)

EXTRACTION_PROMPT_ABSTRACT = """You are a precise scientific information extractor. Given a paper title and abstract, extract all performance result tuples.

For each result mentioned, extract:
- method: The name of the model/method/system/approach
- dataset: The benchmark/dataset name
- metric: The evaluation metric name (e.g., accuracy, F1, BLEU, ROUGE-L)
- value: The numeric score/value (as a string, preserving the original format)
- year: The year if mentioned or inferrable from context (null if unknown)
- source_span: The exact sentence(s) from the abstract that support this extraction
- confidence: Your confidence in this extraction (0.0-1.0)

Rules:
- Only extract tuples where you can identify at least method + dataset + metric + value
- The source_span MUST be copied verbatim from the abstract
- If no performance tuples are found, return an empty array []
- Return ONLY valid JSON, no markdown formatting

Title: {title}

Abstract: {abstract}

Return a JSON array of objects with keys: method, dataset, metric, value, year, source_span, confidence"""

EXTRACTION_PROMPT_FULLTEXT = """You are a precise scientific information extractor. Given a paper title and key sections from the paper, extract all quantitative performance results.

For each result, extract:
- method: The name of the model/method/system/approach being evaluated
- dataset: The benchmark/dataset name (e.g., SQuAD, GLUE, CoNLL-2003)
- metric: The evaluation metric name (e.g., accuracy, F1, BLEU, ROUGE-L, EM, perplexity)
- value: The numeric score/value as a string, preserving the original format (e.g., "89.4", "0.874", "23.5")
- year: The year if mentioned or inferrable from context (null if unknown)
- source_span: The exact sentence or phrase from the text containing this result
- confidence: Your confidence in this extraction (0.0-1.0)

Rules:
- Focus on QUANTITATIVE results — numeric scores, percentages, ratios
- Only extract tuples where you can identify at least method + dataset + metric + value
- The value MUST be a specific number, not a qualitative statement
- Extract results for the paper's proposed method AND baselines/comparisons if clearly stated
- The source_span MUST be copied verbatim from the text
- If no quantitative results are found, return an empty array []
- Return ONLY valid JSON, no markdown formatting

Title: {title}

Paper sections:
{text}

Return a JSON array of objects with keys: method, dataset, metric, value, year, source_span, confidence"""


def build_extraction_prompt(title: str, abstract: str, relevant_text: str = None) -> str:
    """Build the extraction prompt for a given paper.

    If relevant_text is provided, uses full-text prompt; otherwise abstract-only.
    """
    if relevant_text and relevant_text.strip():
        # Truncate to ~3000 words to stay within reasonable context
        words = relevant_text.split()
        if len(words) > 3000:
            relevant_text = " ".join(words[:3000]) + "\n\n[...truncated...]"
        return EXTRACTION_PROMPT_FULLTEXT.format(
            title=title or "Unknown",
            text=relevant_text,
        )
    else:
        return EXTRACTION_PROMPT_ABSTRACT.format(
            title=title or "Unknown",
            abstract=abstract or "",
        )


def call_llm(
    client: anthropic.Anthropic,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
) -> str:
    """Call the Anthropic API with retry logic.

    Returns the raw text response.
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            delay = retry_base_delay * (2 ** attempt)
            logging.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except anthropic.APIError as e:
            delay = retry_base_delay * (2 ** attempt)
            logging.warning(f"API error: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Unexpected error calling LLM: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_base_delay)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_extraction_response(raw_response: str) -> list[dict]:
    """Parse the LLM response into a list of extraction dicts.

    Handles various JSON formatting issues.
    """
    text = raw_response.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are code fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            # Sometimes LLM wraps in {"results": [...]} or similar
            for key in ["results", "extractions", "tuples", "data"]:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]
        else:
            logging.warning(f"Unexpected JSON type: {type(parsed)}")
            return []
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parse error: {e}. Raw: {text[:200]}...")
        # Try to find a JSON array in the text
        import re
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


def extract_paper(
    client: anthropic.Anthropic,
    paper_id: str,
    title: str,
    abstract: str,
    ext_cfg: dict,
    extraction_counter: int = 0,
    relevant_text: str = None,
) -> list[dict]:
    """Extract tuples from a single paper.

    Returns list of extraction dicts with paper_id and extraction_id added.
    If relevant_text is provided, uses full-text extraction prompt.
    """
    prompt = build_extraction_prompt(title, abstract, relevant_text)

    raw_response = call_llm(
        client=client,
        prompt=prompt,
        model=ext_cfg["model"],
        max_tokens=ext_cfg["max_tokens"],
        temperature=ext_cfg["temperature"],
        max_retries=ext_cfg.get("max_retries", 3),
        retry_base_delay=ext_cfg.get("retry_base_delay", 2.0),
    )

    tuples = parse_extraction_response(raw_response)

    results = []
    for i, t in enumerate(tuples):
        ext_id = f"ext_{extraction_counter + i:04d}"
        record = {
            "paper_id": paper_id,
            "extraction_id": ext_id,
            "method": t.get("method", ""),
            "dataset": t.get("dataset", ""),
            "metric": t.get("metric", ""),
            "value": str(t.get("value", "")),
            "year": t.get("year"),
            "source_span": t.get("source_span", ""),
            "confidence": float(t.get("confidence", 0.5)),
            "raw_response": raw_response,
        }
        results.append(record)

    return results


def run_extraction(config: dict = None, project_root: Path = None) -> Path:
    """Run the full LLM extraction pipeline with checkpoint/resume.

    Returns path to the extractions JSONL file.
    """
    if config is None:
        config = load_config()
    if project_root is None:
        project_root = get_project_root()

    ext_cfg = config["extraction"]
    data_dir = project_root / config["data_dir"]

    # Input
    joined_path = data_dir / config["data_acquisition"]["output"]["joined"]
    if not joined_path.exists():
        raise FileNotFoundError(f"Joined data not found at {joined_path}. Run Stage A first.")

    joined_df = pd.read_parquet(joined_path)
    logging.info(f"Loaded {len(joined_df)} rows from {joined_path}")

    # Get unique papers with their metadata
    # Group by arxiv_id, take first title, abstract, and relevant_text
    agg_dict = {"title": "first", "abstract": "first"}
    if "relevant_text" in joined_df.columns:
        agg_dict["relevant_text"] = "first"
        logging.info("Full-text mode: relevant_text column found in joined data")
    papers = (
        joined_df.groupby("arxiv_id")
        .agg(agg_dict)
        .reset_index()
    )
    # Handle case where columns might not exist
    if "title" not in papers.columns:
        papers["title"] = ""
    has_fulltext = "relevant_text" in papers.columns
    logging.info(f"Unique papers to extract from: {len(papers)} "
                 f"(fulltext={'yes' if has_fulltext else 'no'})")

    # Output path
    output_path = data_dir / ext_cfg["output"]

    # Load existing extractions for resume
    processed_papers = set()
    existing_extractions = []
    if output_path.exists():
        existing_extractions = read_jsonl(output_path)
        processed_papers = {r["paper_id"] for r in existing_extractions}
        logging.info(f"Resuming: {len(processed_papers)} papers already processed, "
                     f"{len(existing_extractions)} extractions exist")

    # Papers to process
    remaining = papers[~papers["arxiv_id"].isin(processed_papers)]
    logging.info(f"Papers remaining: {len(remaining)}")

    if len(remaining) == 0:
        logging.info("All papers already processed.")
        return output_path

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    extraction_counter = len(existing_extractions)
    checkpoint_interval = ext_cfg.get("checkpoint_interval", 50)
    rate_limit_delay = 1.0 / ext_cfg.get("requests_per_second", 5)
    batch_extractions = []

    for idx, (_, row) in enumerate(remaining.iterrows()):
        paper_id = row["arxiv_id"]
        title = row.get("title", "") or ""
        abstract = row.get("abstract", "") or ""
        relevant_text = row.get("relevant_text", "") or "" if has_fulltext else ""

        if not abstract.strip() and not relevant_text.strip():
            logging.warning(f"Skipping paper {paper_id}: empty abstract and no full text")
            continue

        mode = "fulltext" if relevant_text.strip() else "abstract"
        logging.info(f"[{idx + 1}/{len(remaining)}] Extracting from {paper_id} ({mode})")

        try:
            extractions = extract_paper(
                client=client,
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                ext_cfg=ext_cfg,
                extraction_counter=extraction_counter,
                relevant_text=relevant_text if relevant_text.strip() else None,
            )
            extraction_counter += len(extractions)
            batch_extractions.extend(extractions)
            logging.info(f"  -> {len(extractions)} tuples extracted")

        except Exception as e:
            logging.error(f"Failed to extract from {paper_id}: {e}")
            continue

        # Rate limiting
        time.sleep(rate_limit_delay)

        # Checkpoint
        if (idx + 1) % checkpoint_interval == 0 and batch_extractions:
            append_jsonl(batch_extractions, output_path)
            logging.info(f"Checkpoint: saved {len(batch_extractions)} extractions "
                         f"(total: {extraction_counter})")
            batch_extractions = []

    # Final save
    if batch_extractions:
        append_jsonl(batch_extractions, output_path)
        logging.info(f"Final save: {len(batch_extractions)} extractions")

    total_extractions = len(read_jsonl(output_path)) if output_path.exists() else 0
    logging.info(f"=== Extraction complete: {total_extractions} total tuples ===")

    return output_path
