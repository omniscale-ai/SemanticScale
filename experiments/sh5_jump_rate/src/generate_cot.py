"""Stage B: Generate chain-of-thought reasoning traces via Claude Haiku."""

import json
import logging
import re
import time
from pathlib import Path

import anthropic

from src.utils import load_jsonl, resolve_path, save_jsonl

logger = logging.getLogger(__name__)


def build_prompt(question_text: str, evidence_texts: list[str], config: dict) -> str:
    """Construct the CoT prompt from template and evidence."""
    # Number evidence passages
    evidence_parts = []
    for i, text in enumerate(evidence_texts, 1):
        evidence_parts.append(f"[{i}] {text.strip()}")
    evidence_str = "\n\n".join(evidence_parts)

    template = config["cot"]["prompt_template"]
    prompt = template.format(question_text=question_text, evidence=evidence_str)
    return prompt


def call_haiku(
    prompt: str,
    client: anthropic.Anthropic,
    config: dict,
) -> str:
    """Single API call to Claude Haiku with retry/backoff.

    Returns the response text.
    """
    model = config["cot"]["model"]
    max_tokens = config["cot"]["max_tokens"]
    temperature = config["cot"]["temperature"]
    max_retries = config["cot"]["max_retries"]
    base_delay = config["cot"]["retry_base_delay"]
    max_delay = config["cot"]["max_retry_delay"]

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                raise
        except anthropic.APIError as e:
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"API error: {e}, retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                raise

    return ""  # Should not reach here


def parse_cot_response(response_text: str) -> dict:
    """Parse CoT response into numbered steps and final answer.

    Returns dict with 'cot_steps' (list of str) and 'final_answer' (str).
    """
    # Extract final answer
    final_answer = ""
    answer_match = re.search(r"ANSWER:\s*(.*)", response_text, re.DOTALL)
    if answer_match:
        final_answer = answer_match.group(1).strip()
        # Remove the ANSWER: part for step parsing
        text_before_answer = response_text[:answer_match.start()].strip()
    else:
        text_before_answer = response_text.strip()
        # Try to get last sentence as answer
        lines = text_before_answer.strip().split("\n")
        if lines:
            final_answer = lines[-1].strip()

    # Parse numbered steps
    # Pattern: "Step N:" or "Step N." at the start of a segment
    step_pattern = re.compile(r"Step\s+\d+\s*[:.]", re.IGNORECASE)
    splits = step_pattern.split(text_before_answer)

    # First split is text before "Step 1:" — discard preamble (not a reasoning step)
    steps = []
    for idx, part in enumerate(splits):
        part = part.strip()
        if not part:
            continue
        # Skip preamble: the first segment is text before the first "Step N:" marker
        if idx == 0 and len(splits) > 1:
            continue
        # Clean up: remove trailing empty lines
        part = re.sub(r"\s+$", "", part)
        if part:
            steps.append(part)

    # If no "Step N:" pattern found, try splitting on numbered lines like "1." or "1)"
    if len(steps) <= 1:
        numbered_pattern = re.compile(r"\n\s*\d+[.)]\s+")
        splits = numbered_pattern.split(text_before_answer)
        steps = [s.strip() for s in splits if s.strip()]

    # Last resort: split on double newlines
    if len(steps) <= 1:
        steps = [s.strip() for s in text_before_answer.split("\n\n") if s.strip()]

    # If still just one block, keep it as a single step
    if not steps:
        steps = [text_before_answer] if text_before_answer else []

    return {
        "cot_steps": steps,
        "final_answer": final_answer,
    }


def load_retrieval_for_condition(
    condition: str, top_k: int, config: dict
) -> dict[str, list[str]]:
    """Load SH3 retrieval results for a specific condition and k.

    Returns dict mapping question_id -> list of retrieved text passages.
    """
    path = resolve_path(
        config, config["paths"]["sh3_data_dir"], config["paths"]["sh3_retrieval_results"]
    )
    logger.info(f"Loading retrieval results from {path}")
    with open(path) as f:
        all_results = json.load(f)

    if condition not in all_results:
        raise ValueError(
            f"Condition '{condition}' not found. Available: {list(all_results.keys())}"
        )

    k_str = str(top_k)
    if k_str not in all_results[condition]:
        raise ValueError(
            f"k={top_k} not found for condition '{condition}'. "
            f"Available: {list(all_results[condition].keys())}"
        )

    results_for_k = all_results[condition][k_str]
    retrieval_map = {}
    for item in results_for_k:
        qid = item["question_id"]
        texts = [r["text"] for r in item.get("retrieved", [])]
        retrieval_map[qid] = texts

    logger.info(
        f"Loaded retrieval for condition={condition}, k={top_k}: "
        f"{len(retrieval_map)} questions"
    )
    return retrieval_map


def generate_for_condition(
    questions: list[dict],
    condition: str,
    retrieval_map: dict[str, list[str]],
    client: anthropic.Anthropic,
    config: dict,
    existing_ids: set[str] | None = None,
) -> list[dict]:
    """Generate CoT traces for all questions under one condition.

    Supports resume by skipping question_ids in existing_ids.
    """
    rpm_limit = config["cot"]["requests_per_minute"]
    min_interval = 60.0 / rpm_limit

    results = []
    skipped = 0
    errors = 0

    for i, q in enumerate(questions):
        qid = q["question_id"]

        if existing_ids and qid in existing_ids:
            skipped += 1
            continue

        evidence_texts = retrieval_map.get(qid, [])
        if not evidence_texts:
            logger.warning(f"No evidence for {qid} under {condition}, skipping")
            continue

        prompt = build_prompt(q["question_text"], evidence_texts, config)

        start_time = time.time()
        try:
            raw_response = call_haiku(prompt, client, config)
            parsed = parse_cot_response(raw_response)

            result = {
                "question_id": qid,
                "condition": condition,
                "retrieved_texts": evidence_texts,
                "raw_response": raw_response,
                "cot_steps": parsed["cot_steps"],
                "final_answer": parsed["final_answer"],
                "n_steps": len(parsed["cot_steps"]),
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error generating CoT for {qid}/{condition}: {e}")
            errors += 1
            continue

        # Rate limiting
        elapsed = time.time() - start_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        if (i + 1 - skipped) % 50 == 0:
            logger.info(
                f"[{condition}] Progress: {i + 1}/{len(questions)} "
                f"(generated: {len(results)}, skipped: {skipped}, errors: {errors})"
            )

    logger.info(
        f"[{condition}] Done: {len(results)} generated, {skipped} skipped, {errors} errors"
    )
    return results


def generate_all_cot(config: dict) -> dict[str, Path]:
    """Orchestrate CoT generation for all conditions with resume support.

    Returns dict mapping condition -> output path.
    """
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    traces_dir = data_dir / "cot_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Load selected questions
    questions_path = data_dir / "selected_questions.jsonl"
    questions = load_jsonl(questions_path)
    logger.info(f"Loaded {len(questions)} selected questions")

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    conditions = config["retrieval"]["conditions"]
    top_k = config["retrieval"]["top_k"]
    output_paths = {}

    for condition in conditions:
        output_path = traces_dir / f"{condition}.jsonl"
        output_paths[condition] = output_path

        # Check for existing results (resume support)
        existing_ids = set()
        existing_records = []
        if output_path.exists():
            existing_records = load_jsonl(output_path)
            existing_ids = {r["question_id"] for r in existing_records}
            logger.info(
                f"[{condition}] Found {len(existing_ids)} existing traces, resuming"
            )

        if len(existing_ids) >= len(questions):
            logger.info(f"[{condition}] All questions already processed, skipping")
            continue

        # Load retrieval results for this condition
        retrieval_map = load_retrieval_for_condition(condition, top_k, config)

        # Generate
        new_results = generate_for_condition(
            questions, condition, retrieval_map, client, config, existing_ids
        )

        # Merge and save
        all_results = existing_records + new_results
        save_jsonl(all_results, output_path)
        logger.info(f"[{condition}] Saved {len(all_results)} traces to {output_path}")

    # Report summary
    total_traces = 0
    total_steps = 0
    parse_failures = 0
    for condition in conditions:
        path = output_paths[condition]
        if path.exists():
            traces = load_jsonl(path)
            total_traces += len(traces)
            for t in traces:
                total_steps += t["n_steps"]
                if t["n_steps"] == 0:
                    parse_failures += 1

    logger.info(
        f"CoT generation complete: {total_traces} traces, {total_steps} total steps, "
        f"{parse_failures} parse failures "
        f"({100 * (1 - parse_failures / max(total_traces, 1)):.1f}% parse success)"
    )

    return output_paths
