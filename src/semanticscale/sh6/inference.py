"""Async OpenAI Responses API inference for SH6."""

import asyncio
import datetime
from datetime import timezone
import logging
import os
import re

import openai

from semanticscale.openai_utils import (
    create_chat_completion,
    create_response,
    extract_chat_completion_text,
    extract_response_text,
    extract_usage,
)

logger = logging.getLogger(__name__)


def make_model_slug(
    model: str,
    reasoning: dict,
    question_types: list[str] | None = None,
) -> str:
    """Return a filesystem-safe identifier for a model configuration.

    When *question_types* is given (e.g. ``["olympiad"]``), a suffix is appended
    so filtered runs are stored in a separate output directory.
    """
    if reasoning and reasoning.get("enabled", True) is False:
        effort = "none"
    else:
        effort = reasoning.get("effort", "auto") if reasoning else "none"
        
    slug = f"{model}_reasoning-{effort}"
    if question_types:
        slug += "_types-" + "+".join(sorted(qt.lower() for qt in question_types))
    return slug


_FINAL_ANSWER_RE = re.compile(
    r"FINAL ANSWER\b\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z ]{2,}\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _normalise_answer_text(text: str) -> str:
    """Normalise short-form answers for comparison."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*FINAL ANSWER\b\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("`'\"").strip()
    cleaned = re.sub(r"^\$\$(.*)\$\$$", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^\\\[(.*)\\\]$", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^\\\((.*)\\\)$", r"\1", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip().strip("`'\"").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_predicted_answer(answer_text: str, has_final_answer: bool) -> str:
    """Best-effort extraction of the predicted answer text."""
    if not answer_text:
        return ""
    if has_final_answer:
        matches = list(_FINAL_ANSWER_RE.finditer(answer_text))
        if matches:
            return _normalise_answer_text(matches[-1].group(1))
    return _normalise_answer_text(answer_text.strip().split("\n")[0][:200])


def _is_correct(predicted: str, correct: str) -> bool:
    if not predicted or not correct:
        return False
    return _normalise_answer_text(predicted).lower() == _normalise_answer_text(correct).lower()


async def _call_one(
    client: openai.AsyncOpenAI,
    item: dict,
    model: str,
    reasoning: dict,
    api_type: str,
    service_tier: str | None,
    max_output_tokens: int | None,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> dict:
    """Call the chosen API for a single item, with tenacity retry."""

    async with semaphore:
        try:
            if api_type == "completions":
                response = await create_chat_completion(
                    client=client,
                    model=model,
                    prompt=item["problem"],
                    reasoning=reasoning,
                    service_tier=service_tier,
                    max_output_tokens=max_output_tokens,
                    max_retries=max_retries,
                    retry_min_wait=retry_min_wait,
                    retry_max_wait=retry_max_wait,
                )
            else:
                response = await create_response(
                    client=client,
                    model=model,
                    prompt=item["problem"],
                    reasoning=reasoning,
                    service_tier=service_tier,
                    max_output_tokens=max_output_tokens,
                    max_retries=max_retries,
                    retry_min_wait=retry_min_wait,
                    retry_max_wait=retry_max_wait,
                )
        except Exception as exc:
            logger.exception("Failed item %s: %s", item["id"], exc)
            return {
                "id": item["id"],
                "model": model,
                "reasoning_effort": reasoning.get("effort", "auto"),
                "service_tier": service_tier,
                "model_slug": make_model_slug(model, reasoning),
                "problem": item["problem"],
                "correct_answer": item["correct_answer"],
                "subject": item.get("subject", "unknown"),
                "has_final_answer": item.get("has_final_answer", True),
                "reasoning_text": "",
                "answer_text": "",
                "usage": None,
                "error": str(exc),
                "timestamp": datetime.datetime.now(tz=timezone.utc).isoformat(),
            }

    if api_type == "completions":
        reasoning_text, answer_text = extract_chat_completion_text(response)
    else:
        reasoning_text, answer_text = extract_response_text(response)
        
    has_final_answer = item.get("has_final_answer", True)
    usage = extract_usage(response)

    result = {
        "id": item["id"],
        "model": model,
        "reasoning_effort": reasoning.get("effort", "auto"),
        "service_tier": service_tier,
        "model_slug": make_model_slug(model, reasoning),
        "problem": item["problem"],
        "correct_answer": item["correct_answer"],
        "subject": item.get("subject", "unknown"),
        "has_final_answer": has_final_answer,
        "reasoning_text": reasoning_text,
        "answer_text": answer_text,
        "usage": usage,
        "error": None,
        "timestamp": datetime.datetime.now(tz=timezone.utc).isoformat(),
    }
    if has_final_answer:
        predicted = _extract_predicted_answer(answer_text, has_final_answer)
        result["predicted_answer"] = predicted
        result["is_correct"] = _is_correct(predicted, item["correct_answer"])
    return result


async def _run_async(
    items: list[dict],
    model: str,
    reasoning: dict,
    api_type: str,
    base_url: str | None,
    api_key_env: str | None,
    service_tier: str | None,
    max_output_tokens: int | None,
    max_concurrent: int,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> list[dict]:
    api_key = os.environ.get(api_key_env) if api_key_env else os.environ.get("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _call_one(
            client=client,
            item=item,
            model=model,
            reasoning=reasoning,
            api_type=api_type,
            service_tier=service_tier,
            max_output_tokens=max_output_tokens,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
        )
        for item in items
    ]

    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        if i % 50 == 0 or i == len(tasks):
            n_errors = sum(1 for r in results if r.get("error"))
            logger.info(
                "Progress %d/%d — %d errors so far",
                i,
                len(tasks),
                n_errors,
            )

    return results


def run_inference(
    items: list[dict],
    model: str,
    service_tier: str | None,
    config: dict,
) -> list[dict]:
    """Run async batch inference on items.

    Returns a list of result records (one per item).
    """
    inf = config.get("inference", {})
    model_config = config.get("model", {})
    return asyncio.run(
        _run_async(
            items=items,
            model=model,
            reasoning=model_config.get("reasoning", {}),
            api_type=model_config.get("api_type", "responses"),
            base_url=model_config.get("base_url"),
            api_key_env=model_config.get("api_key_env"),
            service_tier=service_tier,
            max_output_tokens=inf.get("max_output_tokens", 4096),
            max_concurrent=inf.get("max_concurrent", 10),
            max_retries=inf.get("max_retries", 5),
            retry_min_wait=inf.get("retry_min_wait", 1.0),
            retry_max_wait=inf.get("retry_max_wait", 60.0),
        )
    )
