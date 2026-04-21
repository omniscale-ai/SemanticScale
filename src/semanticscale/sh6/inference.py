"""Async OpenAI Responses API inference for SH6."""

import asyncio
import datetime
from datetime import timezone
import logging
import os
import re

import openai

from semanticscale.openai_utils import (
    create_response,
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
    r"FINAL ANSWER\b\s*[:\-]?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_predicted_answer(answer_text: str, has_final_answer: bool) -> str:
    """Best-effort extraction of the predicted answer text."""
    if not answer_text:
        return ""
    if has_final_answer:
        matches = list(_FINAL_ANSWER_RE.finditer(answer_text))
        if matches:
            return matches[-1].group(1).strip()
    return answer_text


async def _call_one(
    client: openai.AsyncOpenAI,
    item: dict,
    model: str,
    reasoning: dict,
    service_tier: str | None,
    extra_body: dict | None,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> dict:
    """Call the Responses API for a single item, with tenacity retry."""

    async with semaphore:
        try:
            response = await create_response(
                client=client,
                model=model,
                prompt=item["problem"],
                reasoning=reasoning,
                service_tier=service_tier,
                extra_body=extra_body,
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
    return result


async def _run_async(
    items: list[dict],
    model: str,
    reasoning: dict,
    base_url: str | None,
    api_key_env: str | None,
    service_tier: str | None,
    extra_body: dict | None,
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
            service_tier=service_tier,
            extra_body=extra_body,
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
    config: dict,
    model_override: str | None = None,
    service_tier_override: str | None = None,
) -> list[dict]:
    """Run async batch inference on items using the ``traces`` config section."""
    traces_cfg = config.get("traces", {})
    model_cfg = traces_cfg.get("model", {})
    model = model_override or model_cfg["name"]
    service_tier = service_tier_override or model_cfg.get("service_tier")

    return asyncio.run(
        _run_async(
            items=items,
            model=model,
            reasoning=model_cfg.get("reasoning", {}),
            base_url=model_cfg.get("base_url"),
            api_key_env=model_cfg.get("api_key_env"),
            service_tier=service_tier,
            extra_body=model_cfg.get("extra_body"),
            max_concurrent=traces_cfg.get("max_concurrent", 10),
            max_retries=traces_cfg.get("max_retries", 5),
            retry_min_wait=traces_cfg.get("retry_min_wait", 1.0),
            retry_max_wait=traces_cfg.get("retry_max_wait", 60.0),
        )
    )
