"""Async LLM inference for SH6.

Dispatches between the OpenAI Responses API and the OpenRouter Python SDK
via :mod:`semanticscale.llm_backend`. OpenRouter is selected when the
config's ``traces.model.base_url`` points at ``openrouter.ai`` or when
``api_key_env`` is ``OPENROUTER_API_KEY``.
"""

import asyncio
import datetime
from datetime import timezone
import logging
import re

from semanticscale.llm_backend import Backend, make_backend

logger = logging.getLogger(__name__)


def make_model_slug(
    model: str,
    reasoning: dict,
    question_types: list[str] | None = None,
    sample_idx: int | None = None,
) -> str:
    """Return a filesystem-safe identifier for a model configuration.

    When *question_types* is given (e.g. ``["olympiad"]``), a suffix is appended
    so filtered runs are stored in a separate output directory. When
    *sample_idx* is given, an ``_s{N}`` suffix marks one sample of a
    best-of-N draw, so each sample's artifacts live in their own directory.
    """
    if reasoning and reasoning.get("enabled", True) is False:
        effort = "none"
    else:
        effort = reasoning.get("effort", "auto") if reasoning else "none"

    slug = f"{model}_reasoning-{effort}"
    if question_types:
        slug += "_types-" + "+".join(sorted(qt.lower() for qt in question_types))
    if sample_idx is not None:
        slug += f"_s{sample_idx}"
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
    backend: Backend,
    item: dict,
    model: str,
    reasoning: dict,
    service_tier: str | None,
    extra_body: dict | None,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
    sample_idx: int | None = None,
) -> dict:
    """Call the backend for a single item, with retry."""

    async with semaphore:
        try:
            out = await backend.create(
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
                "model_slug": make_model_slug(model, reasoning, sample_idx=sample_idx),
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

    has_final_answer = item.get("has_final_answer", True)
    reasoning_text = out["reasoning_text"]
    answer_text = out["answer_text"]

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
        "usage": out["usage"],
        "finish_reason": out.get("finish_reason"),
        "error": None,
        "timestamp": datetime.datetime.now(tz=timezone.utc).isoformat(),
    }
    if has_final_answer:
        result["predicted_answer"] = _extract_predicted_answer(answer_text, has_final_answer)
    return result


async def _run_async(
    items: list[dict],
    model_cfg: dict,
    model: str,
    reasoning: dict,
    service_tier: str | None,
    extra_body: dict | None,
    max_concurrent: int,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
    sample_idx: int | None = None,
) -> list[dict]:
    backend = make_backend(model_cfg)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _call_one(
            backend=backend,
            item=item,
            model=model,
            reasoning=reasoning,
            service_tier=service_tier,
            extra_body=extra_body,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
            sample_idx=sample_idx,
        )
        for item in items
    ]

    try:
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
    finally:
        await backend.aclose()


def run_inference(
    items: list[dict],
    config: dict,
    model_override: str | None = None,
    service_tier_override: str | None = None,
    sample_idx: int | None = None,
) -> list[dict]:
    """Run async batch inference on items using the ``traces`` config section.

    When *sample_idx* is given, the inference call is biased toward a diverse
    draw: ``temperature`` defaults to 0.6 (DeepSeek's recommended setting for
    R1; reasonable for v3.2) and ``seed`` is set to *sample_idx*. Both can be
    overridden per-sample via ``traces.model.extra_body`` in config.
    """
    traces_cfg = config.get("traces", {})
    model_cfg = traces_cfg.get("model", {})
    model = model_override or model_cfg["name"]
    service_tier = service_tier_override or model_cfg.get("service_tier")

    extra_body = dict(model_cfg.get("extra_body") or {})
    if sample_idx is not None:
        extra_body.setdefault("temperature", 0.6)
        extra_body["seed"] = sample_idx

    return asyncio.run(
        _run_async(
            items=items,
            model_cfg=model_cfg,
            model=model,
            reasoning=model_cfg.get("reasoning", {}),
            service_tier=service_tier,
            extra_body=extra_body or None,
            max_concurrent=traces_cfg.get("max_concurrent", 10),
            max_retries=traces_cfg.get("max_retries", 5),
            retry_min_wait=traces_cfg.get("retry_min_wait", 1.0),
            retry_max_wait=traces_cfg.get("retry_max_wait", 60.0),
            sample_idx=sample_idx,
        )
    )
