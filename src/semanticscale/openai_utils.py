"""Shared helpers for OpenAI Responses API calls."""

from typing import Any
import json

import openai
import pydantic_core
import tenacity


def should_retry_openai_exception(exc: BaseException) -> bool:
    """Return whether an OpenAI exception should be retried."""
    if isinstance(exc, json.decoder.JSONDecodeError):
        return True
    if isinstance(exc, pydantic_core.ValidationError):
        return True
    if isinstance(exc, ValueError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def extract_response_text(response: Any) -> tuple[str, str]:
    """Extract (reasoning_text, answer_text) from an OpenAI Responses API response."""
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []

    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            summaries = getattr(item, "summary", []) or []
            for summary in summaries:
                text = getattr(summary, "text", None) or (
                    summary if isinstance(summary, str) else ""
                )
                if text:
                    reasoning_parts.append(text)
        elif item_type == "message":
            content = getattr(item, "content", []) or []
            for block in content:
                if getattr(block, "type", None) == "output_text":
                    text = getattr(block, "text", "")
                    if text:
                        answer_parts.append(text)

    return "\n\n".join(reasoning_parts), "\n\n".join(answer_parts)


def extract_usage(response: Any) -> dict | None:
    """Return a serializable usage summary from a Responses API response."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return None

    usage = {
        "input_tokens": getattr(usage_obj, "input_tokens", None),
        "output_tokens": getattr(usage_obj, "output_tokens", None),
    }
    details = getattr(usage_obj, "output_tokens_details", None)
    if details is not None:
        usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return usage


def extract_finish_reason(response: Any) -> str | None:
    """Map Responses-API status to a finish_reason-style string.

    The Responses API exposes ``status`` (``completed`` / ``incomplete`` / ...)
    and ``incomplete_details.reason`` (e.g. ``max_output_tokens``,
    ``content_filter``). For "completed" we return ``"stop"`` to match Chat
    Completions semantics; for "incomplete" we return the underlying reason
    if available, or ``"incomplete"`` otherwise.
    """
    status = getattr(response, "status", None)
    if status == "completed":
        return "stop"
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        return reason or "incomplete"
    return status


async def create_response(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    reasoning: dict,
    service_tier: str | None,
    extra_body: dict | None = None,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> Any:
    """Call the OpenAI Responses API with retry for retryable failures."""

    @tenacity.retry(
        retry=tenacity.retry_if_exception(should_retry_openai_exception),
        wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
        stop=tenacity.stop_after_attempt(max_retries),
        reraise=True,
    )
    async def _call() -> Any:
        kwargs: dict = {}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if service_tier is not None:
            kwargs["service_tier"] = service_tier

        return await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            reasoning=reasoning,
            **kwargs,
        )

    return await _call()
