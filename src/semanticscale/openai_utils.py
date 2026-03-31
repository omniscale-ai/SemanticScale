"""Shared helpers for OpenAI Responses API calls."""

from typing import Any

import openai
import tenacity


def should_retry_openai_exception(exc: BaseException) -> bool:
    """Return whether an OpenAI exception should be retried."""
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


async def create_response(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    reasoning_effort: str,
    summary: str,
    service_tier: str,
    max_output_tokens: int,
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
        return await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            reasoning={
                "effort": reasoning_effort,
                "summary": summary,
            },
            service_tier=service_tier,
            max_output_tokens=max_output_tokens,
        )

    return await _call()
