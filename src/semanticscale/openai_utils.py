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
    """Return a serializable usage summary from a Responses API or Completions API response."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return None

    usage = {
        "input_tokens": getattr(usage_obj, "prompt_tokens", getattr(usage_obj, "input_tokens", None)),
        "output_tokens": getattr(usage_obj, "completion_tokens", getattr(usage_obj, "output_tokens", None)),
    }
    details = getattr(usage_obj, "completion_tokens_details", getattr(usage_obj, "output_tokens_details", None))
    if details is not None:
        usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return usage

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
        kwargs = {}
        if extra_body:
            kwargs["extra_body"] = extra_body
            
        return await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            reasoning=reasoning,
            service_tier=service_tier,
            **kwargs,
        )

    return await _call()


def extract_chat_completion_text(response: Any) -> tuple[str, str]:
    """Extract (reasoning_text, answer_text) from an OpenAI Chat Completions API response."""
    if not hasattr(response, "choices") or not response.choices:
        return "", ""
    
    message = response.choices[0].message
    answer = message.content or ""
    reasoning = ""
    
    # Try to extract explicit reasoning details for OpenRouter and other APIs
    # In Pydantic dict mode or using custom extractors:
    if hasattr(message, "model_extra") and message.model_extra:
        reasoning = message.model_extra.get("reasoning_details", message.model_extra.get("reasoning", ""))
    elif hasattr(message, "reasoning_content") and message.reasoning_content:
        reasoning = message.reasoning_content
    elif isinstance(message, dict):
        reasoning = message.get("reasoning_details", message.get("reasoning", ""))
        
    return str(reasoning), answer


async def create_chat_completion(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    reasoning: dict | str | None,
    service_tier: str | None,
    extra_body: dict | None = None,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> Any:
    """Call the OpenAI Chat Completions API with retry for retryable failures."""

    @tenacity.retry(
        retry=tenacity.retry_if_exception(should_retry_openai_exception),
        wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
        stop=tenacity.stop_after_attempt(max_retries),
        reraise=True,
    )
    async def _call() -> Any:
        kwargs = {}
        if service_tier:
            kwargs["service_tier"] = service_tier
            
        eb = dict(extra_body) if extra_body else {}
        if reasoning:
            if isinstance(reasoning, dict):
                # Distinguish standard "effort" vs arbitrary dict (sent as extra_body for OpenRouter)
                if "effort" in reasoning and not any(k in reasoning for k in ("enabled", "summary")):
                    kwargs["reasoning_effort"] = reasoning["effort"]
                else:
                    eb["reasoning"] = reasoning
            elif isinstance(reasoning, str):
                kwargs["reasoning_effort"] = reasoning
                
        if eb:
            kwargs["extra_body"] = eb

        return await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    return await _call()
