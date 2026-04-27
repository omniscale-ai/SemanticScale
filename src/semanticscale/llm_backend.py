"""LLM backend abstraction for SH6.

Routes between three implementations with a common async interface:

* :class:`OpenAIBackend` — uses the OpenAI Responses API
  (``client.responses.create`` / ``.parse``).

* :class:`OpenRouterBackend` — uses the OpenRouter Python SDK
  (``client.chat.send_async``). Selected for configs whose ``model.base_url``
  points at ``openrouter.ai`` or whose ``api_key_env`` is
  ``OPENROUTER_API_KEY``.

* :class:`LocalChatBackend` — OpenAI-compatible Chat Completions for a
  locally-served model (vLLM, Ollama, llama.cpp). Selected when ``base_url``
  points at localhost or when ``model.backend == "local"``. Reasoning is
  read from ``message.reasoning_content`` if present (vLLM with
  ``--reasoning-parser deepseek_r1``) and otherwise extracted from
  ``<think>…</think>`` tags inside ``message.content``.

All backends expose:

    create(...)  -> {"reasoning_text": str, "answer_text": str, "usage": dict | None}
    parse(...)   -> pydantic instance of ``text_format``
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

import openai
import openrouter
import pydantic_core
import tenacity
from openrouter.errors import OpenRouterError
from pydantic import BaseModel

from semanticscale.openai_utils import (
    create_response,
    extract_response_text,
    extract_usage,
    should_retry_openai_exception,
)

logger = logging.getLogger(__name__)


def is_openrouter(model_cfg: dict) -> bool:
    """Return True when the model config targets OpenRouter."""
    base_url = (model_cfg or {}).get("base_url") or ""
    if "openrouter.ai" in base_url:
        return True
    return ((model_cfg or {}).get("api_key_env") or "") == "OPENROUTER_API_KEY"


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def is_local(model_cfg: dict) -> bool:
    """Return True when the model config targets a locally-served endpoint."""
    cfg = model_cfg or {}
    if (cfg.get("backend") or "").lower() == "local":
        return True
    base_url = cfg.get("base_url") or ""
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return host.lower() in _LOCAL_HOSTS


_RETRYABLE_OR_STATUS = {408, 429, 500, 502, 503, 504, 524, 529}


def should_retry_openrouter_exception(exc: BaseException) -> bool:
    """Retry policy for OpenRouter SDK calls and structured-output parsing."""
    if isinstance(exc, (json.decoder.JSONDecodeError, pydantic_core.ValidationError, ValueError)):
        return True
    if isinstance(exc, OpenRouterError):
        return getattr(exc, "status_code", 0) in _RETRYABLE_OR_STATUS or exc.status_code >= 500
    return False


# ---------------------------------------------------------------------------
# OpenAI (Responses API) backend
# ---------------------------------------------------------------------------


class OpenAIBackend:
    def __init__(self, model_cfg: dict) -> None:
        api_key_env = model_cfg.get("api_key_env") or "OPENAI_API_KEY"
        self._client = openai.AsyncOpenAI(
            base_url=model_cfg.get("base_url"),
            api_key=os.environ.get(api_key_env),
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def create(
        self,
        *,
        model: str,
        prompt: str,
        reasoning: dict,
        service_tier: str | None,
        extra_body: dict | None,
        max_retries: int,
        retry_min_wait: float,
        retry_max_wait: float,
    ) -> dict:
        response = await create_response(
            client=self._client,
            model=model,
            prompt=prompt,
            reasoning=reasoning,
            service_tier=service_tier,
            extra_body=extra_body,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
        )
        reasoning_text, answer_text = extract_response_text(response)
        return {
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "usage": extract_usage(response),
        }

    async def parse(
        self,
        *,
        model: str,
        messages: list[dict],
        text_format: type[BaseModel],
        service_tier: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
    ) -> BaseModel:
        @tenacity.retry(
            retry=tenacity.retry_if_exception(should_retry_openai_exception),
            wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
            stop=tenacity.stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _call() -> BaseModel:
            kwargs: dict[str, Any] = {}
            if service_tier is not None:
                kwargs["service_tier"] = service_tier
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = await self._client.responses.parse(
                model=model,
                input=messages,
                text_format=text_format,
                **kwargs,
            )
            if response.output_parsed is None:
                raise ValueError("LLM returned unparseable output (output_parsed is None)")
            return response.output_parsed

        return await _call()


# ---------------------------------------------------------------------------
# OpenRouter (Chat Completions) backend
# ---------------------------------------------------------------------------


class OpenRouterBackend:
    def __init__(self, model_cfg: dict) -> None:
        api_key_env = model_cfg.get("api_key_env") or "OPENROUTER_API_KEY"
        api_key = os.environ.get(api_key_env)
        self._client = openrouter.OpenRouter(api_key=api_key)

    async def aclose(self) -> None:
        try:
            async_client = self._client.sdk_configuration.async_client
            if async_client is not None:
                await async_client.aclose()
                self._client.sdk_configuration.async_client = None
        except Exception:
            logger.debug("OpenRouter client aclose() failed", exc_info=True)

    @staticmethod
    def _reasoning_arg(reasoning: dict | None) -> dict | None:
        """Map the project's reasoning dict to OpenRouter's ``Reasoning`` shape.

        OpenRouter's typed ``Reasoning`` only exposes ``effort`` and ``summary``;
        a bare ``{"enabled": true}`` gets dropped by pydantic and yields zero
        reasoning tokens. When ``enabled: true`` is set without an explicit
        effort, default to ``medium`` so reasoning traces are actually
        generated. Returns ``None`` to omit the field entirely.
        """
        if not reasoning or reasoning.get("enabled") is False:
            return None
        out: dict[str, Any] = {}
        if "effort" in reasoning and reasoning["effort"] is not None:
            out["effort"] = reasoning["effort"]
        if "summary" in reasoning and reasoning["summary"] is not None:
            out["summary"] = reasoning["summary"]
        if not out and reasoning.get("enabled") is True:
            out["effort"] = "medium"
        return out or None

    @staticmethod
    def _extract_text(response: Any) -> tuple[str, str]:
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError):
            return "", ""
        reasoning_text = getattr(message, "reasoning", None) or ""
        content = getattr(message, "content", None)
        answer_text = content if isinstance(content, str) else ""
        return reasoning_text, answer_text

    @staticmethod
    def _extract_usage(response: Any) -> dict | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        out: dict[str, Any] = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            out["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
        return out

    async def create(
        self,
        *,
        model: str,
        prompt: str,
        reasoning: dict,
        service_tier: str | None,
        extra_body: dict | None,
        max_retries: int,
        retry_min_wait: float,
        retry_max_wait: float,
    ) -> dict:
        messages = [{"role": "user", "content": prompt}]
        reasoning_arg = self._reasoning_arg(reasoning)

        @tenacity.retry(
            retry=tenacity.retry_if_exception(should_retry_openrouter_exception),
            wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
            stop=tenacity.stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model,
                "stream": False,
            }
            if reasoning_arg is not None:
                kwargs["reasoning"] = reasoning_arg
            if service_tier is not None:
                kwargs["service_tier"] = service_tier
            return await self._client.chat.send_async(**kwargs)

        response = await _call()
        reasoning_text, answer_text = self._extract_text(response)
        return {
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "usage": self._extract_usage(response),
        }

    async def parse(
        self,
        *,
        model: str,
        messages: list[dict],
        text_format: type[BaseModel],
        service_tier: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
    ) -> BaseModel:
        schema = text_format.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": text_format.__name__,
                "schema": schema,
            },
        }

        @tenacity.retry(
            retry=tenacity.retry_if_exception(should_retry_openrouter_exception),
            wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
            stop=tenacity.stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _call() -> BaseModel:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model,
                "stream": False,
                "response_format": response_format,
            }
            if service_tier is not None:
                kwargs["service_tier"] = service_tier
            response = await self._client.chat.send_async(**kwargs)
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError) as exc:
                raise ValueError(f"OpenRouter response missing choices/message: {exc}") from exc
            if not content or not isinstance(content, str):
                raise ValueError("OpenRouter response has empty content for structured output")
            return text_format.model_validate_json(content)

        return await _call()


# ---------------------------------------------------------------------------
# Local OpenAI-compatible Chat Completions backend (vLLM, Ollama, llama.cpp)
# ---------------------------------------------------------------------------


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>(.*)\Z", re.DOTALL)


def _split_think_tags(content: str) -> tuple[str, str]:
    """Split DeepSeek-R1-style ``<think>…</think>`` reasoning out of content.

    Returns ``(reasoning, answer)``. If no closing tag is present (e.g. the
    response was cut off by max_tokens), the whole ``<think>``-onward span is
    treated as reasoning and the answer is empty.
    """
    if not content:
        return "", ""
    blocks = _THINK_BLOCK_RE.findall(content)
    if blocks:
        reasoning = "\n\n".join(b.strip() for b in blocks if b.strip())
        answer = _THINK_BLOCK_RE.sub("", content).strip()
        return reasoning, answer
    open_match = _OPEN_THINK_RE.search(content)
    if open_match:
        return open_match.group(1).strip(), ""
    return "", content.strip()


class LocalChatBackend:
    """OpenAI-compatible Chat Completions client for a locally-served LLM."""

    def __init__(self, model_cfg: dict) -> None:
        api_key_env = model_cfg.get("api_key_env") or "LOCAL_LLM_API_KEY"
        api_key = os.environ.get(api_key_env) or "EMPTY"
        self._client = openai.AsyncOpenAI(
            base_url=model_cfg.get("base_url"),
            api_key=api_key,
        )

    async def aclose(self) -> None:
        await self._client.close()

    @staticmethod
    def _extract_text(response: Any) -> tuple[str, str]:
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError):
            return "", ""
        reasoning_text = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        content = getattr(message, "content", None) or ""
        if not isinstance(content, str):
            content = ""
        if reasoning_text:
            return reasoning_text, content.strip()
        return _split_think_tags(content)

    @staticmethod
    def _extract_usage(response: Any) -> dict | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        out: dict[str, Any] = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            out["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
        return out

    async def create(
        self,
        *,
        model: str,
        prompt: str,
        reasoning: dict,
        service_tier: str | None,
        extra_body: dict | None,
        max_retries: int,
        retry_min_wait: float,
        retry_max_wait: float,
    ) -> dict:
        messages: list[dict] = []
        if reasoning and reasoning.get("enabled") is False:
            messages.append({
                "role": "system",
                "content": "Answer directly. Do not produce a <think> reasoning block.",
            })
        messages.append({"role": "user", "content": prompt})

        @tenacity.retry(
            retry=tenacity.retry_if_exception(should_retry_openai_exception),
            wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
            stop=tenacity.stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            return await self._client.chat.completions.create(**kwargs)

        response = await _call()
        reasoning_text, answer_text = self._extract_text(response)
        return {
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "usage": self._extract_usage(response),
        }

    async def parse(
        self,
        *,
        model: str,
        messages: list[dict],
        text_format: type[BaseModel],
        service_tier: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
    ) -> BaseModel:
        schema = text_format.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": text_format.__name__,
                "schema": schema,
            },
        }

        @tenacity.retry(
            retry=tenacity.retry_if_exception(should_retry_openai_exception),
            wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
            stop=tenacity.stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _call() -> BaseModel:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model,
                "response_format": response_format,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = await self._client.chat.completions.create(**kwargs)
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError) as exc:
                raise ValueError(f"Local response missing choices/message: {exc}") from exc
            if not content or not isinstance(content, str):
                raise ValueError("Local response has empty content for structured output")
            # Some local servers prepend a <think>…</think> block even with
            # JSON-schema guidance; strip it so json.loads sees pure JSON.
            _, answer = _split_think_tags(content)
            payload = answer or content
            return text_format.model_validate_json(payload)

        return await _call()


Backend = OpenAIBackend | OpenRouterBackend | LocalChatBackend


def make_backend(model_cfg: dict) -> Backend:
    """Pick the right backend based on the model config."""
    if is_local(model_cfg):
        return LocalChatBackend(model_cfg)
    if is_openrouter(model_cfg):
        return OpenRouterBackend(model_cfg)
    return OpenAIBackend(model_cfg)
