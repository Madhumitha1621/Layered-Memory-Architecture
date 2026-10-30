"""OpenAI LLM provider.

Also works with any OpenAI-compatible API (Azure OpenAI, vLLM, Ollama,
Together AI, etc.) — set LLM_BASE_URL to the custom endpoint.
"""

from __future__ import annotations

import logging
from typing import Generator

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, model: str = "", base_url: str = ""):
        from openai import OpenAI  # type: ignore[import-untyped]

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model or self.DEFAULT_MODEL
        logger.info(
            f"OpenAI provider ready (model={self._model}"
            f"{', base_url=' + base_url if base_url else ''})"
        )

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def stream_text_deltas(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
