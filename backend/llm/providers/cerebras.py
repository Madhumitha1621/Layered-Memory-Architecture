"""Cerebras LLM provider."""

from __future__ import annotations

import logging
from typing import Generator

from .base import LLMProvider

logger = logging.getLogger(__name__)


class CerebrasProvider(LLMProvider):
    """Cerebras Cloud SDK — fast inference, OpenAI-compatible API."""

    DEFAULT_MODEL = "gpt-oss-120b"

    def __init__(self, api_key: str, model: str = ""):
        from cerebras.cloud.sdk import Cerebras

        self._client = Cerebras(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL
        logger.info(f"Cerebras provider ready (model={self._model})")

    @property
    def name(self) -> str:
        return "cerebras"

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
        return response.choices[0].message.content  # type: ignore[union-attr]

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
            if not chunk.choices:  # type: ignore[union-attr]
                continue
            delta = chunk.choices[0].delta  # type: ignore[union-attr]
            if delta and delta.content:
                yield delta.content  # type: ignore[misc]
