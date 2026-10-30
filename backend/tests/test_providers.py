"""Tests for LLM provider streaming — robustness to provider chunk shapes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm.providers.cerebras import CerebrasProvider
from llm.providers.openai import OpenAIProvider


def _chunk(content=None, *, empty=False):
    if empty:
        return SimpleNamespace(choices=[])
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


@pytest.mark.parametrize("provider_cls", [OpenAIProvider, CerebrasProvider])
def test_stream_skips_chunks_without_choices(provider_cls):
    # Bypass __init__ so the vendor SDK doesn't need to be installed.
    provider = provider_cls.__new__(provider_cls)
    provider._model = "test-model"
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = iter([
        _chunk(empty=True),  # Azure OpenAI sends a prompt-filter chunk first
        _chunk("Hello"),
        _chunk(None),
        _chunk(" world"),
    ])

    out = list(provider.stream_text_deltas([{"role": "user", "content": "hi"}]))

    assert out == ["Hello", " world"]
