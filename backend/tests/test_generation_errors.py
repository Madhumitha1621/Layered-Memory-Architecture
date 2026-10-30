"""LLM failures must surface as errors — never be persisted as assistant replies."""

import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from llm import generators
from llm.generators import GENERATION_ERROR_MESSAGE


def _import_main():
    """Import main.py, stubbing the embedding model when torch isn't installed."""
    if "main" in sys.modules:
        return sys.modules["main"]
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("embeddings")
        stub.get_embedding = stub.get_embeddings = stub.get_query_embedding = (
            lambda *a, **k: [0.0] * 4
        )
        sys.modules["embeddings"] = stub
        try:
            import main
        finally:
            # Don't leak the stub into other tests' imports
            if sys.modules.get("embeddings") is stub:
                del sys.modules["embeddings"]
        return main
    import main
    return main


class _Pipeline(SimpleNamespace):
    """Stand-in PipelineResult: unspecified fields read as None."""

    def __getattr__(self, name):
        return None


def _pipeline():
    return _Pipeline(intent="general", confidence=0.9, retrieval_info={},
                     conversation_id="conv-1", query="hi")


def _drain(streaming_response):
    async def collect():
        return [chunk async for chunk in streaming_response.body_iterator]
    return asyncio.run(collect())


# ── generators ────────────────────────────────────────────────────────────

class TestGenerators:
    def test_generate_response_raises_on_llm_error(self):
        with patch.object(generators, "build_messages", return_value=[]), \
             patch.object(generators, "completion", side_effect=RuntimeError("rate limited")):
            with pytest.raises(RuntimeError):
                generators.generate_response("hi")

    def test_stream_emits_error_part_not_text(self):
        def boom(_messages):
            yield "partial"
            raise RuntimeError("provider secret detail")

        with patch.object(generators, "build_messages", return_value=[]), \
             patch.object(generators, "stream_text_deltas", side_effect=boom):
            lines = list(generators.generate_response_stream("hi"))

        assert lines[0] == '0:"partial"\n'
        assert lines[1] == f"3:{json.dumps(GENERATION_ERROR_MESSAGE)}\n"
        assert not any("provider secret detail" in line for line in lines)
        assert lines[-1] == 'd:{"finishReason": "error"}\n'


# ── /chat and /chat/stream ────────────────────────────────────────────────

class TestChatEndpoints:
    @pytest.fixture
    def main(self):
        return _import_main()

    def test_chat_returns_502_and_skips_persist_on_error(self, main):
        with patch.object(main, "run_pipeline", return_value=_pipeline()), \
             patch.object(main, "generate_response", side_effect=RuntimeError("down")), \
             patch.object(main, "persist_after_response") as persist:
            with pytest.raises(HTTPException) as exc:
                main.chat(main.ChatRequest(user_query="hi"))

        assert exc.value.status_code == 502
        assert "down" not in exc.value.detail
        persist.assert_not_called()

    def test_chat_returns_502_on_empty_response(self, main):
        with patch.object(main, "run_pipeline", return_value=_pipeline()), \
             patch.object(main, "generate_response", return_value=None), \
             patch.object(main, "persist_after_response") as persist:
            with pytest.raises(HTTPException) as exc:
                main.chat(main.ChatRequest(user_query="hi"))

        assert exc.value.status_code == 502
        persist.assert_not_called()

    def test_stream_skips_persist_on_error(self, main):
        chunks = ['0:"partial"\n', f"3:{json.dumps(GENERATION_ERROR_MESSAGE)}\n",
                  'e:{"finishReason": "error"}\n', 'd:{"finishReason": "error"}\n']
        with patch.object(main, "run_pipeline", return_value=_pipeline()), \
             patch.object(main, "generate_response_stream", return_value=iter(chunks)), \
             patch.object(main, "persist_after_response") as persist:
            out = _drain(main.chat_stream(main.ChatRequest(user_query="hi")))

        assert any(line.startswith("3:") for line in out)
        persist.assert_not_called()

    def test_stream_persists_successful_reply(self, main):
        chunks = ['0:"Hello"\n', '0:" world"\n',
                  'e:{"finishReason": "stop"}\n', 'd:{"finishReason": "stop"}\n']
        p = _pipeline()
        with patch.object(main, "run_pipeline", return_value=p), \
             patch.object(main, "generate_response_stream", return_value=iter(chunks)), \
             patch.object(main, "persist_after_response") as persist:
            _drain(main.chat_stream(main.ChatRequest(user_query="hi")))

        persist.assert_called_once_with(p, "Hello world")
