"""/chat/stream must not echo internal exception text to clients."""

import sys
import types
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _import_main():
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
            if sys.modules.get("embeddings") is stub:
                del sys.modules["embeddings"]
        return main
    import main
    return main


def test_pipeline_error_detail_is_generic():
    main = _import_main()
    secret = 'connection to server at "db.internal" failed for user "root"'
    with patch.object(main, "run_pipeline", side_effect=RuntimeError(secret)):
        with pytest.raises(HTTPException) as exc:
            main.chat_stream(main.ChatRequest(user_query="hi"))

    assert exc.value.status_code == 500
    assert "db.internal" not in exc.value.detail
