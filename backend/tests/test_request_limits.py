"""Request validation bounds on client-controlled sizes (no DB / LLM needed)."""

import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    if "main" not in sys.modules:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            # main imports the embedding model at module load; stub it for this import only
            stub = types.ModuleType("embeddings")
            stub.get_embedding = stub.get_embeddings = stub.get_query_embedding = (
                lambda *a, **k: [0.0] * 4
            )
            sys.modules["embeddings"] = stub
            try:
                import main  # noqa: F401
            finally:
                if sys.modules.get("embeddings") is stub:
                    del sys.modules["embeddings"]
    import main
    # No `with`: skip the lifespan (DB init / ingestion); validation runs first anyway
    return TestClient(main.app)


@pytest.mark.parametrize("method,url,body", [
    ("post", "/chat", {"user_query": ""}),
    ("post", "/chat", {"user_query": "x" * 16_001}),
    ("post", "/retrieval/test", {"query": "q", "k": 1000}),
    ("post", "/retrieval/test", {"query": "q", "min_similarity": 2}),
    ("get", "/conversations?limit=100000", None),
    ("get", "/conversations/search?q=a&limit=0", None),
    ("get", "/conversations/c1/messages?limit=5000", None),
    ("get", "/conversations/c1/insights?limit=-1", None),
    ("get", "/concepts/search?q=a&k=1000", None),
    ("get", "/insights/search?q=a&k=1000", None),
])
def test_out_of_range_inputs_are_rejected(client, method, url, body):
    resp = getattr(client, method)(url, json=body) if body is not None else getattr(client, method)(url)
    assert resp.status_code == 422


def test_defaults_used_by_frontend_are_accepted(client):
    # Frontend calls these with limit=50 / limit=200; must stay within bounds.
    assert client.get("/conversations?limit=50").status_code != 422
    assert client.get("/conversations/c1/messages?limit=200").status_code != 422
