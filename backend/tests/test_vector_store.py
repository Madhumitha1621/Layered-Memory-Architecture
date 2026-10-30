"""Tests for vector_store.search_with_scores — score normalization + thresholding."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import vector_store


@pytest.fixture
def db_backend(monkeypatch):
    """Pretend pgvector is available and stub the embedding model."""
    monkeypatch.setattr(vector_store, "_db_available", True)
    fake_embeddings = types.ModuleType("embeddings")
    fake_embeddings.get_query_embedding = lambda q: [0.0] * 4
    with patch.dict(sys.modules, {"embeddings": fake_embeddings}):
        yield


class TestHybridMinSimilarity:
    def test_bm25_only_hits_below_threshold_are_dropped(self, db_backend):
        import query_db

        # Hybrid returns RRF scores; "keyword-only" has no vector match.
        fused = [("relevant chunk", 0.03), ("keyword-only chunk", 0.02)]
        with patch("hybrid_search.hybrid_search", return_value=fused), \
             patch.object(query_db, "get_connection", return_value=MagicMock()), \
             patch.object(query_db, "put_connection"), \
             patch.object(query_db, "search_document_chunks_with_scores",
                          return_value=[("relevant chunk", 0.8)]):
            results = vector_store.search_with_scores(
                "query", k=4, min_similarity=0.45, use_hybrid=True, use_reranker=False,
            )

        assert results == [("relevant chunk", 0.8)]

    def test_zero_threshold_keeps_all_hits(self, db_backend):
        import query_db

        fused = [("relevant chunk", 0.03), ("keyword-only chunk", 0.02)]
        with patch("hybrid_search.hybrid_search", return_value=fused), \
             patch.object(query_db, "get_connection", return_value=MagicMock()), \
             patch.object(query_db, "put_connection"), \
             patch.object(query_db, "search_document_chunks_with_scores",
                          return_value=[("relevant chunk", 0.8)]):
            results = vector_store.search_with_scores(
                "query", k=4, min_similarity=0.0, use_hybrid=True, use_reranker=False,
            )

        assert [text for text, _ in results] == ["relevant chunk", "keyword-only chunk"]
