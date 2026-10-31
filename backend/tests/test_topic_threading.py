"""Tests for the topic threading engine."""

import numpy as np
import pytest

from topic_threading import (
    ThreadResolution,
    _ema_centroid,
    cosine_similarity,
)


# ═══════════════════════════════════════════════════════════════════════════
#  EMA CENTROID
# ═══════════════════════════════════════════════════════════════════════════

class TestEmaCentroid:
    """Centroid update mathematics."""

    def test_first_message_returns_embedding(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _ema_centroid(np.zeros(3), emb, message_count=1)
        np.testing.assert_array_almost_equal(result, emb)

    def test_second_message_averages(self):
        old = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = _ema_centroid(old, new, message_count=2)
        # With count=2, uses simple average (weight=0.5), then L2-normalizes
        raw = old * 0.5 + new * 0.5
        expected = raw / np.linalg.norm(raw)
        np.testing.assert_array_almost_equal(result, expected)

    def test_ema_with_high_count_uses_alpha(self):
        old = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        alpha = 0.3
        result = _ema_centroid(old, new, message_count=5, alpha=alpha)
        expected_raw = old * (1 - alpha) + new * alpha
        expected = expected_raw / np.linalg.norm(expected_raw)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_output_is_normalized(self):
        old = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        new = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        result = _ema_centroid(old, new, message_count=10)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Centroid not normalized: norm={norm}"


# ═══════════════════════════════════════════════════════════════════════════
#  COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    """Cosine similarity computation."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-5

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert cosine_similarity(a, b) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD RESOLUTION (with mocked DB)
# ═══════════════════════════════════════════════════════════════════════════

class TestThreadResolution:
    """Thread resolution with DB disabled (in-memory only)."""

    def test_disabled_returns_empty(self):
        from topic_threading import resolve_thread

        emb = np.random.randn(768).astype(np.float32)
        result = resolve_thread("conv-1", emb, db_enabled=False)
        assert result.thread_id == ""
        assert not result.is_new

    def test_resolution_dataclass_defaults(self):
        r = ThreadResolution(thread_id="abc-123")
        assert r.thread_id == "abc-123"
        assert r.is_new is False
        assert r.similarity == 0.0
        assert r.thread_summary == ""
        assert r.thread_label == ""
        assert r.message_count == 0

    def test_resolution_with_all_fields(self):
        r = ThreadResolution(
            thread_id="t1",
            is_new=True,
            similarity=0.85,
            thread_summary="Discussion about deployment",
            thread_label="Deployment Strategy",
            message_count=5,
        )
        assert r.is_new
        assert r.similarity == 0.85
        assert r.message_count == 5
