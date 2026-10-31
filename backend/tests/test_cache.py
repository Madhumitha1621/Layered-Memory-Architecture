"""Tests for cache.py — intent classification cache keys (no Redis needed)."""

import importlib.util
from pathlib import Path

import pytest

# test_classifier.py installs a stub "cache" module; load the real file directly.
_spec = importlib.util.spec_from_file_location(
    "cache_under_test", Path(__file__).resolve().parent.parent / "cache.py"
)
cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache)


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, _ttl, value):
        self.store[key] = value


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    monkeypatch.setattr(cache, "_initialized", True)
    monkeypatch.setattr(cache, "_redis", _FakeRedis())


def test_same_query_different_context_does_not_collide():
    ctx = [{"role": "user", "content": "tell me about pgvector"},
           {"role": "assistant", "content": "pgvector is..."}]
    cache.set_classification("why?", {"intent": "continuation"}, ctx)

    assert cache.get_classification("why?", ctx) == {"intent": "continuation"}
    assert cache.get_classification("why?", None) is None


def test_long_queries_sharing_a_prefix_do_not_collide():
    prefix = "x" * 250
    cache.set_classification(prefix + " about my profile", {"intent": "profile"})

    assert cache.get_classification(prefix + " about the docs") is None
    assert cache.get_classification(prefix + " about my profile") == {"intent": "profile"}


def test_query_normalization_still_applies():
    cache.set_classification("  Hello There ", {"intent": "general"})
    assert cache.get_classification("hello there") == {"intent": "general"}
