"""Optional Redis cache — no-op when disabled.

Enable: set ENABLE_CACHE=true and REDIS_URL in .env.
When Redis is unavailable, everything still works — zero impact.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_redis = None
_initialized = False


def _init():
    """Lazy Redis init — only connects if ENABLE_CACHE=true."""
    global _redis, _initialized
    if _initialized:
        return
    _initialized = True
    try:
        from settings import settings

        if not settings.ENABLE_CACHE:
            return
        import redis as _redis_lib  # type: ignore[import-untyped]

        _redis = _redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
        _redis.ping()
        logger.info("Redis cache connected")
    except Exception as e:
        logger.warning(f"Redis not available (caching disabled): {e}")
        _redis = None


def _key(*parts: str) -> str:
    raw = ":".join(str(p) for p in parts)
    return f"rag:{hashlib.md5(raw.encode()).hexdigest()}"


# ── Generic get / set ─────────────────────────────────────────────────────

def get(key: str):
    """Get a cached value.  Returns None on miss or if cache disabled."""
    _init()
    if not _redis:
        return None
    try:
        val = _redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def put(key: str, value, ttl: int | None = None):
    """Store a value in cache.  No-op if cache disabled."""
    _init()
    if not _redis:
        return
    try:
        from settings import settings

        _redis.setex(key, ttl or settings.CACHE_TTL, json.dumps(value))
    except Exception:
        pass


# ── Intent classification cache ───────────────────────────────────────────

_CLASSIFICATION_TTL = 1800  # 30 minutes


def _classification_key(query: str, context: list | None) -> str:
    # The LLM classifies with the recent conversation as context (e.g. a short
    # follow-up is "continuation" only mid-conversation), so the context is
    # part of the key.  The full query is hashed: truncating it made distinct
    # long queries with a shared prefix collide.
    recent = [
        (m.get("role", ""), m.get("content", ""))
        for m in (context or [])[-6:]
        if isinstance(m, dict)
    ]
    return _key("intent", query.strip().lower(), json.dumps(recent))


def get_classification(query: str, context: list | None = None) -> dict | None:
    """Get cached intent classification result for this query + context."""
    return get(_classification_key(query, context))


def set_classification(query: str, result: dict, context: list | None = None) -> None:
    """Cache intent classification (30 min TTL)."""
    put(_classification_key(query, context), result, ttl=_CLASSIFICATION_TTL)


# ── Embedding cache ───────────────────────────────────────────────────────

def get_embedding(text: str) -> list | None:
    """Get cached embedding vector.  Returns list[float] or None."""
    return get(_key("emb", text[:200]))


def set_embedding(text: str, vector: Any) -> None:
    """Cache embedding vector."""
    vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    put(_key("emb", text[:200]), vec_list)
