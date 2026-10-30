"""Tests for Settings configuration.

All tests run without a real .env; we use environment variable injection
via monkeypatch so there's no filesystem dependency.
"""

import os
import importlib
import pytest


def _reload_settings(monkeypatch, env_overrides: dict) -> object:
    """Reload settings with specific env vars patched.

    Returns the reloaded `settings` singleton.
    """
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)
    import settings as settings_mod
    importlib.reload(settings_mod)
    return settings_mod.settings


class TestSettingsDefaults:
    def test_default_llm_provider(self):
        from settings import settings
        assert settings.LLM_PROVIDER in ("cerebras", "openai", "anthropic", "")
        # Default is cerebras unless overridden
        # We just check it's a non-empty string from a known set or an empty env
        assert isinstance(settings.LLM_PROVIDER, str)

    def test_token_budget_defaults(self):
        from settings import settings
        assert settings.MAX_RESPONSE_TOKENS == 2048
        assert settings.MAX_CLASSIFIER_TOKENS == 50
        assert settings.MAX_PROFILE_DETECT_TOKENS == 300
        assert settings.MAX_TITLE_TOKENS == 20

    def test_embedding_defaults(self):
        from settings import settings
        assert settings.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert settings.EMBEDDING_DIMENSION == 768
        assert settings.QUERY_INSTRUCTION == ""  # empty = no prefix (bge-v1.5 works without)

    def test_retrieval_defaults(self):
        from settings import settings
        assert settings.RETRIEVAL_K == 4
        assert settings.QA_K == 4
        assert settings.QA_MIN_SIMILARITY == pytest.approx(0.72)

    def test_pipeline_defaults(self):
        from settings import settings
        assert settings.TOPIC_CONTINUATION_THRESHOLD == pytest.approx(0.35)
        assert settings.RECENCY_WINDOW == 6
        assert settings.SEMANTIC_K == 3

    def test_knowledge_base_defaults(self):
        from pathlib import Path
        from settings import settings
        expected_kb = str(Path(__file__).resolve().parent.parent.parent / "knowledge")
        assert settings.KNOWLEDGE_DIR == expected_kb
        assert settings.CHUNK_SIZE == 500
        assert settings.CHUNK_OVERLAP == 50
        assert settings.FORCE_REINDEX is False

    def test_server_defaults(self):
        from settings import settings
        assert settings.PORT == 8000
        assert settings.HOST == "0.0.0.0"
        assert settings.DEBUG_MODE is False
        assert settings.STAGE_STREAMING is True

    def test_cache_disabled_by_default(self):
        from settings import settings
        assert settings.ENABLE_CACHE is False

    def test_db_pool_defaults(self):
        from settings import settings
        assert settings.DB_POOL_MIN == 1
        assert settings.DB_POOL_MAX == 10

    def test_behavior_engine_defaults(self):
        from settings import settings
        assert settings.BEHAVIOR_ENGINE_ENABLED is True
        assert settings.BEHAVIOR_REPETITION_THRESHOLD == pytest.approx(0.7)
        assert settings.BEHAVIOR_PATTERN_WINDOW == 10
        assert settings.BEHAVIOR_STATE_PERSIST is True

    def test_thread_defaults(self):
        from settings import settings
        assert settings.THREAD_ENABLED is True
        assert settings.THREAD_ATTACH_THRESHOLD == pytest.approx(0.55)
        assert settings.THREAD_SUMMARY_INTERVAL == 8
        assert settings.THREAD_MAX_ACTIVE == 12

    def test_research_defaults(self):
        from settings import settings
        assert settings.RESEARCH_INSIGHTS_ENABLED is True
        assert settings.RESEARCH_INSIGHT_MIN_CONFIDENCE == pytest.approx(0.6)
        assert settings.CONCEPT_LINKING_ENABLED is True
        assert settings.CONCEPT_LINK_K == 5


class TestSettingsEnvOverride:
    def test_max_response_tokens_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"MAX_RESPONSE_TOKENS": "4096"})
        assert s.MAX_RESPONSE_TOKENS == 4096

    def test_bool_true_variants(self, monkeypatch):
        for truthy in ("true", "1", "yes", "True", "YES"):
            s = _reload_settings(monkeypatch, {"DEBUG_MODE": truthy})
            assert s.DEBUG_MODE is True

    def test_bool_false_variants(self, monkeypatch):
        for falsy in ("false", "0", "no", "False", "NO"):
            s = _reload_settings(monkeypatch, {"DEBUG_MODE": falsy})
            assert s.DEBUG_MODE is False

    def test_llm_provider_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"LLM_PROVIDER": "anthropic"})
        assert s.LLM_PROVIDER == "anthropic"

    def test_retrieval_k_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"RETRIEVAL_K": "8"})
        assert s.RETRIEVAL_K == 8

    def test_float_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"QA_MIN_SIMILARITY": "0.75"})
        assert s.QA_MIN_SIMILARITY == pytest.approx(0.75)

    def test_behavior_engine_disabled_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"BEHAVIOR_ENGINE_ENABLED": "false"})
        assert s.BEHAVIOR_ENGINE_ENABLED is False

    def test_behavior_repetition_threshold_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"BEHAVIOR_REPETITION_THRESHOLD": "0.5"})
        assert s.BEHAVIOR_REPETITION_THRESHOLD == pytest.approx(0.5)

    def test_behavior_pattern_window_override(self, monkeypatch):
        s = _reload_settings(monkeypatch, {"BEHAVIOR_PATTERN_WINDOW": "20"})
        assert s.BEHAVIOR_PATTERN_WINDOW == 20


class TestSettingsImmutability:
    def test_settings_is_frozen(self):
        from settings import settings
        with pytest.raises((TypeError, AttributeError)):
            settings.PORT = 9999  # type: ignore[misc]
