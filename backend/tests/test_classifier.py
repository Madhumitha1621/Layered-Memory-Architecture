"""Tests for intent classification — pre-heuristics and JSON extraction logic.

These tests cover all the fast-path branches that run without an LLM call,
plus the JSON extraction fallback logic.  LLM calls are mocked so the tests
run offline.
"""

import json
from unittest.mock import patch, MagicMock
import pytest

# Patch the cache module and the llm.client before importing classifier
# so we never need a running Redis or LLM provider.
import sys
import types

# Stub out the cache module
cache_stub = types.ModuleType("cache")
cache_stub.get_classification = lambda q: None
cache_stub.set_classification = lambda q, r: None
sys.modules.setdefault("cache", cache_stub)

from llm.classifier import (
    classify_intent,
    VALID_INTENTS,
    PRIVACY_SIGNALS,
    CONTINUATION_PRONOUNS,
    CONTINUATION_SIGNALS,
)


# ─── Greeting fast-path ───────────────────────────────────────────────────

class TestGreetingFastPath:
    """Greetings must map to 'general' without hitting the LLM."""

    @pytest.mark.parametrize("query", [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "Hey there",
        "Hello there",
        "hi there",
    ])
    def test_common_greetings(self, query):
        result = classify_intent(query)
        assert result["intent"] == "general"
        assert result["confidence"] >= 0.9

    def test_greeting_with_exclamation(self):
        result = classify_intent("hello!")
        assert result["intent"] == "general"

    def test_greeting_with_comma(self):
        result = classify_intent("hey, how are you")
        assert result["intent"] == "general"

    def test_long_greeting_falls_through(self):
        """Greetings longer than 8 words must NOT be fast-pathed."""
        # Should reach LLM (which we mock to return 'general').
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"general","confidence":0.8}'
            result = classify_intent(
                "hello there my good friend how are you doing today"
            )
        # Whether it's a greeting or general both are acceptable here;
        # the key assertion is that .completion was called.
        mock_completion.assert_called_once()
        assert result["source"] == "llm"


# ─── Profile statement fast-path ─────────────────────────────────────────

class TestProfileStatementFastPath:
    """Profile openers without '?' must map to 'profile'."""

    @pytest.mark.parametrize("query", [
        "my name is Alice",
        "I am a software engineer",
        "I'm a backend developer",
        "I like Python",
        "I work at a startup",
        "Call me Bob",
        "I speak Spanish",
        "I graduated from MIT",
    ])
    def test_profile_statements(self, query):
        result = classify_intent(query)
        assert result["intent"] == "profile"
        assert result["confidence"] >= 0.85

    def test_profile_opener_with_question_hits_llm(self):
        """'I am confused?' has a '?' so it should NOT be fast-pathed."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"general","confidence":0.8}'
            classify_intent("I am confused?")
        mock_completion.assert_called_once()


# ─── Privacy fast-path ────────────────────────────────────────────────────

class TestPrivacyFastPath:
    """Privacy signal phrases must map to 'privacy'."""

    @pytest.mark.parametrize("signal", [
        "do you store my messages",
        "what data do you have on me",
        "delete my data please",
        "what do you know about me",
        "is my data safe",
        "do you track my activity",
        "what have you stored about me",
        "data about me",
    ])
    def test_privacy_signals(self, signal):
        result = classify_intent(signal)
        assert result["intent"] == "privacy"
        assert result["confidence"] >= 0.9


# ─── Continuation fast-path ───────────────────────────────────────────────

class TestContinuationFastPath:
    """Short pronoun questions with conversation context → continuation."""

    def _ctx(self):
        return [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language..."},
        ]

    def test_continuation_pronoun_question(self):
        result = classify_intent("what is it used for?", conversation_context=self._ctx())
        assert result["intent"] == "continuation"
        assert result["confidence"] >= 0.8

    def test_continuation_no_context(self):
        """Without conversation context, can't be a continuation."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"general","confidence":0.7}'
            result = classify_intent("what is it used for?")
        # No continuation without context
        assert result["intent"] != "continuation"

    def test_continuation_no_question_mark(self):
        """A pronoun without '?' should not be fast-pathed as continuation."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"general","confidence":0.7}'
            classify_intent("tell me more about it", conversation_context=self._ctx())
        mock_completion.assert_called_once()


# ─── LLM fallback & JSON extraction ──────────────────────────────────────

class TestLLMFallback:
    """Queries that miss all heuristics fall back to the LLM."""

    def test_plain_json_response(self):
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"knowledge_base","confidence":0.88}'
            result = classify_intent("What is pgvector?")
        assert result["intent"] == "knowledge_base"
        assert result["confidence"] == pytest.approx(0.88)

    def test_json_wrapped_in_code_fence(self):
        """Classifier must strip markdown code fences."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = (
                "```json\n{\"intent\":\"general\",\"confidence\":0.75}\n```"
            )
            result = classify_intent("How does this work?")
        assert result["intent"] == "general"

    def test_json_inside_prose(self):
        """Classifier must find JSON even when LLM adds prose around it."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = (
                "Sure, the intent is: {\"intent\":\"general\",\"confidence\":0.8} — good luck!"
            )
            result = classify_intent("How does this work?")
        assert result["intent"] == "general"

    def test_unknown_intent_falls_back_to_general(self):
        """An unrecognised intent label must be coerced to 'general'."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"banana","confidence":0.9}'
            result = classify_intent("Something odd?")
        assert result["intent"] == "general"

    def test_llm_exception_returns_safe_fallback(self):
        """Any LLM error must return the safe fallback, not raise."""
        with patch("llm.classifier.completion", side_effect=RuntimeError("boom")):
            result = classify_intent("What do you think?")
        assert result["intent"] == "general"
        assert result["confidence"] == 0.5


# ─── VALID_INTENTS completeness ───────────────────────────────────────────

def test_valid_intents_set():
    """Pipeline relies on exactly these 5 labels."""
    assert VALID_INTENTS == {"general", "continuation", "knowledge_base", "profile", "privacy"}


# ─── Continuation signal fast-path ────────────────────────────────────────

class TestContinuationSignalsFastPath:
    """Short continuation signals with context should be fast-pathed."""

    def _ctx(self):
        return [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language..."},
        ]

    def test_why_question(self):
        result = classify_intent("Why?", conversation_context=self._ctx())
        assert result["intent"] == "continuation"
        assert result["confidence"] >= 0.8

    def test_elaborate(self):
        result = classify_intent("Elaborate", conversation_context=self._ctx())
        assert result["intent"] == "continuation"
        assert result["confidence"] >= 0.8

    def test_more_details(self):
        result = classify_intent("More details", conversation_context=self._ctx())
        assert result["intent"] == "continuation"
        assert result["confidence"] >= 0.8

    def test_longer_signal_falls_through(self):
        """Longer messages with signals should NOT be fast-pathed."""
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"continuation","confidence":0.8}'
            classify_intent("I need more details about this", conversation_context=self._ctx())
        mock_completion.assert_called_once()


# ─── Profile opener protection ────────────────────────────────────────────

class TestProfileOpenerProtection:
    """Long messages with profile openers should fall through to the LLM."""

    def test_long_opener_not_fast_pathed(self):
        with patch("llm.classifier.completion") as mock_completion:
            mock_completion.return_value = '{"intent":"knowledge_base","confidence":0.85}'
            result = classify_intent(
                "I have a really important question about machine learning transformers "
                "and how they process attention mechanisms in parallel"
            )
        mock_completion.assert_called_once()
        assert result["intent"] == "knowledge_base"
