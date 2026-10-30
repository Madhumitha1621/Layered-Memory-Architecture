"""Tests for conversation_state.py — state tracking + pattern detection."""

import time
from conversation_state import (
    ConversationState,
    StateTracker,
    get_or_create_state,
    set_state,
    clear_state,
    _state_cache,
)


class TestConversationState:
    """ConversationState dataclass tests."""

    def test_default_state(self):
        s = ConversationState()
        assert s.current_topic == ""
        assert s.emotional_tone == "neutral"
        assert s.interaction_pattern == "normal"
        assert s.testing_flag is False
        assert s.repetition_count == 0
        assert s.message_count == 0
        assert s.dynamic_personality_mode == "default"

    def test_to_dict_round_trip(self):
        s = ConversationState(
            current_topic="python",
            emotional_tone="curious",
            message_count=5,
            intent_history=["general", "knowledge_base"],
        )
        d = s.to_dict()
        assert d["current_topic"] == "python"
        assert d["emotional_tone"] == "curious"
        assert d["message_count"] == 5

        restored = ConversationState.from_dict(d)
        assert restored.current_topic == "python"
        assert restored.emotional_tone == "curious"
        assert restored.message_count == 5
        assert restored.intent_history == ["general", "knowledge_base"]

    def test_from_dict_handles_unknown_keys(self):
        """Schema evolution: unknown keys should be silently ignored."""
        d = {"current_topic": "test", "unknown_field": "ignored", "future_feature": 42}
        s = ConversationState.from_dict(d)
        assert s.current_topic == "test"
        assert not hasattr(s, "unknown_field")

    def test_from_dict_empty(self):
        s = ConversationState.from_dict({})
        assert s.message_count == 0

    def test_from_dict_none(self):
        s = ConversationState.from_dict(None)
        assert s.message_count == 0


class TestStateCache:
    """In-memory cache tests."""

    def setup_method(self):
        _state_cache.clear()

    def test_get_or_create(self):
        s = get_or_create_state("conv-1")
        assert s.message_count == 0
        assert s.conversation_start > 0

    def test_get_returns_same_instance(self):
        s1 = get_or_create_state("conv-1")
        s1.message_count = 5
        s2 = get_or_create_state("conv-1")
        assert s2.message_count == 5
        assert s1 is s2

    def test_clear_state(self):
        get_or_create_state("conv-1")
        clear_state("conv-1")
        s = get_or_create_state("conv-1")
        assert s.message_count == 0

    def test_set_state(self):
        s = ConversationState(message_count=10)
        set_state("conv-2", s)
        retrieved = get_or_create_state("conv-2")
        assert retrieved.message_count == 10

    def test_concurrent_eviction_is_safe(self):
        import threading
        import conversation_state

        errors = []

        def worker(n):
            try:
                for i in range(300):
                    get_or_create_state(f"t{n}-{i}")
            except Exception as e:  # pragma: no cover - only on regression
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(_state_cache) <= conversation_state._MAX_CACHE_SIZE


class TestStateTracker:
    """StateTracker update + detection tests."""

    def test_basic_update(self):
        s = ConversationState()
        StateTracker.update(s, "hello there", "general", 0.95)
        assert s.message_count == 1
        assert s.last_intent == "general"
        assert s.intent_streak == 1
        assert s.last_update > 0

    def test_intent_streak(self):
        s = ConversationState()
        StateTracker.update(s, "hello", "general", 0.95)
        StateTracker.update(s, "hey", "general", 0.90)
        StateTracker.update(s, "hi", "general", 0.90)
        assert s.intent_streak == 3

    def test_intent_streak_resets(self):
        s = ConversationState()
        StateTracker.update(s, "hello", "general", 0.95)
        StateTracker.update(s, "hello", "general", 0.90)
        StateTracker.update(s, "what is python", "knowledge_base", 0.90)
        assert s.intent_streak == 1
        assert s.last_intent == "knowledge_base"

    def test_tone_detection_frustrated(self):
        s = ConversationState()
        StateTracker.update(s, "that's not right, you're wrong", "general", 0.5)
        assert s.emotional_tone == "frustrated"

    def test_tone_detection_playful(self):
        s = ConversationState()
        StateTracker.update(s, "haha that's funny lol", "general", 0.5)
        assert s.emotional_tone == "playful"

    def test_tone_detection_curious(self):
        s = ConversationState()
        StateTracker.update(s, "how does this work, can you explain", "general", 0.5)
        assert s.emotional_tone == "curious"

    def test_tone_detection_positive(self):
        s = ConversationState()
        StateTracker.update(s, "thanks, that was awesome!", "general", 0.5)
        assert s.emotional_tone == "positive"

    def test_tone_shift_counting(self):
        s = ConversationState()
        StateTracker.update(s, "hello", "general", 0.5)  # neutral
        StateTracker.update(s, "thanks awesome", "general", 0.5)  # positive
        StateTracker.update(s, "that's wrong", "general", 0.5)  # frustrated
        assert s.tone_shift_count == 2

    def test_testing_detection(self):
        s = ConversationState()
        StateTracker.update(s, "are you an ai?", "general", 0.5)
        assert s.testing_flag is True
        assert s.meta_comment_count == 1

    def test_meta_detection(self):
        s = ConversationState()
        StateTracker.update(s, "you're just a machine", "general", 0.5)
        assert s.meta_comment_count == 1

    def test_repetition_detection(self):
        recent = ["what is python", "what is python", "what is python"]
        count = StateTracker._detect_repetition("what is python", recent, threshold=0.7)
        assert count >= 2

    def test_repetition_no_match(self):
        recent = ["what is python", "how does javascript work"]
        count = StateTracker._detect_repetition("tell me about databases", recent, threshold=0.7)
        assert count == 0

    def test_pattern_classification_testing(self):
        s = ConversationState(testing_flag=True)
        assert StateTracker._classify_pattern(s) == "testing"

    def test_pattern_classification_repetitive(self):
        s = ConversationState(repetition_count=3)
        assert StateTracker._classify_pattern(s) == "repetitive"

    def test_pattern_classification_rapid_fire(self):
        s = ConversationState(short_query_streak=4)
        assert StateTracker._classify_pattern(s) == "rapid_fire"

    def test_pattern_classification_exploratory(self):
        s = ConversationState(
            intent_history=["general", "knowledge_base", "profile", "continuation"]
        )
        assert StateTracker._classify_pattern(s) == "exploratory"

    def test_pattern_classification_normal(self):
        s = ConversationState()
        assert StateTracker._classify_pattern(s) == "normal"

    def test_precision_mode_adversarial(self):
        s = ConversationState()
        assert StateTracker._compute_precision_mode(s, "that's not right") == "adversarial"

    def test_precision_mode_implementation(self):
        s = ConversationState()
        assert StateTracker._compute_precision_mode(s, "implement a function") == "implementation"

    def test_precision_mode_speculative(self):
        s = ConversationState()
        assert StateTracker._compute_precision_mode(s, "what if we used redis") == "speculative"

    def test_precision_mode_concise_rapid_fire(self):
        s = ConversationState(interaction_pattern="rapid_fire")
        assert StateTracker._compute_precision_mode(s) == "concise"

    def test_precision_mode_concise_short_streak(self):
        s = ConversationState(short_query_streak=3)
        assert StateTracker._compute_precision_mode(s) == "concise"

    def test_precision_mode_default_analytical(self):
        s = ConversationState()
        assert StateTracker._compute_precision_mode(s) == "analytical"

    def test_short_query_streak(self):
        s = ConversationState()
        StateTracker.update(s, "hi", "general", 0.9)
        StateTracker.update(s, "ok", "general", 0.9)
        StateTracker.update(s, "yes", "general", 0.9)
        assert s.short_query_streak == 3

    def test_short_query_streak_resets(self):
        s = ConversationState()
        StateTracker.update(s, "hi", "general", 0.9)
        StateTracker.update(s, "ok", "general", 0.9)
        StateTracker.update(s, "this is a much longer question about something", "general", 0.9)
        assert s.short_query_streak == 0

    def test_avg_query_length(self):
        s = ConversationState()
        StateTracker.update(s, "hello world", "general", 0.9)  # 2 words
        assert s.avg_query_length == 2.0
        StateTracker.update(s, "this is four words", "general", 0.9)  # 4 words
        assert s.avg_query_length == 3.0  # (2+4)/2

    def test_intent_history_window(self):
        s = ConversationState()
        for i in range(15):
            StateTracker.update(s, f"query {i}", "general", 0.9)
        # Should be capped at pattern window (10)
        assert len(s.intent_history) == 10

    def test_testing_flag_clears(self):
        """Testing flag should clear when user moves on."""
        s = ConversationState(testing_flag=True, message_count=5)
        StateTracker.update(s, "what is python", "knowledge_base", 0.9)
        assert s.testing_flag is False
