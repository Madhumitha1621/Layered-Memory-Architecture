"""Tests for context_manager — token estimation, budget fitting, and summarization."""

from unittest.mock import MagicMock
import pytest

from context_manager import (
    estimate_tokens,
    message_tokens,
    history_tokens,
    fit_messages_to_budget,
    summarize_old_turns,
    compute_history_budget,
    SUMMARY_PREFIX,
)


# ─── estimate_tokens ─────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert estimate_tokens("") == 1

    def test_short_text(self):
        # "hello" = 5 chars → 5//4 = 1, but max(1, ...) = 1
        assert estimate_tokens("hello") == 1

    def test_long_text(self):
        text = "a" * 1000
        assert estimate_tokens(text) == 250

    def test_proportional_scaling(self):
        short = estimate_tokens("a" * 400)   # 100
        long = estimate_tokens("a" * 800)    # 200
        assert long == short * 2


# ─── message_tokens ──────────────────────────────────────────────────────

def test_message_tokens_includes_overhead():
    msg = {"role": "user", "content": "a" * 40}  # 40//4=10 content tokens + 10 overhead
    assert message_tokens(msg) == 20

def test_message_tokens_empty_content():
    msg = {"role": "system", "content": ""}
    # max(1,0) + 10 overhead
    assert message_tokens(msg) == 11


# ─── history_tokens ───────────────────────────────────────────────────────

def test_history_tokens_empty():
    assert history_tokens([]) == 0

def test_history_tokens_sum():
    msgs = [
        {"role": "user", "content": "a" * 40},       # 10+10 = 20
        {"role": "assistant", "content": "a" * 80},  # 20+10 = 30
    ]
    assert history_tokens(msgs) == 50


# ─── fit_messages_to_budget ───────────────────────────────────────────────

def _make_msgs(n: int, content_chars: int = 100) -> list[dict]:
    """Create n alternating user/assistant messages."""
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": "x" * content_chars}
        for i in range(n)
    ]


class TestFitMessagesToBudget:
    def test_empty_list_unchanged(self):
        assert fit_messages_to_budget([], 1000) == []

    def test_fits_unchanged_when_under_budget(self):
        msgs = _make_msgs(3, content_chars=10)
        result = fit_messages_to_budget(msgs, budget_tokens=10000)
        assert result == msgs

    def test_trims_oldest_first(self):
        msgs = _make_msgs(10, content_chars=100)
        # Each message: 100//4=25 + 10 overhead = 35 tokens.
        # Budget: 4 messages = 4 * 35 = 140 tokens
        result = fit_messages_to_budget(msgs, budget_tokens=140, min_recent=2)
        assert len(result) <= 4
        # Result must be the TAIL of the original list
        assert result == msgs[-len(result):]

    def test_always_keeps_min_recent(self):
        msgs = _make_msgs(10, content_chars=4000)  # giant messages
        result = fit_messages_to_budget(msgs, budget_tokens=1, min_recent=4)
        assert len(result) == 4
        assert result == msgs[-4:]

    def test_result_within_budget_or_min_recent(self):
        msgs = _make_msgs(20, content_chars=200)
        budget = 500
        result = fit_messages_to_budget(msgs, budget_tokens=budget, min_recent=2)
        total = history_tokens(result)
        # Either it fits in budget OR we're at min_recent (2 messages)
        assert total <= budget or len(result) == 2


# ─── summarize_old_turns ─────────────────────────────────────────────────

class TestSummarizeOldTurns:
    def test_returns_unchanged_when_fits(self):
        msgs = _make_msgs(4, content_chars=10)
        fn = MagicMock()
        result = summarize_old_turns(msgs, max_history_tokens=100000, completion_fn=fn)
        assert result == msgs
        fn.assert_not_called()

    def test_summarizes_overflow(self):
        msgs = _make_msgs(10, content_chars=1000)
        mock_summary = "User discussed Python and asked about pgvector."
        fn = MagicMock(return_value=mock_summary)
        result = summarize_old_turns(msgs, max_history_tokens=100, completion_fn=fn, min_recent=4)
        fn.assert_called_once()
        # Result: summary message + 4 recent
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert "Summary" in result[0]["content"]
        assert mock_summary in result[0]["content"]
        # The last 4 messages should be the tail
        assert result[1:] == msgs[-4:]

    def test_falls_back_on_llm_error(self):
        msgs = _make_msgs(10, content_chars=500)
        fn = MagicMock(side_effect=RuntimeError("LLM down"))
        # Should not raise; falls back to trim
        result = summarize_old_turns(msgs, max_history_tokens=200, completion_fn=fn, min_recent=4)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_not_enough_old_turns_to_summarize(self):
        """When there's only min_recent messages, just trim."""
        msgs = _make_msgs(4, content_chars=4000)  # 4 huge messages
        fn = MagicMock(return_value="summary")
        result = summarize_old_turns(msgs, max_history_tokens=50, completion_fn=fn, min_recent=4)
        # Can't split further — falls back to fit_messages_to_budget
        assert isinstance(result, list)

    def test_progressive_summary_chains_existing(self):
        """When history contains a prior summary, the summarizer receives it as seed."""
        summary_msg = {"role": "system", "content": f"{SUMMARY_PREFIX} User discussed Python basics."}
        msgs = [summary_msg] + _make_msgs(10, content_chars=500)
        mock_new_summary = "User discussed Python basics and then moved to pgvector."
        fn = MagicMock(return_value=mock_new_summary)
        result = summarize_old_turns(msgs, max_history_tokens=200, completion_fn=fn, min_recent=4)
        fn.assert_called_once()
        # The transcript sent to the LLM should contain the prior summary
        call_args = fn.call_args[0][0]  # first positional arg = messages list
        user_content = call_args[1]["content"]  # the transcript
        assert "Python basics" in user_content
        # Result should have new summary + recent
        assert result[0]["role"] == "system"
        assert mock_new_summary in result[0]["content"]


# ─── compute_history_budget ───────────────────────────────────────────────

class TestComputeHistoryBudget:
    def test_basic_budget(self):
        budget = compute_history_budget(
            context_window=65536,
            response_reserve=2048,
            preamble_tokens=500,
        )
        assert budget == 65536 - 2048 - 500

    def test_min_budget_floor(self):
        budget = compute_history_budget(
            context_window=4000,
            response_reserve=3500,
            preamble_tokens=2000,
            min_budget=1000,
        )
        # 4000 - 3500 - 2000 = -1500, but floor is 1000
        assert budget == 1000

    def test_zero_preamble(self):
        budget = compute_history_budget(
            context_window=8000,
            response_reserve=2000,
        )
        assert budget == 6000


# ─── _build_transcript (summarizer input budget) ──────────────────────────

class TestBuildTranscriptBudget:
    def test_keeps_newest_overflow_when_over_budget(self):
        from context_manager import _MAX_SUMMARIZER_INPUT_TOKENS, _build_transcript

        # Each message ~250 tokens; 40 of them far exceed the summarizer budget
        msgs = [{"role": "user", "content": f"m{i:02d} " + "x" * 1000} for i in range(40)]
        transcript = _build_transcript(None, msgs)

        # The newest overflow message (adjacent to the verbatim tail) survives
        assert "m39 " in transcript
        assert "m00 " not in transcript
        assert "earlier message(s) omitted" in transcript
        # Kept messages remain in chronological order
        assert transcript.index("m38 ") < transcript.index("m39 ")
        assert len(transcript) // 4 <= _MAX_SUMMARIZER_INPUT_TOKENS + 50

    def test_everything_fits_unchanged_order(self):
        from context_manager import _build_transcript

        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert _build_transcript(None, msgs) == "User: a\nAssistant: b"
