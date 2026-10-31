"""Tests for the prompt_orchestrator.build_messages() function.

build_messages() is pure Python — no LLM calls, no I/O.
Every test asserts the exact structure of the returned messages list.
"""

import pytest
from llm.prompt_orchestrator import build_messages
from llm.prompts import (
    SYSTEM_PROMPT,
    GREETING_PERSONALIZATION_FRAME,
    PROFILE_CONTEXT_FRAME,
    RAG_CONTEXT_FRAME,
    QA_CONTEXT_FRAME,
    PRIVACY_QA_FRAME,
)


QUERY = "What is pgvector?"


# ─── Baseline structure ───────────────────────────────────────────────────

def test_minimal_messages():
    """Minimal call: system prompt + precision frame + user query."""
    msgs = build_messages(QUERY)
    # Default precision_mode="analytical" always injects a behavior frame
    assert len(msgs) == 3
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[-1] == {"role": "user", "content": QUERY}


def test_user_query_always_last():
    """The user query must always be the final message."""
    msgs = build_messages(
        QUERY,
        rag_context="Some docs",
        profile_context="Name: Alice",
        similar_qa_context="Prior Q&A",
    )
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == QUERY


def test_system_prompt_always_first():
    msgs = build_messages(QUERY, rag_context="docs", profile_context="profile")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT


# ─── Optional context frames ──────────────────────────────────────────────

def test_greeting_name_injected():
    msgs = build_messages(QUERY, greeting_name="Bob")
    system_contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("Bob" in c for c in system_contents)
    # Greeting frame should come before profile frame (index 1)
    greeting_idx = next(
        i for i, m in enumerate(msgs) if m["role"] == "system" and "Bob" in m["content"]
    )
    assert greeting_idx == 1  # immediately after SYSTEM_PROMPT


def test_no_greeting_frame_when_name_not_set():
    msgs = build_messages(QUERY)
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    expected_fragment = GREETING_PERSONALIZATION_FRAME.split("{name}")[0].strip()[:30]
    assert not any(expected_fragment in c for c in contents)


def test_profile_frame_injected():
    msgs = build_messages(QUERY, profile_context="Name: Charlie\nJob: Developer")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("Charlie" in c for c in contents)


def test_rag_frame_injected():
    msgs = build_messages(QUERY, rag_context="pgvector is a PostgreSQL extension.")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("pgvector is a PostgreSQL extension." in c for c in contents)


def test_qa_context_injected_when_no_privacy_mode():
    msgs = build_messages(QUERY, similar_qa_context="Q: What is it? A: It's cool.")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("Q: What is it?" in c for c in contents)


def test_privacy_frame_injected_when_privacy_mode():
    msgs = build_messages(QUERY, privacy_mode=True)
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any(PRIVACY_QA_FRAME[:40] in c for c in contents)


def test_privacy_mode_suppresses_qa_context():
    """privacy_mode=True must inject PRIVACY_QA_FRAME, not QA_CONTEXT_FRAME."""
    msgs = build_messages(
        QUERY,
        privacy_mode=True,
        similar_qa_context="Prior Q&A text",
    )
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    # PRIVACY_QA_FRAME should be present
    assert any(PRIVACY_QA_FRAME[:40] in c for c in contents)
    # QA_CONTEXT_FRAME should NOT be present
    qa_fragment = QA_CONTEXT_FRAME.split("{qa}")[0].strip()[:30]
    assert not any(qa_fragment in c for c in contents)


# ─── History injection ────────────────────────────────────────────────────

def test_curated_history_used_over_chat_history():
    chat = [{"role": "user", "content": "old message"}]
    curated = [{"role": "user", "content": "curated message"}]
    msgs = build_messages(QUERY, chat_history=chat, curated_history=curated)
    contents = [m["content"] for m in msgs]
    assert "curated message" in contents
    assert "old message" not in contents


def test_chat_history_fallback_when_no_curated():
    chat = [{"role": "user", "content": "fallback message"}]
    msgs = build_messages(QUERY, chat_history=chat, curated_history=None)
    contents = [m["content"] for m in msgs]
    assert "fallback message" in contents


def test_no_history_when_both_none():
    msgs = build_messages(QUERY, chat_history=None, curated_history=None)
    # system prompt + precision frame + user query
    assert len(msgs) == 3


def test_history_appears_before_user_query():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
    ]
    msgs = build_messages(QUERY, curated_history=history)
    history_indices = [i for i, m in enumerate(msgs) if m["content"] in ("first", "response")]
    user_idx = next(i for i, m in enumerate(msgs) if m["content"] == QUERY)
    assert all(hi < user_idx for hi in history_indices)


# ─── Frame ordering ───────────────────────────────────────────────────────

def test_frame_order_system_greeting_profile_rag_qa_history_user():
    """Verify the documented injection order is preserved end-to-end."""
    history = [{"role": "user", "content": "earlier question"}]
    msgs = build_messages(
        QUERY,
        greeting_name="Dana",
        profile_context="Name: Dana",
        rag_context="RAG docs",
        similar_qa_context="QA context",
        curated_history=history,
    )
    # All system messages first, then history, then user query
    system_indices = [i for i, m in enumerate(msgs) if m["role"] == "system"]
    history_indices = [i for i, m in enumerate(msgs) if m["content"] == "earlier question"]
    user_index = len(msgs) - 1

    assert max(system_indices) < min(history_indices)
    assert max(history_indices) < user_index


# ─── Behavior frame tests ─────────────────────────────────────────────────

def test_behavior_frame_injected_when_context_provided():
    """Behavior state frame appears when behavior_context is set."""
    msgs = build_messages(QUERY, behavior_context="User is frustrated.")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("User is frustrated." in c for c in contents)


def test_behavior_frame_not_injected_when_empty():
    """With default precision_mode, behavior frame is still injected."""
    msgs = build_messages(QUERY)
    # Default "analytical" precision always creates a behavior frame
    assert len(msgs) == 3
    # The analytical frame text should be in the behavior frame
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    assert "structured analysis" in contents.lower() or "analytical" in contents.lower()


def test_behavior_frame_between_greeting_and_profile():
    """Behavior frame should appear after greeting but before profile."""
    msgs = build_messages(
        QUERY,
        greeting_name="Alice",
        behavior_context="User is curious.",
        profile_context="Name: Alice",
    )
    greeting_idx = next(
        i for i, m in enumerate(msgs) if m["role"] == "system" and "Alice" in m["content"] and "Name:" not in m["content"]
    )
    behavior_idx = next(
        i for i, m in enumerate(msgs) if m["role"] == "system" and "curious" in m["content"]
    )
    profile_idx = next(
        i for i, m in enumerate(msgs) if m["role"] == "system" and "Name:" in m["content"]
    )
    assert greeting_idx < behavior_idx < profile_idx


def test_personality_mode_injected():
    """Personality mode text appears in behavior frame."""
    msgs = build_messages(QUERY, behavior_context="Testing.", personality_mode="concise")
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    # The concise personality frame should be included
    assert "concise" in contents.lower() or "brief" in contents.lower() or "shorter" in contents.lower()


def test_precision_mode_injected():
    """Precision mode text appears in system messages."""
    msgs = build_messages(QUERY, behavior_context="Info.", precision_mode="adversarial")
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    # The adversarial frame should mention stress-testing or critique
    assert "stress-testing" in contents.lower() or "critique" in contents.lower()


def test_thread_context_injected():
    """Thread context (label + summary) appears in system messages."""
    ctx = {"thread_label": "RAG Architecture", "thread_summary": "Discussing vector search approaches."}
    msgs = build_messages(QUERY, behavior_context="Info.", thread_context=ctx)
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    assert "RAG Architecture" in contents
    assert "vector search" in contents


def test_research_context_injected():
    """Research insights and concepts appear in system messages."""
    ctx = {
        "related_insights": [{"type": "decision", "text": "Use pgvector for embeddings"}],
        "concept_links": [{"concept": "pgvector"}, {"concept": "embeddings"}],
    }
    msgs = build_messages(QUERY, behavior_context="Info.", research_context=ctx)
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    assert "pgvector" in contents


def test_meta_instruction_injected():
    """Meta instruction appears in behavior frame."""
    msgs = build_messages(QUERY, behavior_context="State info.", meta_instruction="Acknowledge repetition.")
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    assert "Acknowledge repetition." in contents


def test_response_length_hint_injected():
    """Response length hint text appears in behavior frame."""
    msgs = build_messages(QUERY, behavior_context="Info.", response_length_hint="brief")
    contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
    # Should contain the brief length hint
    assert "brief" in contents.lower() or "short" in contents.lower() or "sentence" in contents.lower()
