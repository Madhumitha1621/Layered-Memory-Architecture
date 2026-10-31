"""Token-budget context management — history fitting and conversation summarization.

Why this module exists
----------------------
LLMs have hard context window limits.  Without token budgeting,
a long conversation silently overflows the model's window, causing
truncation artifacts or outright errors.

The RECENCY_WINDOW heuristic (last N messages) is a fast guard, but it
is *count-based*, not *token-based*.  A single long message can consume
as many tokens as 20 short ones.  This module enforces precision.

Token estimation
----------------
We estimate tokens as ``len(text) // 4``.  This is a well-known
approximation (~97 % accurate for English prose, ~90 % for code).
For exact counting, replace ``estimate_tokens()`` with
``tiktoken.encoding_for_model("gpt-4o").encode(text)``.

Public API
----------
    estimate_tokens(text)             → int
    message_tokens(msg)               → int
    history_tokens(messages)          → int
    fit_messages_to_budget(messages, budget_tokens, min_recent=4) → list[dict]
    summarize_old_turns(messages, max_history_tokens, completion_fn) → list[dict]
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
#  Constants
# --------------------------------------------------------------------------

# Characters-per-token approximation.  4 chars ≈ 1 GPT token for English.
_CHARS_PER_TOKEN: int = 4

# Fixed per-message overhead: role string + JSON framing (~10 tokens).
_MSG_OVERHEAD: int = 10

# Prefix that identifies a summary message produced by this module.
SUMMARY_PREFIX: str = "[Summary of earlier conversation]:"

# Maximum tokens sent to the summarizer LLM as transcript.
_MAX_SUMMARIZER_INPUT_TOKENS: int = 4000


# --------------------------------------------------------------------------
#  Token estimation
# --------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count from character length (chars / 4).

    Fast, zero-dependency approximation.  Replace the body with a tiktoken
    call for exact counting when precision matters.
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)


def message_tokens(msg: dict) -> int:
    """Estimated token cost for a single OpenAI-format message dict."""
    return estimate_tokens(msg.get("content", "")) + _MSG_OVERHEAD


def history_tokens(messages: list[dict]) -> int:
    """Total estimated token cost for a list of messages."""
    return sum(message_tokens(m) for m in messages)


# --------------------------------------------------------------------------
#  Dynamic budget computation
# --------------------------------------------------------------------------

def compute_history_budget(
    context_window: int,
    response_reserve: int,
    preamble_tokens: int = 0,
    min_budget: int = 1000,
) -> int:
    """Compute remaining token budget for conversation history.

    Subtracts system prompts, RAG context, profile, and response reserve
    from the total context window.  Returns at least ``min_budget`` so the
    model always has some conversational context.

    Args:
        context_window:   Total model context window (e.g. 65536).
        response_reserve: Tokens reserved for the model's output.
        preamble_tokens:  Tokens already used by system prompts, RAG, profile, etc.
        min_budget:       Floor — never return less than this.
    """
    remaining = context_window - preamble_tokens - response_reserve
    budget = max(remaining, min_budget)
    logger.debug(
        "History budget: %d (window=%d, preamble=%d, reserve=%d, min=%d)",
        budget, context_window, preamble_tokens, response_reserve, min_budget,
    )
    return budget


# --------------------------------------------------------------------------
#  Budget fitting
# --------------------------------------------------------------------------

def fit_messages_to_budget(
    messages: list[dict],
    budget_tokens: int,
    min_recent: int = 4,
) -> list[dict]:
    """Trim oldest messages until the list fits within *budget_tokens*.

    Always preserves the last ``min_recent`` messages regardless of budget,
    so the model always has some immediate conversational context.

    Uses O(n) prefix-sum trimming instead of re-counting on each drop.

    Args:
        messages:      Chronological ``[{"role": ..., "content": ...}]`` list.
        budget_tokens: Maximum allowed token count for the history block.
        min_recent:    Minimum tail messages to keep unconditionally.

    Returns:
        Trimmed list (same dict objects, not copies).  Empty list is returned
        unchanged.
    """
    if not messages:
        return messages

    costs = [message_tokens(m) for m in messages]
    total = sum(costs)

    # Fast path — already fits.
    if total <= budget_tokens:
        return messages

    # Drop from the front until we fit or reach min_recent.
    n = len(messages)
    cut = 0
    while cut < n - min_recent and total > budget_tokens:
        total -= costs[cut]
        cut += 1

    trimmed = messages[cut:]
    if cut > 0:
        logger.info(
            "Context budget: dropped %d oldest messages to fit %d-token budget "
            "(kept %d, ~%d tokens)",
            cut, budget_tokens, len(trimmed), total,
        )
    return trimmed


# --------------------------------------------------------------------------
#  Progressive summarization
# --------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = (
    "You are a precise conversation summarizer. Given a conversation transcript "
    "(which may include a prior summary of even earlier messages), produce a "
    "cohesive summary that preserves:\n"
    "1. Key topics discussed and any decisions made\n"
    "2. Important facts the user shared about themselves (name, preferences, etc.)\n"
    "3. Unresolved questions or pending tasks\n"
    "4. The user's apparent goals and current focus\n"
    "5. Key technical details, code specifics, or exact data mentioned\n\n"
    "Write 3-8 sentences. Be factual and specific — include names, numbers, "
    "and technical terms rather than vague references. Do NOT add labels, "
    "bullet points, or meta-commentary."
)


def _is_summary_message(msg: dict) -> bool:
    """Check whether a message is a summary produced by this module."""
    return (
        msg.get("role") == "system"
        and msg.get("content", "").startswith(SUMMARY_PREFIX)
    )


def _build_transcript(
    existing_summary: str | None,
    overflow_messages: list[dict],
) -> str:
    """Build a token-budgeted transcript for the summarizer LLM.

    If an existing summary exists, it is prepended as seed context so the
    new summary progressively refines prior knowledge rather than starting
    fresh each time.

    When the overflow exceeds the transcript budget, the *newest* overflow
    messages are kept and the oldest dropped.  The overflow sits directly
    before the verbatim recent tail, so dropping its newer end would leave a
    gap in the middle of the conversation that neither the summary nor the
    tail covers.
    """
    parts: list[str] = []
    used_tokens = 0

    # Include prior summary as seed context
    if existing_summary:
        seed = (
            f"Prior summary of earlier turns:\n{existing_summary}"
            f"\n\nNew messages to incorporate:"
        )
        seed_tokens = estimate_tokens(seed) + 10
        parts.append(seed)
        used_tokens += seed_tokens

    # Walk newest → oldest until we hit the transcript budget
    kept: list[str] = []
    omitted = 0
    for i, m in enumerate(reversed(overflow_messages)):
        line = f"{m['role'].title()}: {m['content']}"
        line_tokens = estimate_tokens(line)
        if used_tokens + line_tokens > _MAX_SUMMARIZER_INPUT_TOKENS:
            remaining_chars = max(
                0, (_MAX_SUMMARIZER_INPUT_TOKENS - used_tokens) * _CHARS_PER_TOKEN
            )
            if remaining_chars > 50:
                # Keep the end of the message — it's closest to what follows
                kept.append(f"{m['role'].title()}: …{m['content'][-remaining_chars:]}")
                i += 1
            omitted = len(overflow_messages) - i
            break
        kept.append(line)
        used_tokens += line_tokens

    if omitted:
        parts.append(f"[{omitted} earlier message(s) omitted]")
    parts.extend(reversed(kept))
    return "\n".join(parts)


def summarize_old_turns(
    messages: list[dict],
    max_history_tokens: int,
    completion_fn: Callable[[list[dict]], str],
    min_recent: int = 6,
) -> list[dict]:
    """Progressive summarization — ChatGPT-style rolling context compression.

    If the history fits within *max_history_tokens*, returns it unchanged.
    Otherwise:

    1. Detects any existing summary message from a prior pass.
    2. Splits remaining messages into ``[overflow | tail(min_recent)]``.
    3. Builds a token-budgeted transcript (seeded with the prior summary).
    4. Calls *completion_fn* to produce a new summary.
    5. Returns ``[summary_system_msg] + tail``.

    Falls back to ``fit_messages_to_budget`` on any LLM error.

    Args:
        messages:           Chronological conversation messages.
        max_history_tokens: Token budget for the full history block.
        completion_fn:      ``(messages) -> str`` callable.
        min_recent:         Recent turns to always keep verbatim.
    """
    if not messages or history_tokens(messages) <= max_history_tokens:
        return messages

    # ── Extract existing summary (progressive chaining) ───────────────
    existing_summary: str | None = None
    non_summary: list[dict] = []
    for m in messages:
        if _is_summary_message(m):
            existing_summary = m["content"][len(SUMMARY_PREFIX):].strip()
        else:
            non_summary.append(m)

    # ── Split into overflow and recent ────────────────────────────────
    if len(non_summary) <= min_recent:
        return fit_messages_to_budget(messages, max_history_tokens, min_recent)

    recent = non_summary[-min_recent:]
    overflow = non_summary[:-min_recent]

    if not overflow and not existing_summary:
        return fit_messages_to_budget(messages, max_history_tokens, min_recent)

    # ── Build token-budgeted transcript ───────────────────────────────
    transcript = _build_transcript(existing_summary, overflow)

    try:
        summary_text = completion_fn([
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {"role": "user", "content": transcript},
        ])
        summary_msg: dict = {
            "role": "system",
            "content": f"{SUMMARY_PREFIX} {summary_text.strip()}",
        }
        result = [summary_msg] + list(recent)

        logger.info(
            "Context: compressed %d turns%s into ~%d-token summary "
            "(kept %d recent, ~%d tokens total)",
            len(overflow),
            " + prior summary" if existing_summary else "",
            estimate_tokens(summary_text),
            len(recent),
            history_tokens(result),
        )
        return result

    except Exception as exc:
        logger.warning("Summarization failed (%s) — falling back to recency trim", exc)
        return fit_messages_to_budget(messages, max_history_tokens, min_recent)
