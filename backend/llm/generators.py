"""Response generation — streaming and non-streaming.

All generation functions delegate message assembly to
:func:`prompt_orchestrator.build_messages` and LLM calls to
:func:`client.completion`.  They contain no prompt text or behavior rules.
"""

import json
import logging
from collections.abc import Generator

from .client import completion, stream_text_deltas, MAX_RESPONSE_TOKENS, MAX_TITLE_TOKENS
from .prompt_orchestrator import build_messages
from .prompts import TITLE_PROMPT

logger = logging.getLogger(__name__)

# Shown to clients on LLM failure; raw exceptions can leak provider details.
GENERATION_ERROR_MESSAGE = "The model failed to generate a response. Please try again."


def generate_response(
    user_query: str,
    chat_history: list | None = None,
    rag_context: str = "",
    profile_context: str = "",
    similar_qa_context: str = "",
    curated_history: list | None = None,
    privacy_mode: bool = False,
    greeting_name: str | None = None,
    behavior_context: str = "",
    meta_instruction: str = "",
    personality_mode: str = "default",
    precision_mode: str = "analytical",
    response_length_hint: str = "normal",
    thread_context: dict | None = None,
    research_context: dict | None = None,
) -> str:
    """Generate a complete (non-streaming) response.

    Raises on LLM failure so callers never mistake an error for a reply
    (and persist it as one).
    """
    messages = build_messages(
        user_query,
        chat_history=chat_history,
        curated_history=curated_history,
        rag_context=rag_context,
        profile_context=profile_context,
        similar_qa_context=similar_qa_context,
        privacy_mode=privacy_mode,
        greeting_name=greeting_name,
        behavior_context=behavior_context,
        meta_instruction=meta_instruction,
        personality_mode=personality_mode,
        precision_mode=precision_mode,
        response_length_hint=response_length_hint,
        thread_context=thread_context,
        research_context=research_context,
    )
    return completion(messages)


def generate_response_stream(
    user_query: str,
    chat_history: list | None = None,
    rag_context: str = "",
    profile_context: str = "",
    similar_qa_context: str = "",
    curated_history: list | None = None,
    privacy_mode: bool = False,
    greeting_name: str | None = None,
    behavior_context: str = "",
    meta_instruction: str = "",
    personality_mode: str = "default",
    precision_mode: str = "analytical",
    response_length_hint: str = "normal",
    thread_context: dict | None = None,
    research_context: dict | None = None,
) -> Generator[str, None, None]:
    """Yield Vercel AI SDK data-stream lines.

    Protocol::

        0:"token"\\n        text delta
        3:"message"\\n      error part (generation failed; not part of the reply)
        e:{"finishReason":"stop"}\\n   finish event
        d:{"finishReason":"stop"}\\n   done signal
    """
    messages = build_messages(
        user_query,
        chat_history=chat_history,
        curated_history=curated_history,
        rag_context=rag_context,
        profile_context=profile_context,
        similar_qa_context=similar_qa_context,
        privacy_mode=privacy_mode,
        greeting_name=greeting_name,
        behavior_context=behavior_context,
        meta_instruction=meta_instruction,
        personality_mode=personality_mode,
        precision_mode=precision_mode,
        response_length_hint=response_length_hint,
        thread_context=thread_context,
        research_context=research_context,
    )
    try:
        for text in stream_text_deltas(messages):
            yield f'0:{json.dumps(text)}\n'
        yield f'e:{json.dumps({"finishReason": "stop"})}\n'
        yield f'd:{json.dumps({"finishReason": "stop"})}\n'
    except Exception as e:
        logger.error("Streaming error: %s", e)
        yield f'3:{json.dumps(GENERATION_ERROR_MESSAGE)}\n'
        yield f'e:{json.dumps({"finishReason": "error"})}\n'
        yield f'd:{json.dumps({"finishReason": "error"})}\n'


def generate_title(user_message: str) -> str:
    """Generate a 3-6 word conversation title from the first user message."""
    try:
        messages = [
            {"role": "system", "content": TITLE_PROMPT},
            {"role": "user", "content": user_message},
        ]
        raw = completion(messages, temperature=0.5, max_tokens=MAX_TITLE_TOKENS)
        title = (raw or "").strip().strip('"').strip("'")
        if not title:
            # LLM returned empty — fall back to first words
            words = user_message.split()[:5]
            return " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
        if len(title) > 50:
            title = title[:50].rsplit(" ", 1)[0] or title[:50]
        return title
    except Exception as e:
        logger.error("Title generation error: %s", e)
        words = user_message.split()[:5]
        return " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
