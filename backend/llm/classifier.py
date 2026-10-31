"""Intent classification — pre-heuristics + LLM call.

Pre-heuristic fast-paths (no LLM round-trip):
  greeting          → general   (saves ~1-2s per greeting message)
  profile statement → profile   (saves ~1-2s for "my name is …" etc.)
  privacy phrase    → privacy
  short pronoun q.  → continuation  (needs context)
"""

import json
import logging

import cache
from .client import completion, MAX_CLASSIFIER_TOKENS
from .prompts import INTENT_PROMPT

# Signal lists are defined in policy.py (single source of truth).
# Import them here to avoid duplication.
from policy import (
    GREETING_PATTERNS as _GREETING_PATTERNS,
    PROFILE_STATEMENT_PREFIXES as _PROFILE_OPENERS,
)

logger = logging.getLogger(__name__)

# ── Classification-only signal lists ──────────────────────────────────────

PRIVACY_SIGNALS = [
    "invasion of privacy", "privacy concern", "what do you know about me",
    "what data do you", "what information do you", "do you store",
    "do you track", "are you tracking", "delete my data", "erase my",
    "forget about me", "what have you stored", "my personal data",
    "data privacy", "how is my data", "who has access", "is my data safe",
    "stop storing", "do you collect", "what do you remember about me",
    "stored about me", "data about me", "what info do you have",
]

CONTINUATION_PRONOUNS = {
    "that", "it", "this", "those", "these", "they", "them",
    "its", "their", "the same",
}

# Short continuation cues that indicate a follow-up without pronouns
CONTINUATION_SIGNALS = {
    "why", "how", "elaborate", "more", "explain", "details",
    "specifically", "example", "another", "also",
}

VALID_INTENTS = {"general", "continuation", "knowledge_base", "profile", "privacy"}


def classify_intent(
    user_query: str,
    conversation_context: list | None = None,
) -> dict:
    """Classify user intent using pre-heuristics then LLM fallback.

    Returns ``{"intent": str, "confidence": float}``.
    """
    query_lower = user_query.strip().lower()

    # ── Cache check ─────────────────────────────────────────────────────
    cached = cache.get_classification(user_query, conversation_context)
    if cached:
        logger.info(f"Cache hit: intent={cached['intent']}")
        cached.setdefault("source", "cache")
        return cached

    # ── Greeting fast-path (saves a full LLM round-trip) ─────────────────
    _words = query_lower.split()
    if len(_words) <= 8:
        for _pat in _GREETING_PATTERNS:
            _rest = query_lower[len(_pat):]
            if query_lower == _pat or (
                query_lower.startswith(_pat) and (_rest == "" or _rest[0] in " ,!?")
            ):
                logger.info("Pre-heuristic: greeting → general (no LLM)")
                return {"intent": "general", "confidence": 0.97, "source": "heuristic:greeting"}

    # ── Profile statement fast-path ───────────────────────────────────────
    # Only for SHORT declarative statements.  Longer messages starting with
    # "I have a question about…" should fall through to the LLM.
    if (
        "?" not in user_query
        and len(_words) <= 15
        and any(query_lower.startswith(p) for p in _PROFILE_OPENERS)
    ):
        logger.info("Pre-heuristic: profile statement → profile (no LLM)")
        return {"intent": "profile", "confidence": 0.92, "source": "heuristic:profile"}

    # ── Fast privacy check ────────────────────────────────────────────────
    if any(sig in query_lower for sig in PRIVACY_SIGNALS):
        logger.info("Pre-heuristic: privacy signal detected")
        return {"intent": "privacy", "confidence": 0.95, "source": "heuristic:privacy"}

    # ── Continuation check (needs context) ────────────────────────────────
    if conversation_context and len(conversation_context) >= 2:
        _raw_words = query_lower.split()
        words = {w.strip("?!.,;:") for w in _raw_words}
        word_count = len(_raw_words)
        has_pronoun = bool(words & CONTINUATION_PRONOUNS)
        has_signal = bool(words & CONTINUATION_SIGNALS)
        has_question = "?" in user_query
        if word_count <= 8 and (
            (has_pronoun and has_question)        # "What is it used for?"
            or (has_signal and word_count <= 4)   # "Why?" "Elaborate" "More details"
        ):
            logger.info("Pre-heuristic: continuation (pronoun/signal + short follow-up)")
            return {"intent": "continuation", "confidence": 0.85, "source": "heuristic:continuation"}

    # ── LLM classification ────────────────────────────────────────────────
    try:
        context_text = ""
        if conversation_context:
            recent = conversation_context[-6:]
            lines = [f"{m['role'].title()}: {m['content'][:120]}" for m in recent]
            context_text = "\nRecent conversation:\n" + "\n".join(lines)

        messages = [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": f"Message: {user_query}{context_text}"},
        ]

        _raw = completion(
            messages,
            temperature=0.0,
            max_tokens=MAX_CLASSIFIER_TOKENS,
        )
        raw = (_raw or "").strip()

        if not raw:
            logger.warning("Empty classifier response — defaulting to general")
            return {"intent": "general", "confidence": 0.5, "source": "llm:empty"}

        # Robust JSON extraction
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        result = json.loads(raw)
        intent = result.get("intent", "general")
        confidence = float(result.get("confidence", 0.5))

        if intent not in VALID_INTENTS:
            logger.warning(f"Unknown intent '{intent}' → general")
            intent = "general"

        result = {"intent": intent, "confidence": confidence, "source": "llm"}
        cache.set_classification(user_query, result, conversation_context)
        return result

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"intent": "general", "confidence": 0.5, "source": "llm:error"}
