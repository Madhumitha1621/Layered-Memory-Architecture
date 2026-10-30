"""Conversation State — the missing 'C tier' memory layer.

Memory tiers:
  A) Episodic   — facts extracted from queries (user_queries table)   ✅ exists
  B) Semantic   — user traits and preferences (user_profile table)    ✅ exists
  C) Conversational — per-conversation behavioral state               ← THIS MODULE

The state object tracks meta-conversational signals that raw retrieval
misses: repetition patterns, emotional tone shifts, testing behavior,
topic drift, and interaction style.  This is what makes the difference
between a search engine and a conversational partner.

Public API:
    ConversationState      — dataclass holding all state fields
    StateTracker           — stateless analyzer that computes state updates
    get_or_create_state()  — in-memory cache + DB fallback
    update_state()         — apply new message to state
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVERSATION STATE OBJECT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationState:
    """Per-conversation behavioral state — updated every message.

    This is NOT chat history.  It is a meta-layer summarizing *how* the
    user is interacting, not *what* they said.
    """

    # ── Topic tracking ────────────────────────────────────────────────────
    current_topic: str = ""                 # rolling topic label
    topic_turns_stable: int = 0             # how many turns on same topic
    topic_drift_count: int = 0              # number of topic changes

    # ── Emotional / tone signals ──────────────────────────────────────────
    emotional_tone: str = "neutral"         # neutral | positive | frustrated | curious | playful
    tone_shift_count: int = 0              # number of tone transitions

    # ── Interaction pattern detection ─────────────────────────────────────
    interaction_pattern: str = "normal"     # normal | repetitive | testing | exploratory | rapid_fire
    testing_flag: bool = False              # user appears to be testing the system
    repetition_count: int = 0              # consecutive similar queries
    meta_comment_count: int = 0            # "you're an AI", "can you really…", etc.

    # ── Intent history ────────────────────────────────────────────────────
    last_intent: str = ""                  # previous message's intent
    intent_history: list = field(default_factory=list)  # last N intents
    intent_streak: int = 0                 # consecutive same-intent messages

    # ── Engagement metrics ────────────────────────────────────────────────
    message_count: int = 0                 # total messages in conversation
    avg_query_length: float = 0.0          # rolling average word count
    short_query_streak: int = 0            # consecutive very short queries

    # ── Personality mode ──────────────────────────────────────────────────
    dynamic_personality_mode: str = "default"  # default | concise | detailed | playful | empathetic

    # ── Timestamps ────────────────────────────────────────────────────────
    last_update: float = 0.0               # time.time() of last update
    conversation_start: float = 0.0        # time.time() of first message

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for DB storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ConversationState:
        """Deserialize from DB-stored dict."""
        if not data:
            return cls()
        # Filter to only known fields to handle schema evolution
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ═══════════════════════════════════════════════════════════════════════════
#  IN-MEMORY STATE CACHE
# ═══════════════════════════════════════════════════════════════════════════

_state_cache: dict[str, ConversationState] = {}
_MAX_CACHE_SIZE = 200
_cache_lock = threading.Lock()


def get_or_create_state(conversation_id: str) -> ConversationState:
    """Get state from cache, or create a fresh one.

    DB persistence is handled separately via save_state_to_db().
    """
    with _cache_lock:
        if conversation_id in _state_cache:
            return _state_cache[conversation_id]

        state = ConversationState(
            conversation_start=time.time(),
            last_update=time.time(),
        )
        _evict_if_needed()
        _state_cache[conversation_id] = state
        return state


def set_state(conversation_id: str, state: ConversationState) -> None:
    """Store/update state in cache."""
    with _cache_lock:
        _evict_if_needed()
        _state_cache[conversation_id] = state


def _evict_if_needed() -> None:
    """LRU-style eviction when cache exceeds max size. Caller holds _cache_lock."""
    if len(_state_cache) >= _MAX_CACHE_SIZE:
        # Evict the oldest-updated state
        oldest_cid = min(_state_cache, key=lambda k: _state_cache[k].last_update)
        del _state_cache[oldest_cid]


def clear_state(conversation_id: str) -> None:
    """Remove state from cache (e.g. on conversation delete)."""
    _state_cache.pop(conversation_id, None)


# ═══════════════════════════════════════════════════════════════════════════
#  STATE TRACKER — pattern detection + state updates
# ═══════════════════════════════════════════════════════════════════════════

# ── Tone / emotion signals ────────────────────────────────────────────────

_POSITIVE_SIGNALS = {
    "thanks", "thank you", "awesome", "great", "perfect", "love it",
    "wonderful", "excellent", "amazing", "nice", "cool", "fantastic",
    "brilliant", "helpful", "appreciate",
}

_FRUSTRATED_SIGNALS = {
    "wrong", "incorrect", "not what i asked", "that's not right",
    "you're wrong", "no that's wrong", "still wrong", "try again",
    "that doesn't work", "doesn't help", "useless", "not helpful",
    "i already said", "i told you", "not what i meant", "frustrated",
    "annoying", "broken",
}

_CURIOUS_SIGNALS = {
    "how does", "why does", "what if", "can you explain", "tell me more",
    "i wonder", "interesting", "elaborate", "what about", "how come",
    "curious", "dig deeper",
}

_PLAYFUL_SIGNALS = {
    "haha", "lol", "lmao", "😂", "🤣", "just kidding", "joke",
    "funny", "rofl", "😄", "😁", ":)", "xd",
}

# ── Testing / meta signals ───────────────────────────────────────────────

_TESTING_SIGNALS = [
    "are you an ai", "are you a bot", "are you real", "are you human",
    "what model are you", "what llm", "are you chatgpt", "are you gpt",
    "can you think", "do you have feelings", "are you sentient",
    "prove you're", "test", "testing", "let me test", "i'm testing",
    "can you really", "do you actually", "are you sure about that",
    "i don't believe you", "prove it",
]

_META_SIGNALS = [
    "you're an ai", "you're a bot", "you're just a", "you can't really",
    "you don't actually", "you're not real", "you're a machine",
    "you're a program", "you're software", "your creators",
    "who made you", "who built you", "your training data",
    "your parameters", "your model", "your weights",
]

# ── Repetition detection ─────────────────────────────────────────────────

# Max history length for pattern tracking — use settings value if available,
# fall back to a sensible default for testing without full app context.
try:
    from settings import settings as _settings
    _PATTERN_WINDOW: int = _settings.BEHAVIOR_PATTERN_WINDOW
except Exception:
    _PATTERN_WINDOW = 10


class StateTracker:
    """Stateless analyzer that computes state transitions.

    All detection methods are pure functions: (state, new_data) → updated_state.
    No side effects, no DB access, no caching.
    """

    @staticmethod
    def update(
        state: ConversationState,
        query: str,
        intent: str,
        confidence: float,
        recent_queries: list[str] | None = None,
    ) -> ConversationState:
        """Apply a new user message to the conversation state.

        Args:
            state:          Current conversation state (will be mutated).
            query:          The new user query text.
            intent:         Classified intent for this query.
            confidence:     Intent classification confidence.
            recent_queries: Last N user queries (for repetition detection).

        Returns:
            The same state object, updated in-place.
        """
        now = time.time()
        q_lower = query.strip().lower()
        words = q_lower.split()
        word_count = len(words)

        # ── Update basic counters ─────────────────────────────────────────
        state.message_count += 1
        state.last_update = now
        if state.conversation_start == 0.0:
            state.conversation_start = now

        # Rolling average query length
        prev_total = state.avg_query_length * max(state.message_count - 1, 1)
        state.avg_query_length = (prev_total + word_count) / state.message_count

        # Short query streak
        if word_count <= 3:
            state.short_query_streak += 1
        else:
            state.short_query_streak = 0

        # ── Intent tracking ───────────────────────────────────────────────
        state.intent_history.append(intent)
        if len(state.intent_history) > _PATTERN_WINDOW:
            state.intent_history = state.intent_history[-_PATTERN_WINDOW:]

        if intent == state.last_intent:
            state.intent_streak += 1
        else:
            state.intent_streak = 1
        state.last_intent = intent

        # ── Tone detection ────────────────────────────────────────────────
        new_tone = StateTracker._detect_tone(q_lower)
        if new_tone != state.emotional_tone:
            state.tone_shift_count += 1
        state.emotional_tone = new_tone

        # ── Repetition detection ──────────────────────────────────────────
        if recent_queries:
            state.repetition_count = StateTracker._detect_repetition(
                q_lower, recent_queries,
            )
        elif state.intent_streak >= 3:
            state.repetition_count = state.intent_streak

        # ── Testing / meta detection ──────────────────────────────────────
        is_testing = StateTracker._detect_testing(q_lower)
        is_meta = StateTracker._detect_meta(q_lower)
        if is_testing:
            state.testing_flag = True
            state.meta_comment_count += 1
        elif is_meta:
            state.meta_comment_count += 1
        elif state.testing_flag and state.message_count > 3:
            # Reset testing flag after user moves on
            state.testing_flag = False

        # ── Interaction pattern classification ────────────────────────────
        state.interaction_pattern = StateTracker._classify_pattern(state)

        # ── Precision mode (research-oriented) ─────────────────────────
        state.precision_mode = StateTracker._compute_precision_mode(state, q_lower)

        return state

    # ── Private detection methods ─────────────────────────────────────────

    @staticmethod
    def _detect_tone(q_lower: str) -> str:
        """Classify emotional tone of a query."""
        # Check in order of specificity
        if any(sig in q_lower for sig in _FRUSTRATED_SIGNALS):
            return "frustrated"
        if any(sig in q_lower for sig in _PLAYFUL_SIGNALS):
            return "playful"
        if any(sig in q_lower for sig in _CURIOUS_SIGNALS):
            return "curious"
        if any(sig in q_lower for sig in _POSITIVE_SIGNALS):
            return "positive"
        return "neutral"

    @staticmethod
    def _detect_testing(q_lower: str) -> bool:
        """Detect if user is testing/probing the system."""
        return any(sig in q_lower for sig in _TESTING_SIGNALS)

    @staticmethod
    def _detect_meta(q_lower: str) -> bool:
        """Detect meta-comments about the AI."""
        return any(sig in q_lower for sig in _META_SIGNALS)

    @staticmethod
    def _detect_repetition(
        current_query: str,
        recent_queries: list[str],
        threshold: float = 0.7,
    ) -> int:
        """Count how many recent queries are semantically similar.

        Uses cheap word-overlap (Jaccard) — not embeddings — since this
        runs every message and needs to be <1ms.
        """
        if not recent_queries:
            return 0

        current_words = set(current_query.split())
        if not current_words:
            return 0

        count = 0
        for prev in recent_queries[-5:]:  # check last 5 only
            prev_words = set(prev.strip().lower().split())
            if not prev_words:
                continue
            intersection = current_words & prev_words
            union = current_words | prev_words
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= threshold:
                count += 1

        return count

    @staticmethod
    def _classify_pattern(state: ConversationState) -> str:
        """Classify the overall interaction pattern."""
        if state.testing_flag or state.meta_comment_count >= 2:
            return "testing"

        if state.repetition_count >= 3:
            return "repetitive"

        if state.short_query_streak >= 4:
            return "rapid_fire"

        # Diverse intents in recent history → exploratory
        if len(state.intent_history) >= 4:
            unique_recent = len(set(state.intent_history[-4:]))
            if unique_recent >= 3:
                return "exploratory"

        return "normal"

    @staticmethod
    def _compute_precision_mode(state: ConversationState, q_lower: str = "") -> str:
        """Select research precision mode based on structural signals.

        Modes:
          - concise:          Short, direct answers (rapid-fire, short queries)
          - analytical:       Thorough, structured analysis (default for research)
          - speculative:      Hypothesis exploration ("what if", conditional)
          - implementation:   Code/build-focused (technical terms, code signals)
          - adversarial:      Challenge assumptions ("but", "however", "wrong")
        """
        # Adversarial: user is pushing back or challenging
        _adversarial_signals = {
            "wrong", "incorrect", "disagree", "but what about",
            "that's not right", "you're wrong", "however",
            "counterpoint", "devil's advocate", "push back",
            "challenge", "flaw", "problem with that",
        }
        if any(sig in q_lower for sig in _adversarial_signals):
            return "adversarial"

        # Implementation: code/build signals
        _impl_signals = {
            "implement", "code", "build", "create", "write",
            "function", "class", "module", "api", "endpoint",
            "database", "deploy", "configure", "setup", "install",
            "dockerfile", "script", "refactor", "debug", "fix",
        }
        if any(sig in q_lower for sig in _impl_signals):
            return "implementation"

        # Speculative: hypothesis / conditional framing
        _speculative_signals = {
            "what if", "hypothetically", "suppose", "imagine",
            "could we", "would it", "might", "possibly",
            "speculate", "theory", "assume", "scenario",
        }
        if any(sig in q_lower for sig in _speculative_signals):
            return "speculative"

        # Concise: rapid-fire or very short queries
        if state.interaction_pattern == "rapid_fire":
            return "concise"
        if state.short_query_streak >= 3:
            return "concise"

        # Default: analytical
        return "analytical"
