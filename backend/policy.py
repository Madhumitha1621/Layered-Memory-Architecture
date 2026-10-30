"""Behavior Policy Engine — deterministic context-injection rules.

Instead of intent alone determining what gets injected into the LLM
context, the policy engine computes *context features* and applies
deterministic rules to decide:

* What context to inject  (profile, RAG, Q&A, history)
* How to frame it  (privacy mode, greeting personalization)
* Which retrieval route to label  (for metadata / debugging)

This separates BEHAVIOR from MODEL-CALLING code.  When behavior is wrong
(e.g. greeting doesn't use the user's name), you fix a rule here — you
never edit prompt strings or generator functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL LISTS
# ═══════════════════════════════════════════════════════════════════════════

GREETING_PATTERNS = [
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "sup", "what's up",
    "whats up", "yo", "hola", "greetings", "hi there",
    "hey there", "hello there",
]

PERSONAL_REF_SIGNALS = [
    "my job", "my role", "my name", "my work", "my career",
    "my background", "my preference", "my language", "my stack",
    "my project", "my company", "what do i do", "what am i",
    "who am i", "i work as", "my position", "my skills",
    "my experience", "for my job", "in my job", "on my job",
    "at my job", "my profession", "my title", "whats my",
    "what's my", "tell me my", "what is my", "my weight",
    "my height", "my age", "my location", "my timezone",
    "my degree", "my education",
]

PROFILE_STATEMENT_PREFIXES = (
    "i am ", "i'm ", "i have ", "i like ", "i prefer ", "i use ",
    "i work ", "i live ", "i weigh ", "i speak ", "i study ",
    "i graduated ", "i code ", "i built ", "i've ", "i am from ",
    "call me ", "my name is ", "remember that ", "i am a ", "i'm a ",
)


# ═══════════════════════════════════════════════════════════════════════════
#  CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContextFeatures:
    """Computed features for the current message — input to policy rules."""

    query: str = ""                         # original user query
    is_greeting: bool = False
    references_profile: bool = False
    privacy_signal: bool = False
    is_followup: bool = False
    is_profile_statement: bool = False
    is_profile_question: bool = False
    topic_similarity: float | None = None
    has_profile_data: bool = False
    profile_name: str | None = None
    conversation_length: int = 0
    structural_followup_score: float = 0.0  # 0.0-1.0 structural continuation signal


def extract_context_features(
    query: str,
    intent: str,
    profile_entries: list[dict] | None = None,
    conversation_length: int = 0,
    topic_similarity: float | None = None,
) -> ContextFeatures:
    """Compute deterministic context features from current state."""
    q = query.strip().lower()
    words = q.split()

    # ── Structural follow-up detection ────────────────────────────────────
    structural_score = _compute_structural_followup_score(q, words, conversation_length)

    # ── Greeting detection ────────────────────────────────────────────────
    is_greeting = False
    if len(words) <= 8:
        for pat in GREETING_PATTERNS:
            if q == pat or q.startswith(pat + " ") or q.startswith(pat + ",") or q.startswith(pat + "!"):
                is_greeting = True
                break

    # ── Personal-reference detection ──────────────────────────────────────
    references_profile = any(sig in q for sig in PERSONAL_REF_SIGNALS)

    # ── Profile statement vs question ─────────────────────────────────────
    is_profile_statement = (
        q.startswith(PROFILE_STATEMENT_PREFIXES) and "?" not in query
    )
    is_profile_question = intent == "profile" and not is_profile_statement

    # ── Profile data check ────────────────────────────────────────────────
    has_profile_data = bool(profile_entries)
    profile_name: str | None = None
    if profile_entries:
        for entry in profile_entries:
            if entry.get("key") in ("name", "first_name", "full_name", "username"):
                profile_name = entry.get("value")
                break

    return ContextFeatures(
        query=query,
        is_greeting=is_greeting,
        references_profile=references_profile,
        privacy_signal=(intent == "privacy"),
        is_followup=(intent == "continuation") or structural_score >= 0.5,
        is_profile_statement=is_profile_statement,
        is_profile_question=is_profile_question,
        topic_similarity=topic_similarity,
        has_profile_data=has_profile_data,
        profile_name=profile_name,
        conversation_length=conversation_length,
        structural_followup_score=structural_score,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  POLICY DECISION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyDecision:
    """What the pipeline should do — determined by rules, not prompts."""

    inject_profile: bool = False
    inject_rag: bool = False
    inject_qa_history: bool = False
    use_curated_history: bool = True
    privacy_mode: bool = False
    greeting_name: str | None = None       # if set → personalise greeting
    retrieval_route: str = "llm_only"      # label for metadata
    rag_k: int = 4
    rag_min_similarity: float = 0.0        # relevance floor for KB docs
    qa_k: int = 4
    qa_min_similarity: float = 0.65


# ═══════════════════════════════════════════════════════════════════════════
#  BEHAVIOR POLICY
# ═══════════════════════════════════════════════════════════════════════════

class BehaviorPolicy:
    """Deterministic policy engine — rules, not prompt rewrites.

    Usage::

        features = extract_context_features(query, intent, profile_entries, …)
        decision = BehaviorPolicy().resolve(features, intent)

    The returned :class:`PolicyDecision` tells the pipeline what to
    retrieve, what to inject, and how to frame the response.
    """

    def resolve(self, features: ContextFeatures, intent: str) -> PolicyDecision:
        d = PolicyDecision()

        # ── Base intent rules ─────────────────────────────────────────────
        # Using if/elif/else (not early returns) so cross-intent overlays
        # below ALWAYS run regardless of which branch was taken.

        if intent == "privacy":
            d.privacy_mode = True
            d.inject_profile = features.has_profile_data
            d.use_curated_history = False
            d.retrieval_route = "privacy"

        elif intent == "profile":
            if features.is_profile_statement:
                d.retrieval_route = "profile_update"
                d.use_curated_history = False
            else:
                d.inject_profile = features.has_profile_data
                d.retrieval_route = "profile"

        elif intent == "knowledge_base":
            d.inject_rag = True
            d.inject_qa_history = True
            d.retrieval_route = "rag"

        elif intent == "continuation":
            d.inject_rag = True
            d.inject_qa_history = True
            d.rag_min_similarity = 0.35
            d.retrieval_route = "conversation"

        else:  # general
            d.inject_rag = True
            d.rag_min_similarity = 0.45
            d.retrieval_route = "adaptive"

        # ── Cross-intent overlays (always apply after base rules) ─────────

        # 1. Name injection — whenever we know the user's name and are NOT
        #    already injecting the full profile (which contains the name),
        #    inject a lightweight name-context frame so the LLM can use it
        #    in greetings, continuations, and knowledge responses alike.
        if features.profile_name and not d.inject_profile:
            d.greeting_name = features.profile_name

        # 2. Personal-reference → inject full profile for any intent
        if (
            features.references_profile
            and features.has_profile_data
            and not d.inject_profile
            and intent != "privacy"   # privacy already injects
        ):
            d.inject_profile = True
            # name is now in the full profile; remove the lightweight frame
            d.greeting_name = None

        return d


# ═══════════════════════════════════════════════════════════════════════════
#  STRUCTURAL FOLLOW-UP DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Pronoun dependency — these only make sense if there's prior context
_PRONOUN_DEPS = re.compile(
    r"\b(it|its|that|those|this|these|they|them|the same|the above|"
    r"the previous|the one|the other|said|mentioned|described)\b",
    re.IGNORECASE,
)

# Conditional / continuation starters
_CONTINUATION_STARTERS = [
    "what if", "but what if", "so if", "but then", "and then",
    "but what about", "what about", "how about", "and what about",
    "so then", "but why", "but how", "and how", "then how",
    "ok but", "okay but", "yeah but", "right but",
    "also", "additionally", "furthermore", "moreover",
    "on the other hand", "alternatively", "conversely",
]

# Variable-reference patterns (common in code/research discussions)
_VARIABLE_REF = re.compile(
    r"\b(the (?:function|method|class|variable|table|column|field|param|"
    r"endpoint|query|result|output|error|issue|bug|problem|solution))\b",
    re.IGNORECASE,
)

# Elaboration requests
_ELABORATION_SIGNALS = [
    "elaborate", "more detail", "tell me more", "explain further",
    "go deeper", "expand on", "can you clarify", "more about",
    "in more depth", "specifically", "in particular",
    "what do you mean", "unpack that", "break that down",
]

# Very short follow-ups (must have conversation context)
_SHORT_FOLLOWUP = re.compile(r"^(why|how|when|where|and|so|really|seriously)\??\s*$", re.IGNORECASE)


def _compute_structural_followup_score(q: str, words: list[str], conversation_length: int) -> float:
    """Compute a 0.0-1.0 structural follow-up score.

    This uses syntactic patterns — NOT embeddings — to detect messages
    that structurally depend on prior context. This catches follow-ups
    that the intent classifier might miss.

    Signals (weighted):
      - Pronoun dependency (it/that/those/this)  → 0.3
      - Continuation starters (what if/but then)  → 0.4
      - Variable references (the function/the error) → 0.3
      - Elaboration requests                      → 0.4
      - Very short follow-ups (< 4 words)         → 0.3
    """
    if conversation_length == 0:
        return 0.0  # No prior context → can't be a follow-up

    q_lower = q.strip().lower()
    score = 0.0

    # Pronoun dependency
    if _PRONOUN_DEPS.search(q_lower):
        score += 0.3

    # Continuation starters
    if any(q_lower.startswith(s) for s in _CONTINUATION_STARTERS):
        score += 0.4

    # Variable references
    if _VARIABLE_REF.search(q_lower):
        score += 0.3

    # Elaboration requests
    if any(sig in q_lower for sig in _ELABORATION_SIGNALS):
        score += 0.4

    # Very short follow-ups in active conversation
    if len(words) <= 3 and conversation_length >= 2:
        if _SHORT_FOLLOWUP.match(q_lower) or q_lower.endswith("?"):
            score += 0.3

    return min(score, 1.0)
