"""Topic Threading Engine — multi-thread conversation graph.

Instead of a single topic vector per conversation, topics are tracked
as a *graph of threads*.  Each thread has:

  - A centroid embedding (exponential moving average of message embeddings)
  - A list of attached message IDs
  - A progressive summary
  - A human-readable label

When a new message arrives, the engine:

  1. Compares its embedding against all thread centroids.
  2. If similarity ≥ THREAD_ATTACH_THRESHOLD → attaches to that thread.
  3. Otherwise → creates a new thread.
  4. Updates the centroid via EMA.

This replaces the old single-`topic_vector` model with a richly
structured thread graph — the foundation of the research engine.

Public API:
    resolve_thread()   — find or create the best thread for a message
    get_active_thread() — return the currently active thread
    ThreadResolution   — result of thread resolution
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD RESOLUTION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThreadResolution:
    """Result of resolving which thread a message belongs to."""

    thread_id: str
    is_new: bool = False
    similarity: float = 0.0
    thread_summary: str = ""
    thread_label: str = ""
    message_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  CENTROID MATH
# ═══════════════════════════════════════════════════════════════════════════

def _ema_centroid(
    old_centroid: np.ndarray,
    new_embedding: np.ndarray,
    message_count: int,
    alpha: float = 0.3,
) -> np.ndarray:
    """Exponential moving average update for thread centroid.

    Early messages (count < 4) use simple mean for stability.
    After that, EMA with configurable alpha provides momentum.
    """
    if message_count <= 1:
        return new_embedding.copy()

    if message_count < 4:
        # Simple cumulative average for first few messages
        weight = 1.0 / message_count
        updated = old_centroid * (1 - weight) + new_embedding * weight
    else:
        updated = old_centroid * (1 - alpha) + new_embedding * alpha

    # L2 normalize to keep cosine similarity meaningful
    norm = np.linalg.norm(updated)
    if norm > 0:
        updated = updated / norm
    return updated


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

def resolve_thread(
    conversation_id: str,
    query_embedding: np.ndarray,
    message_id: str | None = None,
    *,
    db_enabled: bool = True,
) -> ThreadResolution:
    """Find or create the best thread for a new message.

    Steps:
      1. Query DB for existing threads' centroids.
      2. Compute cosine similarity against each.
      3. If best match ≥ threshold → attach to that thread.
      4. Otherwise → create a new thread.
      5. Update centroid via EMA.

    Returns a ThreadResolution with all metadata.
    """
    import query_db

    if not settings.THREAD_ENABLED or not db_enabled:
        return ThreadResolution(thread_id="", is_new=False)

    threshold = settings.THREAD_ATTACH_THRESHOLD
    q_emb = np.asarray(query_embedding, dtype=np.float32).flatten()

    # Try to find the nearest existing thread
    nearest = query_db.find_nearest_thread(conversation_id, q_emb, threshold=threshold)

    if nearest:
        # Attach to existing thread
        thread_id = nearest["thread_id"]
        similarity = nearest["similarity"]
        thread_data = query_db.get_thread(thread_id)

        if thread_data:
            # Update centroid
            old_centroid = np.asarray(thread_data["centroid_embedding"], dtype=np.float32)
            new_count = thread_data["message_count"] + 1
            new_centroid = _ema_centroid(old_centroid, q_emb, new_count)
            query_db.update_thread_centroid(thread_id, new_centroid, message_id)

            logger.info(
                f"Thread attach: {thread_id[:8]}… (sim={similarity:.3f}, "
                f"msgs={new_count})"
            )
            return ThreadResolution(
                thread_id=thread_id,
                is_new=False,
                similarity=similarity,
                thread_summary=thread_data.get("summary", ""),
                thread_label=thread_data.get("label", ""),
                message_count=new_count,
            )

    # No match — create a new thread
    thread_id = str(uuid.uuid4())

    # Check if we're at max threads; if so, still create (merging is future work)
    thread_count = query_db.count_threads(conversation_id)
    if thread_count >= settings.THREAD_MAX_ACTIVE:
        logger.warning(
            f"Thread limit reached ({thread_count}/{settings.THREAD_MAX_ACTIVE}) "
            f"for conversation {conversation_id}"
        )

    # Normalize the initial centroid
    norm = np.linalg.norm(q_emb)
    initial_centroid = q_emb / norm if norm > 0 else q_emb

    # conversation_threads.conversation_id references conversations(id), and the
    # conversation row is otherwise only created later, in the persist step.
    query_db.ensure_conversation_exists(conversation_id)
    created = query_db.create_thread(
        thread_id=thread_id,
        conversation_id=conversation_id,
        centroid_embedding=initial_centroid,
        label="",
    )
    if not created:
        # Don't hand out an ID for a thread that doesn't exist: insights and
        # concepts tagged with it would fail their own FK and be lost.
        logger.warning(f"Thread creation failed for conversation {conversation_id[:8]}…")
        return ThreadResolution(thread_id="", is_new=False)

    logger.info(f"New thread: {thread_id[:8]}… (conv={conversation_id[:8]}…)")
    return ThreadResolution(
        thread_id=thread_id,
        is_new=True,
        similarity=0.0,
        thread_summary="",
        thread_label="",
        message_count=1,
    )


def get_thread_context(conversation_id: str, thread_id: str) -> dict:
    """Get enriched context for the active thread for prompt injection.

    Returns a dict with summary, recent insights, and related concepts.
    """
    import query_db

    if not thread_id:
        return {}

    thread = query_db.get_thread(thread_id)
    if not thread:
        return {}

    insights = query_db.get_insights_for_thread(thread_id, limit=5)

    return {
        "thread_id": thread_id,
        "thread_summary": thread.get("summary", ""),
        "thread_label": thread.get("label", ""),
        "message_count": thread.get("message_count", 0),
        "recent_insights": [
            {"type": i["insight_type"], "text": i["insight_text"]}
            for i in insights
        ],
    }


def should_summarize_thread(thread_id: str) -> bool:
    """Check if a thread needs a summary update (based on message count)."""
    import query_db

    thread = query_db.get_thread(thread_id)
    if not thread:
        return False

    count = thread.get("message_count", 0)
    interval = settings.THREAD_SUMMARY_INTERVAL

    # Summarize at interval milestones (8, 16, 24, …)
    return count > 0 and count % interval == 0
