"""Document vector store — pgvector (persistent) with in-memory fallback.

Single vector layer: PostgreSQL + pgvector.
When the database is unavailable, falls back to numpy cosine similarity
over an in-memory list.  No FAISS dependency.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────
_db_available = False
_fallback_docs: list[str] = []
_fallback_embeddings: list[np.ndarray] = []


def init(db_enabled: bool) -> None:
    """Called from main.py startup to set the storage backend."""
    global _db_available
    _db_available = db_enabled
    if _db_available:
        logger.info("Vector store: pgvector (persistent)")
    else:
        logger.info("Vector store: in-memory fallback (non-persistent)")


def add_documents(text_chunks: list[str], source: str = "default") -> int:
    """Index document chunks.  Returns the number of chunks stored."""
    if not text_chunks:
        return 0

    if _db_available:
        import query_db
        query_db.store_document_chunks(text_chunks, source=source)
    else:
        from embeddings import get_embeddings
        embeddings = get_embeddings(text_chunks)
        for chunk, emb in zip(text_chunks, embeddings):
            _fallback_docs.append(chunk)
            _fallback_embeddings.append(emb)

    logger.info(f"Indexed {len(text_chunks)} chunks (source={source})")
    return len(text_chunks)


def _cosine_similarities(query_emb: np.ndarray, embeddings: list[np.ndarray]) -> list[float]:
    """Compute cosine similarity between a query and a list of embeddings."""
    sims: list[float] = []
    for emb in embeddings:
        dot = float(np.dot(query_emb, emb))
        norm = float(np.linalg.norm(query_emb) * np.linalg.norm(emb))
        sims.append(dot / norm if norm > 0 else 0.0)
    return sims


def search(query: str, k: int | None = None, min_similarity: float = 0.0) -> list[str]:
    """Semantic search over indexed documents.  Returns chunk texts.

    Results below *min_similarity* are excluded to prevent irrelevant
    context from being injected when the user's intent is ambiguous.

    When HYBRID_SEARCH_ENABLED, uses BM25 + vector RRF fusion.
    When RERANKER_ENABLED, applies cross-encoder reranking.
    """
    if k is None:
        from settings import settings
        k = settings.RETRIEVAL_K

    if _db_available:
        from settings import settings
        if settings.HYBRID_SEARCH_ENABLED or settings.RERANKER_ENABLED:
            # Delegate to search_with_scores and strip scores
            return [text for text, _score in search_with_scores(query, k, min_similarity)]
        import query_db
        from embeddings import get_query_embedding
        embedding = get_query_embedding(query)
        return query_db.search_document_chunks(embedding, k=k, min_similarity=min_similarity)
    else:
        from embeddings import get_query_embedding
        if not _fallback_docs:
            return []
        qe = get_query_embedding(query)
        sims = _cosine_similarities(qe, _fallback_embeddings)
        topk = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [_fallback_docs[i] for i in topk if sims[i] >= min_similarity]


def search_with_scores(
    query: str,
    k: int | None = None,
    min_similarity: float = 0.0,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
) -> list[tuple[str, float]]:
    """Semantic search returning (text, similarity) tuples.

    Same as search() but preserves cosine similarity scores for
    retrieval quality analysis and telemetry.

    When HYBRID_SEARCH_ENABLED, uses BM25 + vector RRF fusion.
    When RERANKER_ENABLED, applies cross-encoder reranking.

    Args:
        use_hybrid:   Override HYBRID_SEARCH_ENABLED for this call.
        use_reranker: Override RERANKER_ENABLED for this call.
    """
    if k is None:
        from settings import settings
        k = settings.RETRIEVAL_K

    if _db_available:
        import query_db
        from embeddings import get_query_embedding
        from settings import settings
        embedding = get_query_embedding(query)

        hybrid_on = use_hybrid if use_hybrid is not None else settings.HYBRID_SEARCH_ENABLED
        reranker_on = use_reranker if use_reranker is not None else settings.RERANKER_ENABLED

        # ── Hybrid Search ─────────────────────────────────────
        if hybrid_on:
            from hybrid_search import hybrid_search, HybridConfig
            conn = query_db.get_connection()
            try:
                hybrid_config = HybridConfig(
                    rrf_k=settings.HYBRID_RRF_K,
                    vector_weight=settings.HYBRID_VECTOR_WEIGHT,
                    bm25_weight=settings.HYBRID_BM25_WEIGHT,
                    candidate_multiplier=settings.HYBRID_CANDIDATE_MULTIPLIER,
                    vector_min_similarity=min_similarity,
                )
                # Fetch more candidates for reranking
                candidate_k = k * 3 if reranker_on else k
                results = hybrid_search(
                    conn, query, embedding, k=candidate_k, config=hybrid_config,
                )
            finally:
                query_db.put_connection(conn)
        else:
            # Pure vector search
            candidate_k = k * 3 if reranker_on else k
            results = query_db.search_document_chunks_with_scores(
                embedding, k=candidate_k, min_similarity=min_similarity,
            )

        # ── Reranking ─────────────────────────────────────────
        if reranker_on and results:
            from reranker import rerank
            texts = [text for text, _score in results]
            reranked = rerank(query, texts, top_k=k)
            results = reranked
        else:
            results = results[:k]

        # ── Normalize scores to cosine similarity ─────────────
        # Hybrid returns RRF scores, reranker returns cross-encoder logits.
        # Normalize all to cosine similarity for fair cross-arm comparison.
        if hybrid_on or reranker_on:
            vector_lookup = query_db.search_document_chunks_with_scores(
                embedding, k=max(k * 5, 50), min_similarity=0.0,
            )
            cos_map = {text: sim for text, sim in vector_lookup}
            results = [(text, cos_map.get(text, 0.0)) for text, _score in results]
            # BM25-only hits bypass the vector threshold inside hybrid search
            results = [(text, sim) for text, sim in results if sim >= min_similarity]

        return results
    else:
        from embeddings import get_query_embedding
        if not _fallback_docs:
            return []
        qe = get_query_embedding(query)
        sims = _cosine_similarities(qe, _fallback_embeddings)
        topk = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [(_fallback_docs[i], sims[i]) for i in topk if sims[i] >= min_similarity]


def has_documents() -> bool:
    """Check if any documents are indexed."""
    if _db_available:
        import query_db
        return query_db.count_document_chunks() > 0
    return len(_fallback_docs) > 0


def count() -> int:
    """Return the number of indexed document chunks."""
    if _db_available:
        import query_db
        return query_db.count_document_chunks()
    return len(_fallback_docs)


def clear(source: str | None = None) -> None:
    """Remove indexed documents (optionally by source)."""
    if _db_available:
        import query_db
        query_db.clear_document_chunks(source=source)
    else:
        _fallback_docs.clear()
        _fallback_embeddings.clear()
    logger.info(f"Cleared document index{f' (source={source})' if source else ''}")
