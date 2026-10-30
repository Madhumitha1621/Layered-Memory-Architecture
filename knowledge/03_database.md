# Database Schema & Query Functions — query_db.py

## Connection Management

The database layer uses psycopg2.pool.ThreadedConnectionPool. Configuration from settings.py — DATABASE_URL overrides individual POSTGRES_* settings. Pool initialized lazily via _get_pool(). Min: DB_POOL_MIN (2). Max: DB_POOL_MAX (10).

init_db() creates all 9 tables using CREATE TABLE IF NOT EXISTS plus indexes. Runs migrations for columns added after initial schema. Returns True on success, False if unreachable. If False, application runs in in-memory mode (no persistence).

## Schema — Nine Tables

### conversations
Primary key: id (TEXT, UUID). Columns: title (TEXT), user_id (TEXT), tags (TEXT[]), topic_embedding (vector(768)), created_at (TIMESTAMPTZ), updated_at (TIMESTAMPTZ).
The topic_embedding stores the rolling conversation topic vector, updated every message using EMA (α=TOPIC_DECAY_ALPHA=0.2).
Indexes: HNSW on topic_embedding for vector search.

### chat_messages
Primary key: id (SERIAL). FK: conversation_id → conversations(id) ON DELETE CASCADE. Columns: role (TEXT), content (TEXT), timestamp (TIMESTAMPTZ).

### user_queries
Primary key: id (SERIAL). FK: conversation_id → conversations(id) ON DELETE CASCADE. Columns: query_text (TEXT), ai_response (TEXT), embedding (vector(768)), tags (TEXT[]), timestamp (TIMESTAMPTZ).
Enables semantic Q&A search across conversations. HNSW index on embedding.

### user_profile
Primary key: id (SERIAL). Columns: user_id (TEXT), key (TEXT), value (TEXT), category (TEXT), updated_at (TIMESTAMPTZ). UNIQUE(user_id, key) enables upsert logic.
Categories: personal, professional, preferences, health, education, other.

### document_chunks
Primary key: id (SERIAL). Columns: content (TEXT), embedding (vector(768)), source (TEXT), created_at (TIMESTAMPTZ).
IS the knowledge base vector store. HNSW index on embedding.

### conversation_state (v6.0.0)
Primary key: conversation_id (TEXT, FK → conversations ON DELETE CASCADE). Columns: state_data (JSONB), updated_at (TIMESTAMP).
Stores ConversationState dataclass as JSON: tone, repetition, testing flags, interaction patterns, precision mode.

### conversation_threads (v6.0.0)
Primary key: id (TEXT). Columns: conversation_id (TEXT), centroid_embedding (vector(768)), message_ids (TEXT[]), message_count (INT), summary (TEXT), label (TEXT), last_active (TIMESTAMPTZ), created_at (TIMESTAMPTZ).
HNSW index on centroid_embedding for nearest-thread resolution.

### research_insights (v6.0.0)
Primary key: id (SERIAL). Columns: conversation_id (TEXT), thread_id (TEXT), insight_type (TEXT), insight_text (TEXT), embedding (vector(768)), confidence_score (FLOAT), source_message_id (TEXT), created_at (TIMESTAMPTZ).
Types: decision, conclusion, hypothesis, open_question, observation. HNSW index on embedding.

### concept_links (v6.0.0)
Primary key: id (SERIAL). Columns: concept (TEXT), embedding (vector(768)), source_type (TEXT), source_id (TEXT), conversation_id (TEXT), thread_id (TEXT), created_at (TIMESTAMPTZ).
HNSW index on embedding for semantic concept search.

## Public Functions (52 total)

### Connection (2)
- get_connection() — checkout from pool
- put_connection(conn) — return with defensive rollback

### Schema (1)
- init_db() — create all tables + extensions + indexes

### Conversations (11)
- create_conversation(title) — UUID
- list_conversations(limit=50) — list[dict] by updated_at DESC
- get_conversation(cid) — dict | None
- rename_conversation(cid, title) — dict | None
- delete_conversation(cid) — cascade deletes messages, queries, state, threads, insights, concepts
- touch_conversation(cid) — update updated_at timestamp
- increment_message_count(cid, n) — increment message_count
- search_conversations(query, limit=20) — ILIKE search
- export_conversation(cid) — full conversation export as dict
- get_conversation_messages(cid, limit=200) — chronological messages
- get_recent_chat_messages(cid, limit=10) — last k messages for pipeline

### Messages (3)
- store_chat_message(role, content, cid) — insert message
- get_first_user_message(cid) — for auto-titling
- delete_last_assistant_message(cid) — for regenerate

### Queries / Semantic Search (5)
- store_query(query_text, embedding, response, cid, tags) — save Q&A pair with vector
- retrieve_similar_queries(embedding, k, cid, min_similarity) — cross-conversation semantic Q&A search
- retrieve_same_conversation_queries(embedding, cid, k, min_similarity) — same-conversation Q&A
- get_similar_messages_in_conversation(embedding, cid, k, min_similarity) — for semantic history augmentation
- infer_tags(query_text) — keyword-based tag inference

### Profile (3)
- get_user_profile(user_id) — list[dict]
- update_profile_entry(key, value, category, user_id) — UPSERT
- delete_profile_entry(entry_id) — by integer id

### Topic Vectors (2)
- get_topic_vector(cid) — numpy float32 array | None
- update_topic_vector(cid, embedding, alpha=0.1) — EMA blend (main.py passes α=TOPIC_DECAY_ALPHA=0.2)

### Document Chunks (4)
- store_document_chunks(chunks, source) — bulk insert (text, embedding) tuples
- search_document_chunks(embedding, k=4, min_similarity=0.0) — pgvector cosine search
- count_document_chunks() — row count
- clear_document_chunks(source=None) — delete all or by source

### Conversation State (3) — v6.0.0
- get_conversation_state(cid) — JSONB dict | None
- save_conversation_state(cid, state_data) — UPSERT
- delete_conversation_state(cid) — cleanup

### Threads (9) — v6.0.0
- create_thread(thread_id, cid, centroid_embedding, message_ids, label) — insert
- get_threads(cid) — all threads for conversation
- get_thread(thread_id) — single thread dict | None
- update_thread_centroid(thread_id, centroid, message_id) — update centroid + append message_id
- update_thread_summary(thread_id, summary) — set summary text
- update_thread_label(thread_id, label) — set label text
- find_nearest_thread(cid, embedding, threshold=0.55) — pgvector cosine search for nearest centroid
- count_threads(cid) — thread count per conversation
- delete_threads_for_conversation(cid) — cleanup

### Research Insights (5) — v6.0.0
- create_insight(cid, thread_id, type, text, embedding, confidence, source_msg_id) — insert
- get_insights(cid, limit=50) — all insights for conversation
- get_insights_for_thread(thread_id, limit=20) — insights scoped to thread
- search_similar_insights(embedding, k, cid, insight_type) — cross-thread semantic search with optional filters
- delete_insights_for_conversation(cid) — cleanup

### Concept Links (4) — v6.0.0
- create_concept_link(concept, embedding, source_type, source_id, cid, thread_id) — insert
- get_concepts_for_conversation(cid) — all concepts
- search_similar_concepts(embedding, k=5, cid=None) — cross-thread semantic search
- delete_concepts_for_conversation(cid) — cleanup

## HNSW Indexing

All vector columns use pgvector's HNSW (Hierarchical Navigable Small World) index with vector_cosine_ops operator class. This provides sub-millisecond approximate nearest-neighbor search. Indexed columns: user_queries.embedding, document_chunks.embedding, conversations.topic_embedding, conversation_threads.centroid_embedding, research_insights.embedding, concept_links.embedding.
