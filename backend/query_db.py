"""PostgreSQL + pgvector persistence layer.

Single database for all storage needs:
  - conversations & chat_messages  — conversation history
  - user_queries                   — per-query embeddings for semantic Q&A search
  - user_profile                   — key-value personal data
  - document_chunks                — knowledge base vector store (pgvector)
  - conversations.topic_embedding  — rolling topic anchor per conversation

Connection pooling via psycopg2 SimpleConnectionPool.
DATABASE_URL env var takes priority over individual POSTGRES_* vars.
"""
import math
import uuid
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import psycopg2
from psycopg2.extras import Json
from psycopg2 import pool

from settings import settings

logger = logging.getLogger(__name__)


def _parse_vector(val):
    """Convert a pgvector string like '[0.1,0.2,...]' to a Python list of floats.

    psycopg2 returns vector columns as strings when the pgvector adapter is not
    registered.  This helper normalises the value so callers always get a list.
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        return [float(x) for x in val.strip("[]").split(",")]
    # numpy array or similar — convert via tolist
    return list(val)


# ---------------------------------------------------------------------------
#  Connection config — DATABASE_URL takes priority, falls back to settings.
# ---------------------------------------------------------------------------
if settings.DATABASE_URL:
    _p = urlparse(settings.DATABASE_URL)
    DB_CONFIG = {
        "host": _p.hostname or "localhost",
        "port": _p.port or 5432,
        "database": (_p.path or "/chatapp").lstrip("/"),
        "user": _p.username or "root",
        "password": _p.password or "password",
    }
else:
    DB_CONFIG = {
        "host": settings.POSTGRES_HOST,
        "port": settings.POSTGRES_PORT,
        "database": settings.POSTGRES_DB,
        "user": settings.POSTGRES_USER,
        "password": settings.POSTGRES_PASSWORD,
    }

# Connection pool — replaces individual get_connection() calls
_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        _pool = pool.SimpleConnectionPool(
            minconn=settings.DB_POOL_MIN,
            maxconn=settings.DB_POOL_MAX,
            **DB_CONFIG,
        )
    return _pool


def get_connection():
    """Get a pooled connection. Caller must call put_connection() when done."""
    return _get_pool().getconn()


def put_connection(conn):
    """Return a connection to the pool, rolling back any dirty transaction first."""
    if conn is None:
        return
    try:
        # A failed statement leaves psycopg2 in an aborted-transaction state.
        # Rolling back here ensures every recycled connection is clean.
        if conn.status != 1:  # 1 = STATUS_READY (idle, no open transaction)
            conn.rollback()
    except Exception:
        pass
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
#  SCHEMA
# ═══════════════════════════════════════════════════════════════════

def init_db():
    """Create all tables needed for the intent-gated architecture."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # autocommit=True means each statement runs in its own implicit transaction.
        # This prevents a failed migration ALTER TABLE from aborting all subsequent DDL.
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # ── conversations ─────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL DEFAULT 'New Chat',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count   INTEGER DEFAULT 0,
                is_archived     BOOLEAN DEFAULT FALSE,
                metadata        JSONB DEFAULT '{}'
            );
        """)

        # Migration: add message_count if missing
        try:
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0;")
        except Exception:
            pass

        # Migration: add topic_embedding (rolling topic anchor vector)
        # NOTE: if you change EMBEDDING_DIMENSION on an existing DB you must
        # DROP and recreate this column (ALTER COLUMN cannot change vector size).
        try:
            dim = settings.EMBEDDING_DIMENSION
            cur.execute(f"ALTER TABLE conversations ADD COLUMN IF NOT EXISTS topic_embedding vector({dim});")
        except Exception:
            pass

        # ── user_profile (structured identity data) ───────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id          SERIAL PRIMARY KEY,
                user_id     TEXT NOT NULL DEFAULT 'public',
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                category    TEXT DEFAULT 'general',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, key)
            );
        """)

        # Migration: add user_id column to existing user_profile tables
        try:
            cur.execute(
                "ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS "
                "user_id TEXT NOT NULL DEFAULT 'public';"
            )
        except Exception:
            pass
        # Migration: replace single-key unique constraint with (user_id, key)
        try:
            cur.execute(
                "ALTER TABLE user_profile DROP CONSTRAINT IF EXISTS user_profile_key_key;"
            )
            cur.execute(
                "ALTER TABLE user_profile "
                "ADD CONSTRAINT user_profile_user_id_key_key UNIQUE (user_id, key);"
            )
        except Exception:
            pass

        # ── user_queries (semantic search over past Q&A) ──────────
        _dim = settings.EMBEDDING_DIMENSION
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_queries (
                id              SERIAL PRIMARY KEY,
                query_text      TEXT NOT NULL,
                embedding       vector({_dim}),
                user_id         TEXT DEFAULT 'public',
                conversation_id TEXT,
                response_text   TEXT,
                tags            TEXT[] DEFAULT '{{}}',
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata        JSONB
            );
        """)

        # ── chat_messages ─────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id              SERIAL PRIMARY KEY,
                user_id         TEXT NOT NULL DEFAULT 'public',
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                tags            TEXT[] DEFAULT '{}',
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata        JSONB
            );
        """)

        # ── Indexes ───────────────────────────────────────────────
        idx = [
            "CREATE INDEX IF NOT EXISTS idx_queries_conv ON user_queries(conversation_id);",
            "CREATE INDEX IF NOT EXISTS idx_queries_ts   ON user_queries(timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_msgs_conv    ON chat_messages(conversation_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_convs_upd    ON conversations(updated_at DESC);",
        ]
        for ddl in idx:
            cur.execute(ddl)

        # Vector index for query embeddings (HNSW – works at any scale)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_emb
                ON user_queries USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception:
            pass

        # ── document_chunks (pgvector knowledge base) ─────────────
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                embedding   vector({_dim}),
                source      TEXT DEFAULT 'default',
                chunk_index INTEGER DEFAULT 0,
                metadata    JSONB DEFAULT '{{}}',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_emb
                ON document_chunks USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception:
            pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_src ON document_chunks(source);"
        )

        # ── Full-text search (BM25 via tsvector) ─────────────────
        # Add tsvector column for hybrid search if missing.
        try:
            cur.execute(
                "ALTER TABLE document_chunks "
                "ADD COLUMN IF NOT EXISTS tsv tsvector;"
            )
        except Exception:
            pass

        # GIN index for fast full-text lookup
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv
                ON document_chunks USING gin(tsv);
            """)
        except Exception:
            pass

        # Trigger: auto-populate tsv on INSERT/UPDATE
        try:
            cur.execute("""
                CREATE OR REPLACE FUNCTION doc_chunks_tsv_trigger() RETURNS trigger AS $$
                BEGIN
                    NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS trg_doc_chunks_tsv ON document_chunks;
            """)
            cur.execute("""
                CREATE TRIGGER trg_doc_chunks_tsv
                BEFORE INSERT OR UPDATE OF content ON document_chunks
                FOR EACH ROW EXECUTE FUNCTION doc_chunks_tsv_trigger();
            """)
        except Exception:
            pass

        # Backfill existing rows that have NULL tsv
        try:
            cur.execute("""
                UPDATE document_chunks
                SET tsv = to_tsvector('english', COALESCE(content, ''))
                WHERE tsv IS NULL;
            """)
        except Exception:
            pass

        # ── conversation_state (behavioral intelligence layer) ────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                state_data      JSONB NOT NULL DEFAULT '{}',
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ── conversation_threads (topic threading engine) ─────────
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS conversation_threads (
                id                TEXT PRIMARY KEY,
                conversation_id   TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                centroid_embedding vector({_dim}),
                message_ids       TEXT[] DEFAULT '{{}}',
                message_count     INTEGER DEFAULT 0,
                summary           TEXT DEFAULT '',
                label             TEXT DEFAULT '',
                last_active       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_threads_conv "
            "ON conversation_threads(conversation_id, last_active DESC);"
        )
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_threads_emb
                ON conversation_threads USING hnsw (centroid_embedding vector_cosine_ops);
            """)
        except Exception:
            pass

        # ── research_insights (extracted knowledge units) ─────────
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS research_insights (
                id                SERIAL PRIMARY KEY,
                conversation_id   TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                thread_id         TEXT REFERENCES conversation_threads(id) ON DELETE SET NULL,
                insight_type      TEXT NOT NULL DEFAULT 'observation',
                insight_text      TEXT NOT NULL,
                embedding         vector({_dim}),
                confidence_score  FLOAT DEFAULT 0.8,
                source_message_id TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_insights_emb
                ON research_insights USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception:
            pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_insights_conv "
            "ON research_insights(conversation_id, created_at DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_insights_thread "
            "ON research_insights(thread_id);"
        )

        # ── concept_links (cross-thread concept graph) ────────────
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS concept_links (
                id          SERIAL PRIMARY KEY,
                concept     TEXT NOT NULL,
                embedding   vector({_dim}),
                source_type TEXT NOT NULL DEFAULT 'insight',
                source_id   TEXT NOT NULL,
                conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                thread_id   TEXT REFERENCES conversation_threads(id) ON DELETE SET NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_concepts_emb
                ON concept_links USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception:
            pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_concepts_conv "
            "ON concept_links(conversation_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_concepts_text "
            "ON concept_links(concept);"
        )

        # ── Dimension migration ─────────────────────────────────
        # If EMBEDDING_DIMENSION changed after initial table creation,
        # ALTER COLUMN cannot resize vector columns — drop & recreate.
        _migration_cols = [
            ("conversations", "topic_embedding"),
            ("user_queries", "embedding"),
            ("document_chunks", "embedding"),
            ("conversation_threads", "centroid_embedding"),
            ("research_insights", "embedding"),
            ("concept_links", "embedding"),
        ]
        for tbl, col in _migration_cols:
            try:
                cur.execute(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = %s::regclass AND attname = %s;",
                    (tbl, col),
                )
                row = cur.fetchone()
                if row and row[0] > 0 and row[0] != _dim:
                    logger.info(f"Migrating {tbl}.{col}: vector({row[0]}) → vector({_dim})")
                    cur.execute(f"ALTER TABLE {tbl} DROP COLUMN {col};")
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} vector({_dim});")
                    # After migrating document_chunks, rows have NULL embeddings
                    # so clear them to trigger re-indexing on next startup.
                    if tbl == "document_chunks":
                        cur.execute("DELETE FROM document_chunks;")
                        logger.info("Cleared document_chunks — will re-index on next startup")
            except Exception:
                pass

        cur.close()
        logger.info("Database initialized – intent-gated architecture ready")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
    finally:
        if conn is not None:
            conn.close()


def ensure_conversation_exists(conversation_id: str, title: str = "New Chat") -> None:
    """Insert a conversations row if it doesn't already exist (idempotent)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (id, title) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;",
            (conversation_id, title),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Error ensuring conversation: {e}")
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  USER PROFILE  (structured identity data – replaces "memories")
# ═══════════════════════════════════════════════════════════════════

def get_user_profile(user_id: str = "public") -> list:
    """Get all profile entries for a user as a list of dicts."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, key, value, category, created_at, updated_at
            FROM user_profile
            WHERE user_id = %s
            ORDER BY category, key;
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()
        return [
            {"id": r[0], "key": r[1], "value": r[2], "category": r[3],
             "created_at": r[4].isoformat() if r[4] else None,
             "updated_at": r[5].isoformat() if r[5] else None}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def update_profile_entry(
    key: str,
    value: str,
    category: str = "general",
    user_id: str = "public",
) -> int | None:
    """Upsert a profile entry keyed on (user_id, key)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Manual upsert so we scope by (user_id, key) regardless of DB version
        cur.execute(
            "SELECT id FROM user_profile WHERE user_id = %s AND key = %s;",
            (user_id, key.strip()),
        )
        row = cur.fetchone()
        if row:
            pid = row[0]
            cur.execute(
                "UPDATE user_profile "
                "SET value = %s, category = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = %s;",
                (value.strip(), category.strip(), pid),
            )
        else:
            cur.execute(
                "INSERT INTO user_profile (user_id, key, value, category) "
                "VALUES (%s, %s, %s, %s) RETURNING id;",
                (user_id, key.strip(), value.strip(), category.strip()),
            )
            pid = cur.fetchone()[0]
        conn.commit(); cur.close()
        logger.info(f"Profile upserted #{pid} [user={user_id}]: {key} = {value[:60]}")
        return pid
    except Exception as e:
        logger.error(f"Error upserting profile entry: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def update_profile_entry_by_id(
    entry_id: int,
    key: str,
    value: str,
    category: str = "general",
    user_id: str = "public",
) -> str:
    """Update an existing profile entry in place (key may change).

    Returns ``"updated"``, ``"not_found"`` (no such entry for this user),
    ``"conflict"`` (the new key already exists for this user) or ``"error"``.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE user_profile "
            "SET key = %s, value = %s, category = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = %s AND user_id = %s RETURNING id;",
            (key.strip(), value.strip(), category.strip(), entry_id, user_id),
        )
        found = cur.fetchone() is not None
        conn.commit(); cur.close()
        return "updated" if found else "not_found"
    except psycopg2.errors.UniqueViolation:
        return "conflict"
    except Exception as e:
        logger.error(f"Error updating profile entry {entry_id}: {e}")
        return "error"
    finally:
        if conn is not None:
            put_connection(conn)


def delete_profile_entry(entry_id: int) -> bool:
    """Delete a profile entry by ID."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM user_profile WHERE id = %s;", (entry_id,))
        ok = cur.rowcount > 0
        conn.commit(); cur.close()
        return ok
    except Exception as e:
        logger.error(f"Error deleting profile entry: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  QUERY EMBEDDINGS  (semantic search – used selectively by intent)
# ═══════════════════════════════════════════════════════════════════

def _recency_score(ts):
    if not ts:
        return 0.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_h = max((now - ts).total_seconds() / 3600.0, 0.0)
    return math.exp(-age_h / 72.0)


def retrieve_similar_queries(query_embedding, k=5, conversation_id=None,
                             current_tags=None, min_similarity=0.25):
    """Retrieve semantically similar past Q&A across ALL conversations.
    
    min_similarity: threshold below which results are discarded.
    Prevents injecting irrelevant context when nothing good matches.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        cur.execute("""
            SELECT id, query_text, response_text, tags, timestamp,
                   1 - (embedding <=> %s::vector) AS similarity,
                   conversation_id
            FROM user_queries
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb, emb, max(k * 4, 16)))

        results = cur.fetchall()
        cur.close()

        tag_set = set(current_tags or [])
        ranked = []
        for r in results:
            sim = float(r[5])
            if sim < min_similarity:
                continue  # Skip low-similarity noise
            tags = r[3] or []
            tag_overlap = len(tag_set & set(tags)) if tag_set else 0
            tag_sc = min(tag_overlap * 0.2, 0.4)
            rec = _recency_score(r[4])
            same_conv = 0.05 if (conversation_id and r[6] == conversation_id) else 0.0
            score = 0.70 * sim + 0.18 * rec + 0.05 * tag_sc + same_conv
            ranked.append({
                "id": r[0], "query": r[1], "response": r[2],
                "tags": tags, "timestamp": r[4],
                "similarity": sim, "recency": rec, "score": score,
                "conversation_id": r[6],
            })

        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:k]

    except Exception as e:
        logger.error(f"Error retrieving similar queries: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def retrieve_same_conversation_queries(query_embedding, conversation_id, k=4, min_similarity=0.2):
    """Retrieve similar past Q&A from the SAME conversation only."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        cur.execute("""
            SELECT id, query_text, response_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM user_queries
            WHERE conversation_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb, conversation_id, emb, k))

        results = cur.fetchall()
        cur.close()

        return [
            {"id": r[0], "query": r[1], "response": r[2], "similarity": float(r[3])}
            for r in results if float(r[3]) >= min_similarity
        ]

    except Exception as e:
        logger.error(f"Error retrieving same-conv queries: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  TOPIC STATE  (rolling topic anchor – prevents cross-topic bleed)
# ═══════════════════════════════════════════════════════════════════

def get_topic_vector(conversation_id: str):
    """Return the current topic embedding for a conversation, or None."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT topic_embedding FROM conversations WHERE id = %s;",
            (conversation_id,)
        )
        row = cur.fetchone()
        cur.close()
        if row and row[0] is not None:
            vec = _parse_vector(row[0])
            return np.array(vec, dtype=np.float32)
        return None
    except Exception as e:
        logger.error(f"Error getting topic vector: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def update_topic_vector(conversation_id: str, new_embedding, alpha: float = 0.1):
    """Update rolling topic vector: topic = (1-α)*old + α*new.
    
    On first message, sets the topic vector directly.
    alpha=0.1 means the topic shifts slowly – 10% weight to each new message.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Get current topic vector
        cur.execute(
            "SELECT topic_embedding FROM conversations WHERE id = %s;",
            (conversation_id,)
        )
        row = cur.fetchone()

        new_emb = new_embedding
        if isinstance(new_emb, np.ndarray):
            new_emb = new_emb.tolist()

        if row and row[0] is not None:
            old_vec = _parse_vector(row[0])
            old_arr = np.array(old_vec, dtype=np.float32)
            new_arr = np.array(new_emb, dtype=np.float32)
            blended = ((1.0 - alpha) * old_arr + alpha * new_arr).tolist()
        else:
            blended = new_emb  # First message: set directly

        cur.execute(
            "UPDATE conversations SET topic_embedding = %s::vector WHERE id = %s;",
            (blended, conversation_id)
        )
        conn.commit(); cur.close()
    except Exception as e:
        logger.error(f"Error updating topic vector: {e}")
    finally:
        if conn is not None:
            put_connection(conn)


def get_similar_messages_in_conversation(query_embedding, conversation_id: str,
                                          k: int = 3, min_similarity: float = 0.4):
    """Retrieve the top-k most semantically similar past Q&A turns within a
    single conversation. Used to supplement recency window with relevance.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        cur.execute("""
            SELECT query_text, response_text,
                   1 - (embedding <=> %s::vector) AS similarity,
                   timestamp
            FROM user_queries
            WHERE conversation_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb, conversation_id, emb, k * 3))

        results = cur.fetchall()
        cur.close()

        return [
            {"query": r[0], "response": r[1], "similarity": float(r[2])}
            for r in results if float(r[2]) >= min_similarity
        ][:k]

    except Exception as e:
        logger.error(f"Error retrieving similar conv messages: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  TAG INFERENCE
# ═══════════════════════════════════════════════════════════════════

def infer_tags(query_text):
    text = (query_text or "").lower()
    tags = []
    kw = {
        "user_info": ["my name", "i am", "i'm", "my preference", "i prefer", "i like", "i work", "i live"],
        "topic_rag": ["rag"],
        "topic_faiss": ["faiss"],
        "topic_embeddings": ["embedding"],
        "topic_memory": ["memory", "remember"],
        "topic_db": ["postgres", "pgvector", "database"],
        "topic_code": ["code", "python", "javascript", "function", "class", "def ", "import "],
    }
    for tag, phrases in kw.items():
        if any(p in text for p in phrases):
            tags.append(tag)
    return tags or ["general"]


# ═══════════════════════════════════════════════════════════════════
#  CONVERSATION CRUD
# ═══════════════════════════════════════════════════════════════════

def create_conversation(title="New Chat"):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cid = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO conversations (id, title) VALUES (%s, %s) RETURNING id, title, created_at;",
            (cid, title),
        )
        r = cur.fetchone()
        conn.commit(); cur.close()
        return {"id": r[0], "title": r[1], "created_at": r[2].isoformat() if r[2] else None}
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def list_conversations(limit=50):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, created_at, updated_at, COALESCE(message_count, 0)
            FROM conversations
            WHERE is_archived = FALSE
            ORDER BY updated_at DESC
            LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        return [
            {"id": r[0], "title": r[1],
             "created_at": r[2].isoformat() if r[2] else None,
             "updated_at": r[3].isoformat() if r[3] else None,
             "message_count": r[4]}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def get_conversation(conversation_id):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = %s;", (conversation_id,))
        r = cur.fetchone()
        cur.close()
        if not r:
            return None
        return {"id": r[0], "title": r[1],
                "created_at": r[2].isoformat() if r[2] else None,
                "updated_at": r[3].isoformat() if r[3] else None}
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def rename_conversation(conversation_id, new_title):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s RETURNING id, title;",
            (new_title, conversation_id),
        )
        r = cur.fetchone()
        conn.commit(); cur.close()
        return {"id": r[0], "title": r[1]} if r else None
    except Exception as e:
        logger.error(f"Error renaming conversation: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def delete_conversation(conversation_id):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_messages WHERE conversation_id = %s;", (conversation_id,))
        cur.execute("DELETE FROM user_queries WHERE conversation_id = %s;", (conversation_id,))
        cur.execute("DELETE FROM conversations WHERE id = %s;", (conversation_id,))
        ok = cur.rowcount > 0
        conn.commit(); cur.close()
        return ok
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


def touch_conversation(conversation_id):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=%s;", (conversation_id,))
        conn.commit(); cur.close()
    except Exception as e:
        logger.error(f"Error touching conversation: {e}")
    finally:
        if conn is not None:
            put_connection(conn)


def increment_message_count(conversation_id: str, amount: int = 1):
    """Increment the message counter on a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE conversations
            SET message_count = COALESCE(message_count, 0) + %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING message_count;
        """, (amount, conversation_id))
        row = cur.fetchone()
        conn.commit(); cur.close()
        return row[0] if row else 0
    except Exception as e:
        logger.error(f"Error incrementing msg count: {e}")
        return 0
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  SEARCH / EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════

def search_conversations(query: str, limit: int = 20) -> list[dict]:
    """Full-text search across conversation messages (ILIKE fallback)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        pattern = f"%{query}%"
        cur.execute("""
            SELECT DISTINCT c.id, c.title, c.updated_at, c.message_count
            FROM conversations c
            JOIN chat_messages m ON m.conversation_id = c.id
            WHERE m.content ILIKE %s
            ORDER BY c.updated_at DESC
            LIMIT %s;
        """, (pattern, limit))
        rows = cur.fetchall()
        cur.close()
        return [
            {"id": r[0], "title": r[1],
             "updated_at": r[2].isoformat() if r[2] else None,
             "message_count": r[3] or 0}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def export_conversation(conversation_id: str) -> dict | None:
    """Return a conversation with all its messages for JSON export."""
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    msgs = get_conversation_messages(conversation_id, limit=10000)
    conv["messages"] = msgs
    return conv


def delete_last_assistant_message(conversation_id: str) -> bool:
    """Delete the most recent assistant message in a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM chat_messages
            WHERE id = (
                SELECT id FROM chat_messages
                WHERE conversation_id = %s AND role = 'assistant'
                ORDER BY timestamp DESC LIMIT 1
            ) RETURNING id;
        """, (conversation_id,))
        deleted = cur.fetchone() is not None
        if deleted:
            cur.execute("""
                UPDATE conversations
                SET message_count = GREATEST(COALESCE(message_count, 0) - 1, 0)
                WHERE id = %s;
            """, (conversation_id,))
        conn.commit(); cur.close()
        return deleted
    except Exception as e:
        logger.error(f"Error deleting last assistant message: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  MESSAGE STORAGE & RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

def store_query(query_text, embedding, response_text="", conversation_id=None,
                user_id="public", metadata=None, tags=None):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            INSERT INTO user_queries
            (query_text, embedding, user_id, conversation_id, response_text, tags, metadata)
            VALUES (%s, %s::vector, %s, %s, %s, %s, %s) RETURNING id;
        """, (query_text, emb, user_id, conversation_id, response_text,
              tags or infer_tags(query_text), Json(metadata or {})))
        qid = cur.fetchone()[0]
        conn.commit(); cur.close()
        return qid
    except Exception as e:
        logger.error(f"Error storing query: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def store_chat_message(role, content, conversation_id=None,
                       user_id="public", tags=None, metadata=None):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_messages (user_id, conversation_id, role, content, tags, metadata)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
        """, (user_id, conversation_id, role, content, tags or ["general"], Json(metadata or {})))
        mid = cur.fetchone()[0]
        conn.commit(); cur.close()
        return mid
    except Exception as e:
        logger.error(f"Error storing chat message: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def get_conversation_messages(conversation_id, limit=200):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, tags, timestamp, id
            FROM chat_messages WHERE conversation_id = %s
            ORDER BY timestamp ASC LIMIT %s;
        """, (conversation_id, limit))
        rows = cur.fetchall()
        cur.close()
        return [
            {"role": r[0], "content": r[1], "tags": r[2] or [],
             "timestamp": r[3].isoformat() if r[3] else None, "id": r[4]}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error getting conversation messages: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def get_recent_chat_messages(conversation_id, limit=10):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, tags, timestamp
            FROM chat_messages WHERE conversation_id = %s
            ORDER BY timestamp DESC LIMIT %s;
        """, (conversation_id, limit))
        rows = cur.fetchall()
        cur.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        logger.error(f"Error getting recent msgs: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def get_first_user_message(conversation_id):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT content FROM chat_messages
            WHERE conversation_id=%s AND role='user'
            ORDER BY timestamp ASC LIMIT 1;
        """, (conversation_id,))
        r = cur.fetchone()
        cur.close()
        return r[0] if r else None
    except Exception as e:
        logger.error(f"Error getting first msg: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════ DOCUMENT CHUNKS (pgvector knowledge base) ═══════════════════

def store_document_chunks(chunks: list[str], source: str = "default") -> int:
    """Embed and store document chunks in pgvector.  Returns count stored."""
    if not chunks:
        return 0
    conn = None
    try:
        from embeddings import get_embeddings

        embeddings = get_embeddings(chunks)
        conn = get_connection()
        cur = conn.cursor()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute("""
                INSERT INTO document_chunks (content, embedding, source, chunk_index)
                VALUES (%s, %s::vector, %s, %s)
            """, (chunk, emb.tolist(), source, i))
        conn.commit()
        cur.close()
        logger.info(f"Stored {len(chunks)} chunks (source={source})")
        return len(chunks)
    except Exception as e:
        logger.error(f"Error storing document chunks: {e}")
        return 0
    finally:
        if conn is not None:
            put_connection(conn)


def search_document_chunks(embedding, k: int = 4, min_similarity: float = 0.0) -> list[str]:
    """Semantic search over document chunks.  Returns chunk content strings.

    Results below *min_similarity* cosine similarity are excluded, so callers
    can suppress irrelevant matches when the intent is uncertain.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (emb, emb, k))
        results = cur.fetchall()
        cur.close()
        return [r[0] for r in results if r[1] is not None and r[1] >= min_similarity]
    except Exception as e:
        logger.error(f"Error searching document chunks: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def search_document_chunks_with_scores(embedding, k: int = 4, min_similarity: float = 0.0) -> list[tuple[str, float]]:
    """Semantic search returning (content, similarity) tuples.

    Same as search_document_chunks but preserves similarity scores
    for retrieval quality analysis.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (emb, emb, k))
        results = cur.fetchall()
        cur.close()
        return [(r[0], round(float(r[1]), 4)) for r in results if r[1] is not None and r[1] >= min_similarity]
    except Exception as e:
        logger.error(f"Error searching document chunks with scores: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def count_document_chunks() -> int:
    """Return the total number of indexed document chunks."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        count = cur.fetchone()[0]
        cur.close()
        return count
    except Exception as e:
        logger.error(f"Error counting document chunks: {e}")
        return 0
    finally:
        if conn is not None:
            put_connection(conn)


def clear_document_chunks(source: str | None = None) -> None:
    """Delete document chunks (optionally filtered by source)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        if source:
            cur.execute("DELETE FROM document_chunks WHERE source = %s", (source,))
        else:
            cur.execute("DELETE FROM document_chunks")
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Error clearing document chunks: {e}")
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  CONVERSATION STATE (behavioral intelligence persistence)
# ═══════════════════════════════════════════════════════════════════

def get_conversation_state(conversation_id: str) -> dict | None:
    """Load conversation state from DB.  Returns raw dict or None."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT state_data FROM conversation_state WHERE conversation_id = %s;",
            (conversation_id,),
        )
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"Error loading conversation state: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def save_conversation_state(conversation_id: str, state_data: dict) -> bool:
    """Upsert conversation state to DB."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversation_state (conversation_id, state_data, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (conversation_id)
            DO UPDATE SET state_data = EXCLUDED.state_data,
                          updated_at = CURRENT_TIMESTAMP;
        """, (conversation_id, Json(state_data)))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error saving conversation state: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


def delete_conversation_state(conversation_id: str) -> bool:
    """Delete conversation state (called on conversation delete)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversation_state WHERE conversation_id = %s;",
            (conversation_id,),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting conversation state: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  CONVERSATION THREADS (topic threading engine)
# ═══════════════════════════════════════════════════════════════════

def create_thread(
    thread_id: str,
    conversation_id: str,
    centroid_embedding,
    label: str = "",
) -> bool:
    """Create a new conversation thread with an initial centroid."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = centroid_embedding.tolist() if isinstance(centroid_embedding, np.ndarray) else centroid_embedding
        cur.execute("""
            INSERT INTO conversation_threads
                (id, conversation_id, centroid_embedding, label, message_count)
            VALUES (%s, %s, %s::vector, %s, 1)
            ON CONFLICT (id) DO NOTHING;
        """, (thread_id, conversation_id, emb, label))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error creating thread: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


def get_threads(conversation_id: str) -> list[dict]:
    """Return all threads for a conversation, most recent first."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, conversation_id, centroid_embedding, message_ids,
                   message_count, summary, label, last_active, created_at
            FROM conversation_threads
            WHERE conversation_id = %s
            ORDER BY last_active DESC;
        """, (conversation_id,))
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id": r[0], "conversation_id": r[1],
                "centroid_embedding": _parse_vector(r[2]), "message_ids": r[3] or [],
                "message_count": r[4], "summary": r[5],
                "label": r[6], "last_active": r[7].isoformat() if r[7] else None,
                "created_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching threads: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def get_thread(thread_id: str) -> dict | None:
    """Return a single thread by ID."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, conversation_id, centroid_embedding, message_ids,
                   message_count, summary, label, last_active, created_at
            FROM conversation_threads WHERE id = %s;
        """, (thread_id,))
        r = cur.fetchone()
        cur.close()
        if not r:
            return None
        return {
            "id": r[0], "conversation_id": r[1],
            "centroid_embedding": _parse_vector(r[2]), "message_ids": r[3] or [],
            "message_count": r[4], "summary": r[5],
            "label": r[6], "last_active": r[7].isoformat() if r[7] else None,
            "created_at": r[8].isoformat() if r[8] else None,
        }
    except Exception as e:
        logger.error(f"Error fetching thread {thread_id}: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def update_thread_centroid(thread_id: str, centroid_embedding, message_id: str | None = None) -> bool:
    """Update a thread's centroid embedding and optionally append a message ID."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = centroid_embedding.tolist() if isinstance(centroid_embedding, np.ndarray) else centroid_embedding
        if message_id:
            cur.execute("""
                UPDATE conversation_threads
                SET centroid_embedding = %s::vector,
                    message_ids = array_append(message_ids, %s),
                    message_count = message_count + 1,
                    last_active = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (emb, message_id, thread_id))
        else:
            cur.execute("""
                UPDATE conversation_threads
                SET centroid_embedding = %s::vector,
                    message_count = message_count + 1,
                    last_active = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (emb, thread_id))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error updating thread centroid: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


def update_thread_summary(thread_id: str, summary: str) -> bool:
    """Store or refresh the progressive summary for a thread."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE conversation_threads
            SET summary = %s, last_active = CURRENT_TIMESTAMP
            WHERE id = %s;
        """, (summary, thread_id))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error updating thread summary: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


def update_thread_label(thread_id: str, label: str) -> bool:
    """Update the human-readable label of a thread."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversation_threads SET label = %s WHERE id = %s;",
            (label, thread_id),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error updating thread label: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


def find_nearest_thread(conversation_id: str, embedding, threshold: float = 0.55):
    """Find the most similar thread centroid.  Returns (thread_id, similarity) or None."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            SELECT id, 1 - (centroid_embedding <=> %s::vector) AS similarity
            FROM conversation_threads
            WHERE conversation_id = %s
            ORDER BY centroid_embedding <=> %s::vector
            LIMIT 1;
        """, (emb, conversation_id, emb))
        row = cur.fetchone()
        cur.close()
        if row and row[1] >= threshold:
            return {"thread_id": row[0], "similarity": row[1]}
        return None
    except Exception as e:
        logger.error(f"Error finding nearest thread: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def count_threads(conversation_id: str) -> int:
    """Return how many threads exist for a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM conversation_threads WHERE conversation_id = %s;",
            (conversation_id,),
        )
        count = cur.fetchone()[0]
        cur.close()
        return count
    except Exception as e:
        logger.error(f"Error counting threads: {e}")
        return 0
    finally:
        if conn is not None:
            put_connection(conn)


def delete_threads_for_conversation(conversation_id: str) -> bool:
    """Delete all threads for a conversation (cascade handles FKs)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversation_threads WHERE conversation_id = %s;",
            (conversation_id,),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting threads: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  RESEARCH INSIGHTS (extracted knowledge units)
# ═══════════════════════════════════════════════════════════════════

def create_insight(
    conversation_id: str,
    insight_type: str,
    insight_text: str,
    embedding=None,
    thread_id: str | None = None,
    confidence_score: float = 0.8,
    source_message_id: str | None = None,
) -> int | None:
    """Store a research insight.  Returns the new row ID or None."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = None
        if embedding is not None:
            emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            INSERT INTO research_insights
                (conversation_id, thread_id, insight_type, insight_text,
                 embedding, confidence_score, source_message_id)
            VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
            RETURNING id;
        """, (conversation_id, thread_id, insight_type, insight_text,
              emb, confidence_score, source_message_id))
        row_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return row_id
    except Exception as e:
        logger.error(f"Error creating insight: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def get_insights(conversation_id: str, limit: int = 50) -> list[dict]:
    """Return recent insights for a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, conversation_id, thread_id, insight_type,
                   insight_text, confidence_score, source_message_id, created_at
            FROM research_insights
            WHERE conversation_id = %s
            ORDER BY created_at DESC LIMIT %s;
        """, (conversation_id, limit))
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id": r[0], "conversation_id": r[1], "thread_id": r[2],
                "insight_type": r[3], "insight_text": r[4],
                "confidence_score": r[5], "source_message_id": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching insights: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def get_insights_for_thread(thread_id: str, limit: int = 20) -> list[dict]:
    """Return insights for a specific thread."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, conversation_id, thread_id, insight_type,
                   insight_text, confidence_score, source_message_id, created_at
            FROM research_insights
            WHERE thread_id = %s
            ORDER BY created_at DESC LIMIT %s;
        """, (thread_id, limit))
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id": r[0], "conversation_id": r[1], "thread_id": r[2],
                "insight_type": r[3], "insight_text": r[4],
                "confidence_score": r[5], "source_message_id": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching thread insights: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def search_similar_insights(
    embedding, k: int = 5, conversation_id: str | None = None,
    insight_type: str | None = None,
) -> list[dict]:
    """Semantic search across research insights.

    Optionally scoped to one conversation and/or filtered by insight_type
    (decision, conclusion, hypothesis, open_question, observation).
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        # Build WHERE clauses dynamically
        conditions = ["embedding IS NOT NULL"]
        params: list = [emb]
        if conversation_id:
            conditions.append("conversation_id = %s")
            params.append(conversation_id)
        if insight_type:
            conditions.append("insight_type = %s")
            params.append(insight_type)
        where = " AND ".join(conditions)
        params.extend([emb, k])

        cur.execute(f"""
            SELECT id, conversation_id, thread_id, insight_type,
                   insight_text, confidence_score,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM research_insights
            WHERE {where}
            ORDER BY embedding <=> %s::vector LIMIT %s;
        """, params)
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id": r[0], "conversation_id": r[1], "thread_id": r[2],
                "insight_type": r[3], "insight_text": r[4],
                "confidence_score": r[5], "similarity": r[6],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error searching similar insights: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def delete_insights_for_conversation(conversation_id: str) -> bool:
    """Delete all insights for a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM research_insights WHERE conversation_id = %s;",
            (conversation_id,),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting insights: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)


# ═══════════════════════════════════════════════════════════════════
#  CONCEPT LINKS (cross-thread knowledge graph)
# ═══════════════════════════════════════════════════════════════════

def create_concept_link(
    concept: str,
    embedding,
    source_type: str,
    source_id: str,
    conversation_id: str,
    thread_id: str | None = None,
) -> int | None:
    """Store a concept link.  Returns new row ID or None."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            INSERT INTO concept_links
                (concept, embedding, source_type, source_id,
                 conversation_id, thread_id)
            VALUES (%s, %s::vector, %s, %s, %s, %s)
            RETURNING id;
        """, (concept, emb, source_type, source_id, conversation_id, thread_id))
        row_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return row_id
    except Exception as e:
        logger.error(f"Error creating concept link: {e}")
        return None
    finally:
        if conn is not None:
            put_connection(conn)


def get_concepts_for_conversation(conversation_id: str) -> list[dict]:
    """Return all concept links for a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, concept, source_type, source_id, thread_id, created_at
            FROM concept_links
            WHERE conversation_id = %s
            ORDER BY created_at DESC;
        """, (conversation_id,))
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id": r[0], "concept": r[1], "source_type": r[2],
                "source_id": r[3], "thread_id": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching concepts: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def search_similar_concepts(embedding, k: int = 5, conversation_id: str | None = None) -> list[dict]:
    """Semantic search across concept links."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        if conversation_id:
            cur.execute("""
                SELECT id, concept, source_type, source_id, thread_id,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM concept_links
                WHERE conversation_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT %s;
            """, (emb, conversation_id, emb, k))
        else:
            cur.execute("""
                SELECT id, concept, source_type, source_id, thread_id,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM concept_links
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT %s;
            """, (emb, emb, k))
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id": r[0], "concept": r[1], "source_type": r[2],
                "source_id": r[3], "thread_id": r[4], "similarity": r[5],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error searching concepts: {e}")
        return []
    finally:
        if conn is not None:
            put_connection(conn)


def delete_concepts_for_conversation(conversation_id: str) -> bool:
    """Delete all concept links for a conversation."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM concept_links WHERE conversation_id = %s;",
            (conversation_id,),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting concepts: {e}")
        return False
    finally:
        if conn is not None:
            put_connection(conn)
