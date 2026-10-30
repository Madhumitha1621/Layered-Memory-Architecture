# Configuration Reference — settings.py

## Overview

All configuration lives in settings.py as a frozen dataclass (56 total settings, 53 env-overridable, 3 hardcoded constants). Import from anywhere: `from settings import settings`.

Settings are read from environment variables at startup via python-dotenv. All values are immutable after startup. To change a setting, update .env and restart. No defaults are scattered across files — one file, one source of truth.

## LLM Provider Settings

**LLM_PROVIDER** (default: "cerebras") — Which LLM provider to use. Supported values: "cerebras", "openai", "anthropic". The provider module is lazily imported, so only the selected provider's SDK needs to be installed.

**LLM_API_KEY** (default: "") — API key for the selected provider. Also checks CEREBRAS_API_KEY as a legacy fallback for backward compatibility.

**LLM_MODEL** (default: "") — Model name override. When empty, each provider uses its own default model. Cerebras: gpt-oss-120b. OpenAI: gpt-4o. Anthropic: claude-sonnet-4-20250514.

**LLM_BASE_URL** (default: "") — Optional API endpoint override. Useful for Azure OpenAI, vLLM, Ollama, or any OpenAI-compatible server. Only used by the OpenAI provider.

## Token Budget Settings

**MAX_RESPONSE_TOKENS** (default: 2048) — Maximum tokens in a generated response.

**MAX_CONTEXT_WINDOW** (default: 65536) — Total context window size for the model. Used by the frontend AITokenMeter component.

**MAX_HISTORY_TOKENS** (default: 8000) — Token budget reserved for conversation history. Overflow is trimmed or summarized.

**MAX_CLASSIFIER_TOKENS** (hardcoded: 50) — Maximum tokens for intent classification LLM call. Not env-overridable.

**MAX_PROFILE_DETECT_TOKENS** (hardcoded: 300) — Maximum tokens for profile detection LLM call. Not env-overridable.

**MAX_TITLE_TOKENS** (hardcoded: 20) — Maximum tokens for auto-generating conversation titles. Not env-overridable.

**ENABLE_HISTORY_SUMMARIZATION** (default: True) — When True, overflow turns are compressed into an LLM-generated summary instead of being silently dropped. Costs one extra LLM call per overflowing request.

## Embedding Settings

**EMBEDDING_MODEL** (default: "BAAI/bge-base-en-v1.5") — Local sentence-transformers model. 768 dimensions. ~440 MB download on first use. No API key required.

**EMBEDDING_DIM** (default: 768) — Vector dimension. Must match the model. All pgvector columns use vector(EMBEDDING_DIM). Changing requires dropping and recreating vector columns.

**QUERY_INSTRUCTION** (default: "Represent this sentence for searching relevant passages:") — Prefix applied to queries at retrieval time via get_query_embedding(). Documents encoded without prefix (asymmetric retrieval).

## Retrieval Settings

**RETRIEVAL_K** (default: 4) — Number of document chunks to retrieve for knowledge_base queries.

**QA_K** (default: 4) — Number of similar past Q&A pairs to retrieve.

**SIMILARITY_THRESHOLD** (default: 0.3) — Minimum cosine similarity for knowledge base document results.

## Continuity Settings

**TOPIC_CONTINUATION_THRESHOLD** (default: 0.35) — Minimum cosine similarity to keep "continuation" intent. Below this, demoted to "general".

**TOPIC_DECAY_ALPHA** (default: 0.2) — Exponential decay rate for rolling topic vector. At 0.2, new messages have 20% influence, old have 80%.

**RECENCY_WINDOW** (default: 6) — Number of recent messages always included in curated history.

**SEMANTIC_K** (default: 3) — Maximum semantic history messages beyond recency window.

## Knowledge Base Settings

**KNOWLEDGE_DIR** (default: "knowledge") — Path to knowledge base files. Supports .txt and .md.

**CHUNK_SIZE** (default: 500) — Maximum character length per document chunk.

**CHUNK_OVERLAP** (default: 50) — Overlap characters between consecutive chunks.

**FORCE_REINDEX** (default: false) — When true, clears and rebuilds document_chunks on every startup.

## Database Settings

**DATABASE_URL** (default: "") — Full PostgreSQL connection string. Overrides individual POSTGRES_* settings.

**POSTGRES_HOST** (default: "localhost"), **POSTGRES_PORT** (default: 55432), **POSTGRES_DB** (default: "chatapp"), **POSTGRES_USER** (default: "root"), **POSTGRES_PASSWORD** (default: "password") — Individual connection settings, used when DATABASE_URL is empty.

**DB_POOL_MIN** (default: 2) — Minimum connections in psycopg2 ThreadedConnectionPool.

**DB_POOL_MAX** (default: 10) — Maximum connections.

## Cache Settings

**ENABLE_CACHE** (default: false) — Enable Redis caching. When false, all cache operations are no-ops.

**REDIS_URL** (default: "redis://localhost:6379/0") — Redis connection URL.

**CACHE_TTL** (default: 3600) — Default cache TTL in seconds. Intent classifications use 1800s (30 min).

## Pipeline Settings

**DEFAULT_USER_ID** (default: "default") — User ID used when none is provided in requests.

**HISTORY_FETCH_LIMIT** (default: 50) — Maximum messages loaded from DB for pipeline history.

## Behavior Engine Settings (v6.0.0)

**BEHAVIOR_ENGINE_ENABLED** (default: true) — Enable the behavioral intelligence layer. When false, all behavior modes are skipped and standard mode is used.

**BEHAVIOR_REPETITION_THRESHOLD** (default: 0.7) — Jaccard word-overlap threshold for detecting user repetition. Messages with overlap ≥ 0.7 trigger repetition_aware mode.

**BEHAVIOR_PATTERN_WINDOW** (default: 10) — Number of recent messages to consider for interaction pattern detection (rapid_fire vs exploratory vs deep_dive vs standard).

**BEHAVIOR_STATE_PERSIST** (default: true) — Persist conversation state to database. When false, state is in-memory only (lost on restart).

## Research Engine Settings (v6.0.0)

**THREAD_ENABLED** (default: true) — Enable topic threading. When false, messages are not grouped into threads.

**THREAD_ATTACH_THRESHOLD** (default: 0.55) — Minimum cosine similarity between query embedding and thread centroid to attach the message to an existing thread. Below this, a new thread is created.

**THREAD_SUMMARY_INTERVAL** (default: 8) — Summarize threads at milestone intervals (8, 16, 24... messages). Costs one LLM call per summary.

**THREAD_MAX_ACTIVE** (default: 12) — Maximum active threads per conversation. Prevents unbounded thread growth.

**RESEARCH_INSIGHTS_ENABLED** (default: true) — Enable LLM-powered insight extraction after each response. Types: decision, conclusion, hypothesis, open_question, observation.

**RESEARCH_INSIGHT_MIN_CONFIDENCE** (default: 0.6) — Minimum confidence score for storing an extracted insight. Below this threshold, insights are discarded.

**CONCEPT_LINKING_ENABLED** (default: true) — Enable heuristic concept extraction and cross-linking. No LLM call — uses regex patterns for capitalized nouns, snake_case/camelCase terms, quoted terms, acronyms.

**CONCEPT_LINK_K** (default: 5) — Maximum concept links returned per semantic search query.

## Server Settings

**HOST** (default: "0.0.0.0") — Server bind address.

**PORT** (default: 8000) — Server port.

**ALLOWED_ORIGINS** (default: "*") — CORS allowed origins.

## How Settings Are Loaded

The Settings dataclass is instantiated once as a module-level singleton: `settings = Settings()`. The dataclass is frozen (immutable). load_dotenv() reads .env before the dataclass is constructed. The three hardcoded constants (MAX_CLASSIFIER_TOKENS, MAX_PROFILE_DETECT_TOKENS, MAX_TITLE_TOKENS) use simple `int = N` assignments rather than `_env_int()` calls, making them not overridable via environment variables.
