# RAG Chat v6.0.0 — Complete Implementation Document

> **Auto-generated**: February 26, 2026  
> **Covers**: Full backend pipeline, all 40+ decision gates, 50+ thresholds, 8 database tables, 23 API endpoints, complete frontend architecture, streaming protocol, and component tree.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Configuration & Settings](#3-configuration--settings)
4. [Database Schema](#4-database-schema)
5. [Backend Pipeline — 12-Step Flow](#5-backend-pipeline--12-step-flow)
6. [Intent Classification Gate](#6-intent-classification-gate)
7. [Topic Continuation Gate](#7-topic-continuation-gate)
8. [Behavior Engine & State Tracking](#8-behavior-engine--state-tracking)
9. [Topic Threading System](#9-topic-threading-system)
10. [Research Memory — Insights & Concepts](#10-research-memory--insights--concepts)
11. [Policy Engine](#11-policy-engine)
12. [Context Manager — Token Budgeting](#12-context-manager--token-budgeting)
13. [Prompt Orchestrator](#13-prompt-orchestrator)
14. [LLM Provider Layer](#14-llm-provider-layer)
15. [Vector Store & Embeddings](#15-vector-store--embeddings)
16. [Document Chunking & Ingestion](#16-document-chunking--ingestion)
17. [Caching Layer](#17-caching-layer)
18. [Background Worker & Post-Response](#18-background-worker--post-response)
19. [Thread Summarizer](#19-thread-summarizer)
20. [Lifecycle Hooks](#20-lifecycle-hooks)
21. [API Reference — All 23 Endpoints](#21-api-reference--all-23-endpoints)
22. [Streaming Protocol](#22-streaming-protocol)
23. [Frontend Architecture](#23-frontend-architecture)
24. [Frontend Component Tree](#24-frontend-component-tree)
25. [Frontend State Management](#25-frontend-state-management)
26. [Frontend Streaming Hook](#26-frontend-streaming-hook)
27. [AI Sub-Components](#27-ai-sub-components)
28. [Keyboard Shortcuts](#28-keyboard-shortcuts)
29. [CSS & Animation System](#29-css--animation-system)
30. [Complete Gate Reference Table](#30-complete-gate-reference-table)
31. [Complete Threshold Reference Table](#31-complete-threshold-reference-table)
32. [Infrastructure & Deployment](#32-infrastructure--deployment)

---

## 1. System Overview

RAG Chat is a retrieval-augmented generation chatbot with:

- **Multi-tier memory**: conversation history (recency + semantic), cross-conversation Q&A, user profile, knowledge base (pgvector), topic threads, research insights, and concept links
- **Behavior engine**: real-time detection of user emotional tone, interaction patterns (testing, repetition, rapid-fire, exploratory), and dynamic personality/precision adaptation
- **Topic threading**: automatic message clustering into topic threads using EMA-updated centroids with cosine similarity attachment
- **Research memory**: LLM-powered insight extraction (decisions, conclusions, hypotheses, questions, observations) and heuristic concept linking
- **Streaming**: Server-Sent Events using the Vercel AI SDK data-stream protocol with real-time stage annotations
- **Policy engine**: intent-driven retrieval routing with behavior overrides and lifecycle hooks

### Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Backend | Python + FastAPI | 3.12 / 0.115+ |
| Database | PostgreSQL + pgvector | 16 / 0.3+ |
| Embeddings | sentence-transformers (BAAI/bge-base-en-v1.5) | 3.0+ |
| LLM (default) | Cerebras (gpt-oss-120b) | 1.0+ |
| LLM (optional) | OpenAI (gpt-4o) / Anthropic (claude-sonnet-4-20250514) | — |
| Cache (optional) | Redis | 7+ |
| Frontend | React 18 + Vite 6 + TypeScript 5.7 | — |
| State | Zustand 5 | — |
| Styling | Tailwind CSS 3.4 | — |
| Markdown | react-markdown + remark-gfm + rehype-highlight | — |

---

## 2. Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Vite :5173)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ │
│  │ Sidebar  │ │ ChatArea │ │InputArea │ │ AI Panels │ │  Modals   │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ └───────────┘ │
│       │            │            │              │             │        │
│       └────────────┴────────────┴──────────────┴─────────────┘        │
│                         Zustand Store + API Client                    │
│                         Custom SSE Stream Hook                        │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / SSE (Vite proxy)
┌────────────────────────────────────┴───────────────────────────────────┐
│                     BACKEND — FastAPI (:8000)                         │
│                                                                       │
│  POST /chat/stream ──→ 12-Step Pipeline                              │
│    1. Embed query ─────────────────┐                                  │
│    2. Load history + profile ──────┤ parallel (ThreadPoolExecutor)    │
│    3. Classify intent ─────────────┘                                  │
│    4. Topic continuation gate                                         │
│    5. Behavior engine (state + decisions)                             │
│    6. Topic threading (resolve/create)                                │
│    7. Research context (insights + concepts)                          │
│    8. Policy resolve (retrieval routing)                              │
│    9. History pruning (recency + semantic)                            │
│   10. Selective retrieval (RAG + QA + profile)                       │
│   11. Prompt build + LLM stream                                      │
│   12. Background persist (messages, profile, insights, concepts)     │
│                                                                       │
│  ┌────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────┐     │
│  │ Classifier │ │BehaviorEngine │ │TopicThreading│ │ Research  │     │
│  │ (heuristic │ │  (state +     │ │ (EMA centroid│ │  Memory   │     │
│  │  + LLM)    │ │  decisions)   │ │  + cosine)   │ │(LLM+heur)│     │
│  └────────────┘ └───────────────┘ └──────────────┘ └──────────┘     │
│  ┌────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────┐     │
│  │   Policy   │ │    Context    │ │   Prompt     │ │  Vector   │     │
│  │   Engine   │ │   Manager     │ │ Orchestrator │ │   Store   │     │
│  └────────────┘ └───────────────┘ └──────────────┘ └──────────┘     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
            ┌────────────────────────┼──────────────────────┐
            ▼                        ▼                      ▼
   ┌─────────────┐          ┌──────────────┐        ┌────────────┐
   │ PostgreSQL  │          │  LLM API     │        │   Redis    │
   │ + pgvector  │          │ (Cerebras/   │        │ (optional) │
   │   :55432    │          │  OpenAI/     │        │   :6379    │
   │             │          │  Anthropic)  │        │            │
   │ 8 tables    │          └──────────────┘        └────────────┘
   │ 6 HNSW idx  │
   └─────────────┘
```

---

## 3. Configuration & Settings

All settings live in `backend/settings.py` as a frozen dataclass, loaded from environment variables at import time.

### LLM Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `"cerebras"` | `LLM_PROVIDER` | Provider: `cerebras`, `openai`, or `anthropic` |
| `LLM_API_KEY` | — | `LLM_API_KEY` / `CEREBRAS_API_KEY` | API key for the LLM provider |
| `LLM_MODEL` | provider-dependent | `LLM_MODEL` / `CEREBRAS_MODEL` | Model name (e.g. `gpt-oss-120b`) |
| `LLM_BASE_URL` | `""` | `LLM_BASE_URL` | Custom endpoint (Azure, vLLM, Ollama) |
| `MAX_RESPONSE_TOKENS` | **2048** | `MAX_RESPONSE_TOKENS` | Max tokens per LLM response |
| `MAX_CLASSIFIER_TOKENS` | **50** | hardcoded | Max tokens for intent classification |
| `MAX_PROFILE_DETECT_TOKENS` | **300** | hardcoded | Max tokens for profile update detection |
| `MAX_TITLE_TOKENS` | **20** | hardcoded | Max tokens for auto title generation |
| `MAX_CONTEXT_WINDOW` | **65536** | `MAX_CONTEXT_WINDOW` | Total context window (prompt + response) |

### Embedding Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `EMBEDDING_MODEL` | `"BAAI/bge-base-en-v1.5"` | `EMBEDDING_MODEL` | Sentence-transformer model |
| `EMBEDDING_DIMENSION` | **768** | `EMBEDDING_DIMENSION` | Vector dimension for pgvector columns |
| `QUERY_INSTRUCTION` | `""` | `QUERY_INSTRUCTION` | Optional prefix for query embeddings |

### Retrieval Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `RETRIEVAL_K` | **4** | `RETRIEVAL_K` | Default number of knowledge base docs to retrieve |
| `QA_K` | **4** | `QA_K` | Default number of cross-conv Q&A pairs to retrieve |
| `QA_MIN_SIMILARITY` | **0.72** | `QA_MIN_SIMILARITY` | Floor for cross-conv Q&A cosine similarity |

### History Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `MAX_HISTORY_TOKENS` | **8000** | `MAX_HISTORY_TOKENS` | Hard cap on history token budget |
| `ENABLE_HISTORY_SUMMARIZATION` | **true** | `ENABLE_HISTORY_SUMMARIZATION` | Progressive summarization vs. simple truncation |
| `HISTORY_FETCH_LIMIT` | **100** | `HISTORY_FETCH_LIMIT` | Max messages to load from DB |

### Topic & Semantic Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `TOPIC_CONTINUATION_THRESHOLD` | **0.35** | `TOPIC_CONTINUATION_THRESHOLD` | Cosine min for continuation intent to hold |
| `TOPIC_DECAY_ALPHA` | **0.2** | `TOPIC_DECAY_ALPHA` | EMA alpha for conversation-level topic vector |
| `RECENCY_WINDOW` | **6** | `RECENCY_WINDOW` | Number of recent messages always included |
| `SEMANTIC_K` | **3** | `SEMANTIC_K` | Older messages to retrieve by similarity |
| `SIMILARITY_THRESHOLD` | **0.70** | `SIMILARITY_THRESHOLD` | Floor for semantic history retrieval |

### Behavior Engine Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `BEHAVIOR_ENGINE_ENABLED` | **true** | `BEHAVIOR_ENGINE_ENABLED` | Master toggle for behavior system |
| `BEHAVIOR_REPETITION_THRESHOLD` | **0.7** | `BEHAVIOR_REPETITION_THRESHOLD` | Jaccard similarity threshold for repetition |
| `BEHAVIOR_PATTERN_WINDOW` | **10** | `BEHAVIOR_PATTERN_WINDOW` | Window size for pattern classification |
| `BEHAVIOR_STATE_PERSIST` | **true** | `BEHAVIOR_STATE_PERSIST` | Persist state to DB between sessions |

### Thread Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `THREAD_ENABLED` | **true** | `THREAD_ENABLED` | Master toggle for topic threading |
| `THREAD_ATTACH_THRESHOLD` | **0.55** | `THREAD_ATTACH_THRESHOLD` | Cosine min to attach message to existing thread |
| `THREAD_SUMMARY_INTERVAL` | **8** | `THREAD_SUMMARY_INTERVAL` | Summarize thread every N messages |
| `THREAD_MAX_ACTIVE` | **12** | `THREAD_MAX_ACTIVE` | Max active threads per conversation |

### Research Memory Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `RESEARCH_INSIGHTS_ENABLED` | **true** | `RESEARCH_INSIGHTS_ENABLED` | Master toggle for insight extraction |
| `RESEARCH_INSIGHT_MIN_CONFIDENCE` | **0.6** | `RESEARCH_INSIGHT_MIN_CONFIDENCE` | Minimum LLM confidence to persist an insight |
| `CONCEPT_LINKING_ENABLED` | **true** | `CONCEPT_LINKING_ENABLED` | Master toggle for concept extraction |
| `CONCEPT_LINK_K` | **5** | `CONCEPT_LINK_K` | Max concepts to link per exchange |

### Knowledge Base Settings

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `KNOWLEDGE_DIR` | `<project_root>/knowledge` | `KNOWLEDGE_DIR` | Directory of `.txt`/`.md` reference docs |
| `CHUNK_SIZE` | **500** chars | `CHUNK_SIZE` | Target chunk size for document splitting |
| `CHUNK_OVERLAP` | **50** chars | `CHUNK_OVERLAP` | Overlap between consecutive chunks |
| `FORCE_REINDEX` | **false** | `FORCE_REINDEX` | Re-ingest knowledge base on every startup |

### Database Settings

| Setting | Default | Env Var |
|---|---|---|
| `DATABASE_URL` | `""` | `DATABASE_URL` |
| `POSTGRES_HOST` | `"localhost"` | `POSTGRES_HOST` |
| `POSTGRES_PORT` | **55432** | `POSTGRES_PORT` |
| `POSTGRES_DB` | `"chatapp"` | `POSTGRES_DB` |
| `POSTGRES_USER` | `"root"` | `POSTGRES_USER` |
| `POSTGRES_PASSWORD` | `"password"` | `POSTGRES_PASSWORD` |
| `DB_POOL_MIN` | **1** | `DB_POOL_MIN` |
| `DB_POOL_MAX` | **10** | `DB_POOL_MAX` |

### Cache Settings

| Setting | Default | Env Var |
|---|---|---|
| `ENABLE_CACHE` | **false** | `ENABLE_CACHE` |
| `REDIS_URL` | `"redis://localhost:6379/0"` | `REDIS_URL` |
| `CACHE_TTL` | **3600** s (1 hour) | `CACHE_TTL` |

### Server Settings

| Setting | Default | Env Var |
|---|---|---|
| `HOST` | `"0.0.0.0"` | `HOST` |
| `PORT` | **8000** | `PORT` |
| `DEBUG_MODE` | **false** | `DEBUG_MODE` |
| `STAGE_STREAMING` | **true** | `STAGE_STREAMING` |
| `ALLOWED_ORIGINS` | `"*"` | `ALLOWED_ORIGINS` |
| `DEFAULT_USER_ID` | `"public"` | `DEFAULT_USER_ID` |

---

## 4. Database Schema

PostgreSQL 16 with the `vector` extension (pgvector). All vector columns are `vector(768)`.

### 4.1 `conversations`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | TEXT PK | UUID | Conversation identifier |
| `title` | TEXT NOT NULL | `'New Chat'` | User-visible title |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | Creation time |
| `updated_at` | TIMESTAMP | CURRENT_TIMESTAMP | Last activity time |
| `message_count` | INTEGER | 0 | Total messages in conversation |
| `is_archived` | BOOLEAN | FALSE | Archive flag |
| `metadata` | JSONB | `'{}'` | Extensible metadata |
| `topic_embedding` | vector(768) | NULL | Rolling EMA topic anchor |

**Indexes**: `idx_convs_upd (updated_at DESC)`

### 4.2 `chat_messages`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | SERIAL PK | — | Auto-increment ID |
| `user_id` | TEXT NOT NULL | `'public'` | Owner user |
| `conversation_id` | TEXT NOT NULL | — | Parent conversation |
| `role` | TEXT NOT NULL | — | `'user'` or `'assistant'` |
| `content` | TEXT NOT NULL | — | Message text |
| `tags` | TEXT[] | `'{}'` | Auto-inferred tags |
| `timestamp` | TIMESTAMP | CURRENT_TIMESTAMP | Creation time |
| `metadata` | JSONB | — | Optional metadata |

**Indexes**: `idx_msgs_conv (conversation_id, timestamp)`

### 4.3 `user_queries`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | SERIAL PK | — | Auto-increment ID |
| `query_text` | TEXT NOT NULL | — | Original user query |
| `embedding` | vector(768) | — | Query embedding for similarity search |
| `user_id` | TEXT | `'public'` | Owner user |
| `conversation_id` | TEXT | — | Parent conversation |
| `response_text` | TEXT | — | Paired assistant response |
| `tags` | TEXT[] | `'{}'` | Auto-inferred tags |
| `timestamp` | TIMESTAMP | CURRENT_TIMESTAMP | Creation time |
| `metadata` | JSONB | — | Pipeline metadata |

**Indexes**: `idx_queries_conv`, `idx_queries_ts (DESC)`, `idx_queries_emb (HNSW vector_cosine_ops)`

### 4.4 `user_profile`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | SERIAL PK | — | Auto-increment ID |
| `user_id` | TEXT NOT NULL | `'public'` | Owner user |
| `key` | TEXT NOT NULL | — | Profile attribute name |
| `value` | TEXT NOT NULL | — | Profile attribute value |
| `category` | TEXT | `'general'` | Grouping category |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | — |
| `updated_at` | TIMESTAMP | CURRENT_TIMESTAMP | — |

**Unique constraint**: `(user_id, key)`

### 4.5 `document_chunks`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | SERIAL PK | — | Auto-increment ID |
| `content` | TEXT NOT NULL | — | Chunk text |
| `embedding` | vector(768) | — | Chunk embedding |
| `source` | TEXT | `'default'` | Source file name |
| `chunk_index` | INTEGER | 0 | Position within source |
| `metadata` | JSONB | `'{}'` | Optional metadata |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | — |

**Indexes**: `idx_doc_chunks_emb (HNSW vector_cosine_ops)`, `idx_doc_chunks_src (source)`

### 4.6 `conversation_state`

| Column | Type | Default | Description |
|---|---|---|---|
| `conversation_id` | TEXT PK | FK → conversations.id CASCADE | Parent conversation |
| `state_data` | JSONB NOT NULL | `'{}'` | Serialized ConversationState |
| `updated_at` | TIMESTAMP | CURRENT_TIMESTAMP | Last update |

### 4.7 `conversation_threads`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | TEXT PK | UUID | Thread identifier |
| `conversation_id` | TEXT NOT NULL | FK → conversations.id CASCADE | Parent conversation |
| `centroid_embedding` | vector(768) | — | EMA-updated topic centroid |
| `message_ids` | TEXT[] | `'{}'` | Attached message IDs |
| `message_count` | INTEGER | 0 | Messages in this thread |
| `summary` | TEXT | `''` | LLM-generated thread summary |
| `label` | TEXT | `''` | LLM-generated 3–6 word label |
| `last_active` | TIMESTAMP | CURRENT_TIMESTAMP | Last message attachment time |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | Thread creation time |

**Indexes**: `idx_threads_conv (conversation_id, last_active DESC)`, `idx_threads_emb (HNSW vector_cosine_ops)`

### 4.8 `research_insights`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | SERIAL PK | — | Auto-increment ID |
| `conversation_id` | TEXT NOT NULL | FK → conversations.id CASCADE | Parent conversation |
| `thread_id` | TEXT | FK → conversation_threads.id SET NULL | Associated thread |
| `insight_type` | TEXT NOT NULL | `'observation'` | One of: decision, conclusion, hypothesis, open_question, observation |
| `insight_text` | TEXT NOT NULL | — | Insight content |
| `embedding` | vector(768) | — | Insight embedding for similarity |
| `confidence_score` | FLOAT | 0.8 | LLM-assigned confidence |
| `source_message_id` | TEXT | — | Originating message |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | — |

**Indexes**: `idx_insights_emb (HNSW)`, `idx_insights_conv (conversation_id, created_at DESC)`, `idx_insights_thread (thread_id)`

### 4.9 `concept_links`

| Column | Type | Default | Description |
|---|---|---|---|
| `id` | SERIAL PK | — | Auto-increment ID |
| `concept` | TEXT NOT NULL | — | Concept string |
| `embedding` | vector(768) | — | Concept embedding for similarity |
| `source_type` | TEXT NOT NULL | `'insight'` | `'insight'` or `'message'` |
| `source_id` | TEXT NOT NULL | — | Source insight/message ID |
| `conversation_id` | TEXT | FK → conversations.id CASCADE | Parent conversation |
| `thread_id` | TEXT | FK → conversation_threads.id SET NULL | Associated thread |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | — |

**Indexes**: `idx_concepts_emb (HNSW)`, `idx_concepts_conv (conversation_id)`, `idx_concepts_text (concept)`

**Total: 8 tables, 6 HNSW vector indexes.**

---

## 5. Backend Pipeline — 12-Step Flow

Every message goes through this pipeline (defined in `main.py:run_pipeline()`):

```
User sends message → POST /chat/stream
                          │
     ┌────────────────────┼──────────────────────┐
     ▼                    ▼                      ▼
 1. Embed query    2a. Load history (100)  2b. Load profile
     (BGE 768d)        (PostgreSQL)           (PostgreSQL)
     └────────────────────┼──────────────────────┘
                          │  ThreadPoolExecutor(3) — parallel
                          ▼
                   3. Classify intent
                      (heuristic → LLM)
                          │
                          ▼
                   4. Topic continuation gate
                      IF continuation AND topic_sim < 0.35
                      → Force intent = "general"
                          │
                          ▼
                   5. Behavior engine
                      Load/create ConversationState
                      StateTracker.update()
                      BehaviorEngine.evaluate()
                          │
                          ▼
                   6. Topic threading
                      resolve_thread() → attach or create
                      EMA centroid update
                          │
                          ▼
                   7. Research context
                      get_research_context()
                      Similar insights (k=5) + concepts (k=5)
                          │
                          ▼
                   8. Policy resolve
                      extract_context_features()
                      BehaviorPolicy.resolve()
                      Hooks.run_policy_override()
                      Apply behavior overrides
                          │
                          ▼
                   9. History pruning
                      Recent window (last 6)
                      + Semantic retrieval from older (k=3, sim ≥ 0.65)
                          │
                          ▼
                  10. Selective retrieval
                      RAG docs (k per policy, min_sim per policy)
                      Cross-conv Q&A (k=4/5)
                      Same-conv Q&A (k=4)
                      Profile injection
                          │
                          ▼
                  11. Prompt build + LLM stream
                      prompt_orchestrator.build_messages()
                      Token budget: min(MAX_HISTORY_TOKENS, dynamic_budget)
                      stream_text_deltas() → SSE tokens
                          │
                          ▼
                  12. Background persist
                      (via worker.submit — ThreadPoolExecutor(4))
                      Store messages, update topic, auto-title,
                      detect profile, extract insights/concepts,
                      maybe summarize thread
```

### Step Details

**Step 1–2: Parallel Loading**
- Uses `concurrent.futures.ThreadPoolExecutor(max_workers=3)`
- Embedding: `get_query_embedding(query)` via sentence-transformers
- History: `get_conversation_messages(cid, limit=100)` + `get_recent_chat_messages(cid, limit=10)`
- Profile: `get_user_profile(user_id)`

**Step 3: Intent Classification** — See [Section 6](#6-intent-classification-gate)

**Step 4: Topic Continuation Gate** — See [Section 7](#7-topic-continuation-gate)

**Step 5: Behavior Engine** — See [Section 8](#8-behavior-engine--state-tracking)

**Step 6: Topic Threading** — See [Section 9](#9-topic-threading-system)

**Step 7: Research Context** — See [Section 10](#10-research-memory--insights--concepts)

**Step 8: Policy Resolve** — See [Section 11](#11-policy-engine)

**Step 9: History Pruning**
- Always includes last `RECENCY_WINDOW` (6) messages
- If intent is `continuation` and there are more messages beyond the window, retrieves `SEMANTIC_K` (3) older messages with cosine similarity ≥ `SIMILARITY_THRESHOLD` (0.70)

**Step 10: Selective Retrieval**
- **RAG docs**: calls `vector_store.search(query, k=policy.rag_k, min_similarity=policy.rag_min_similarity)` if `policy.inject_rag` is true
- **Cross-conv Q&A**: calls `retrieve_similar_queries(embedding, k=policy.qa_k, min_similarity=policy.qa_min_similarity)` if `policy.inject_qa_history` and messages > 2
- **Same-conv Q&A**: calls `retrieve_same_conversation_queries(embedding, cid, k=policy.qa_k, min_similarity=0.2)` if `policy.inject_qa_history`
- **Profile**: formats profile entries as `"key: value"` lines if `policy.inject_profile`

**Step 11: Prompt Build + LLM Stream** — See [Section 13](#13-prompt-orchestrator)

**Step 12: Background Persist** — See [Section 18](#18-background-worker--post-response)

---

## 6. Intent Classification Gate

**File**: `backend/llm/classifier.py`

Classifies user intent into one of 5 categories:

| Intent | Description | Triggers |
|---|---|---|
| `general` | Generic question or conversation | Default, greetings, open-ended |
| `continuation` | Follow-up to previous exchange | Pronouns, "what about", "and also" |
| `knowledge_base` | Seeking info from RAG knowledge base | LLM-determined |
| `profile` | Revealing or asking about personal info | "My name is", "What's my email" |
| `privacy` | Privacy-related question | "What do you store", "delete my data" |

### Pre-Heuristic Fast Paths (No LLM Call)

These checks run first, in order. If any matches, the LLM call is skipped entirely:

1. **Cache hit**: If `ENABLE_CACHE`, check `cache.get_classification(query)`. Returns cached `{intent, confidence}`.

2. **Greeting detection**: query ≤ 8 words AND matches any of 17 `GREETING_PATTERNS` (exact match or with trailing separator like `,`, `!`, `.`).  
   → Returns `{"intent": "general", "confidence": 0.97}`  
   Examples: "hello", "hi there", "good morning", "hey!", "what's up"

3. **Profile statement**: query has no `?`, ≤ 15 words, AND starts with any of 19 `PROFILE_STATEMENT_PREFIXES`.  
   → Returns `{"intent": "profile", "confidence": 0.92}`  
   Examples: "my name is keshav", "i prefer dark mode", "i work at google"

4. **Privacy signal**: query contains any of 23 `PRIVACY_SIGNALS`.  
   → Returns `{"intent": "privacy", "confidence": 0.95}`  
   Examples: "what data do you store", "delete my information", "privacy policy"

5. **Continuation heuristic**: context has ≥ 2 messages, query ≤ 8 words, AND either:
   - Contains a continuation pronoun (`it`, `that`, `those`, `this`, `they`, `them`, `which`, `these`, `their`, `its`) AND contains `?`
   - Contains a continuation signal (`also`, `too`, `another`, `more`, `instead`, `else`, `what about`, `how about`, `and`, `but`) AND ≤ 4 words  
   → Returns `{"intent": "continuation", "confidence": 0.85}`

### LLM Fallback

If no heuristic matches:
- Sends the `INTENT_PROMPT` with the last 6 conversation messages as context
- LLM call: `temperature=0.0`, `max_tokens=50`
- Parses JSON response: `{"intent": "...", "confidence": 0.XX}`
- Validates intent against the 5 allowed values
- On parse failure: defaults to `{"intent": "general", "confidence": 0.5}`
- Caches result (TTL 1800s / 30 minutes) if cache enabled

---

## 7. Topic Continuation Gate

**File**: `backend/main.py` (inside `run_pipeline`, step 4)

**Purpose**: Prevents the system from incorrectly treating a new topic as a continuation of the previous topic.

**Gate logic**:
```python
if intent == "continuation":
    topic_vector = get_topic_vector(conversation_id)
    if topic_vector is not None:
        topic_sim = cosine_similarity(query_embedding, topic_vector)
        if topic_sim < TOPIC_CONTINUATION_THRESHOLD:  # 0.35
            intent = "general"
            confidence = 0.7
```

**How the topic vector works**:
- Each conversation has a rolling `topic_embedding` column in the `conversations` table
- Updated after each response via EMA: `topic = (1 - α) * old + α * new` where `α = TOPIC_DECAY_ALPHA` (0.2)
- The vector represents the "center of gravity" of the conversation's semantic content
- If a new query's cosine similarity to this vector is below 0.35, it's too far off-topic to be a genuine continuation

---

## 8. Behavior Engine & State Tracking

### 8.1 Conversation State (`conversation_state.py`)

Each conversation maintains a `ConversationState` dataclass:

| Field | Type | Default | What It Tracks |
|---|---|---|---|
| `current_topic` | str | `""` | Rolling topic label |
| `topic_turns_stable` | int | 0 | Consecutive turns on same topic |
| `topic_drift_count` | int | 0 | Number of topic changes |
| `emotional_tone` | str | `"neutral"` | neutral / positive / frustrated / curious / playful |
| `tone_shift_count` | int | 0 | Total tone transitions |
| `interaction_pattern` | str | `"normal"` | normal / repetitive / testing / exploratory / rapid_fire |
| `testing_flag` | bool | false | User testing the system |
| `repetition_count` | int | 0 | Consecutive similar queries (Jaccard ≥ 0.7) |
| `meta_comment_count` | int | 0 | "you're an AI" type comments |
| `last_intent` | str | `""` | Previous classified intent |
| `intent_history` | list | `[]` | Last N intents (window = 10) |
| `intent_streak` | int | 0 | Consecutive identical intents |
| `message_count` | int | 0 | Total messages |
| `avg_query_length` | float | 0 | Rolling average word count |
| `short_query_streak` | int | 0 | Consecutive queries ≤ 3 words |
| `dynamic_personality_mode` | str | `"default"` | default / concise / detailed / playful / empathetic |
| `last_update` | float | 0 | `time.time()` |
| `conversation_start` | float | 0 | `time.time()` |

**State storage**: In-memory LRU cache (max 200 entries, evicts by `last_update`) + optional DB persistence (in `conversation_state` table).

### 8.2 State Tracker (`StateTracker.update()`)

Called on every message. Updates the state fields based on signal detection:

**Tone detection** (priority ordered — first match wins):
1. **Frustrated**: "wrong", "not what i asked", "useless", "doesn't work", "annoying", etc. (18 signals)
2. **Playful**: "haha", "lol", "😂", "joke", "funny", etc. (13 signals)
3. **Curious**: "how does", "what if", "dig deeper", "explain", "why", etc. (12 signals)
4. **Positive**: "thanks", "awesome", "perfect", "great", "helpful", etc. (15 signals)
5. **Neutral**: default

**Repetition detection**: Jaccard similarity of word sets between current query and each of the last 5 queries. If any pair has Jaccard ≥ `BEHAVIOR_REPETITION_THRESHOLD` (0.7), increments `repetition_count`.

**Testing detection**: checks for 20 `_TESTING_SIGNALS` like "are you an ai", "are you chatgpt", "what model are you", etc.

**Meta-comment detection**: checks for 16 `_META_SIGNALS` like "you're an ai", "who made you", "you're just a bot", etc.

### 8.3 Pattern Classification

After state update, `_classify_pattern()` determines the interaction pattern:

| Priority | Condition | Pattern |
|---|---|---|
| 1 | `testing_flag OR meta_comment_count >= 2` | `"testing"` |
| 2 | `repetition_count >= 3` | `"repetitive"` |
| 3 | `short_query_streak >= 4` | `"rapid_fire"` |
| 4 | `unique intents in last 4 >= 3` | `"exploratory"` |
| 5 | default | `"normal"` |

### 8.4 Precision Mode

`_compute_precision_mode()` determines how precise/focused the response should be:

| Priority | Signals | Mode |
|---|---|---|
| 1 | "prove", "evidence", "source", "citation" | `"adversarial"` |
| 2 | "implement", "code", "build", "step by step", "how to" | `"implementation"` |
| 3 | "what if", "hypothetically", "imagine", "suppose" | `"speculative"` |
| 4 | `rapid_fire` pattern OR `short_query_streak >= 3` | `"concise"` |
| 5 | default | `"analytical"` |

### 8.5 Behavior Engine Decision Tree (`BehaviorEngine.evaluate()`)

After state tracking, the behavior engine produces a `BehaviorDecision` with these possible modes, checked in priority order:

| # | Gate Condition | Mode | Retrieval Effect | Personality | Length |
|---|---|---|---|---|---|
| 1 | `tone == "frustrated"` | `frustration_recovery` | **Boost** (k=6, min_sim=0.3) if repeating | empathetic | detailed |
| 2a | `testing_flag OR pattern == "testing"` | `testing_aware` | **Skip** entirely | playful | normal |
| 2b | `meta_count >= 2 AND !testing` | `meta_aware` | **Reduce** | playful | normal |
| 3a | `repetition_count >= 3` | `repetition_aware` | **Reduce** | empathetic | normal |
| 3b | `repetition_count >= 2` | (overlay only) | — | — | — |
| 4a | `general + low_entropy + msg_count >= 2 + intent_streak >= 2` | `greeting` (loop) | **Skip** | playful | brief |
| 4b | `general + low_entropy + greeting_like` | `greeting` | **Skip** | default | brief |
| 5 | `pattern == "rapid_fire"` | `rapid_fire` | **Reduce** (k=2) | concise | brief |
| 6 | `pattern == "exploratory"` | `exploratory` | **Boost** (k=6, min_sim=0.35) | detailed | detailed |
| 7a | `tone == "playful"` | (overlay) | — | playful | — |
| 7b | `tone == "curious"` | (overlay) | — | detailed | detailed |
| 7c | `tone == "positive"` | (overlay) | — | (context msg) | — |
| default | none | `standard` | — | — | — |

**Helper functions**:
- `_is_greeting_like(q)`: ≤ 2 words and matches greeting words, or any social filler
- `_is_low_entropy(q, word_count)`: ≤ 3 words, or is a greeting, or ≤ 5 words with no "info words"

---

## 9. Topic Threading System

**File**: `backend/topic_threading.py`

### Thread Resolution (`resolve_thread()`)

For each user message:

1. **Gate**: If `THREAD_ENABLED=false` or `DB_ENABLED=false` → return empty resolution
2. **Search**: `find_nearest_thread(conversation_id, query_embedding, threshold=0.55)` — uses pgvector HNSW index on `conversation_threads.centroid_embedding`
3. **Decision**:
   - **Attach** (similarity ≥ 0.55): Update existing thread's centroid via EMA, append message ID, increment count
   - **Create** (similarity < 0.55): Create new thread with query embedding as initial centroid
4. **Warning**: If thread count ≥ `THREAD_MAX_ACTIVE` (12) — logs warning but still creates

### EMA Centroid Update

When a message attaches to an existing thread, the centroid is updated:

```python
if message_count <= 1:
    centroid = new_embedding              # First message: use directly
elif message_count < 4:
    weight = 1.0 / message_count          # Early: cumulative mean
    centroid = old * (1 - weight) + new * weight
else:
    alpha = 0.3                           # Mature: exponential moving average
    centroid = old * (1 - alpha) + new * alpha
centroid = centroid / ||centroid||         # L2 normalize
```

### Thread Context

`get_thread_context(conversation_id, thread_id)` retrieves:
- Thread summary and label
- Message count
- Up to 5 most recent insights from the thread

This context is injected into the LLM prompt so the model knows what topic is being continued.

---

## 10. Research Memory — Insights & Concepts

**File**: `backend/research_memory.py`

### 10.1 Insight Extraction (`extract_insights()`)

**Gates**:
1. `RESEARCH_INSIGHTS_ENABLED=false` or `DB_ENABLED=false` → skip, return `[]`
2. Query < 5 words AND response < 20 words → skip (too short to contain insights)

**Process**:
1. Sends `INSIGHT_EXTRACTION_PROMPT` to LLM with query (truncated to 500 chars) + response (truncated to 1000 chars)
2. LLM call: `temperature=0.3`, `max_tokens=500`
3. Parses JSON array of `{"type", "text", "confidence"}` objects
4. Validates each insight:
   - `type` must be in `{"decision", "conclusion", "hypothesis", "open_question", "observation"}`
   - `confidence` must be ≥ `RESEARCH_INSIGHT_MIN_CONFIDENCE` (0.6)
5. Embeds each insight text → stores in `research_insights` table with embedding

### 10.2 Concept Extraction (`extract_concepts()`)

**Heuristic-only (no LLM call)**. Extracts concepts from text using regex patterns:

1. **Capitalized noun phrases**: sequences of capitalized words (e.g., "Neural Network", "Amazon Web Services")
2. **Technical terms**: snake_case (`my_function`), camelCase (`myFunction`), PascalCase (`MyClass`)
3. **Double-quoted terms**: `"some concept"` (3–50 chars)
4. **Backtick-quoted terms**: `` `code_thing` `` (2–40 chars)

Filters out ~130 common English stop words.

### 10.3 Concept Linking (`link_concepts()`)

For each extracted concept:
1. Embeds the concept text
2. Stores in `concept_links` table with source_type, source_id, conversation_id, thread_id

### 10.4 Research Context Retrieval (`get_research_context()`)

Called during the pipeline (step 7) to inject relevant research memory into the prompt:

1. **Related insights**: `search_similar_insights(embedding, k=5)` → filtered by similarity ≥ 0.4
2. **Concept links**: `search_similar_concepts(embedding, k=5)` → filtered by similarity ≥ 0.4
3. Returns `{"related_insights": [...], "concept_links": [...]}`

The prompt orchestrator injects at most 5 insights and 8 concepts.

---

## 11. Policy Engine

**File**: `backend/policy.py`

### 11.1 Context Feature Extraction (`extract_context_features()`)

Analyzes the query to produce a `ContextFeatures` object:

| Feature | Detection Method |
|---|---|
| `is_greeting` | ≤ 8 words + matches 17 greeting patterns |
| `references_profile` | Contains any of 33 personal reference signals ("my name", "i prefer", etc.) |
| `privacy_signal` | query contains "delete", "privacy", "store", etc. |
| `is_followup` | intent == "continuation" OR structural_score ≥ 0.5 |
| `is_profile_statement` | Starts with profile statement prefix, no "?" |
| `is_profile_question` | References profile AND contains "?" |
| `topic_similarity` | Cosine(query, conversation topic_embedding) |
| `has_profile_data` | profile_entries is non-empty |
| `profile_name` | Extracted from profile entries (keys: name, first_name, full_name, username) |
| `structural_followup_score` | Weighted pronoun/continuation/elaboration signals (see below) |

### 11.2 Structural Follow-up Score

`_compute_structural_followup_score()` returns 0.0–1.0:

| Signal | Weight | Examples |
|---|---|---|
| Pronoun dependency | +0.3 | "it", "that", "those", "this" at start |
| Continuation starters | +0.4 | "what if", "but then", "and what about" |
| Variable references | +0.3 | "the function", "the error", "that code" |
| Elaboration requests | +0.4 | "tell me more", "elaborate", "go deeper" |
| Short follow-ups (≤ 3 words in active conv) | +0.3 | "why?", "how so?" |

Score capped at 1.0.

### 11.3 Policy Decision (`BehaviorPolicy.resolve()`)

Maps intent to retrieval strategy:

| Intent | RAG | Cross-Conv QA | Profile Inject | Route | RAG min_sim |
|---|---|---|---|---|---|
| `privacy` | No | No | If data exists | `"privacy"` | — |
| `profile` (statement) | No | No | No | `"profile_update"` | — |
| `profile` (question) | No | No | If data exists | `"profile"` | — |
| `knowledge_base` | Yes (k=4) | Yes (k=4) | No | `"rag"` | 0.0 |
| `continuation` | Yes (k=4) | Yes (k=4) | No | `"conversation"` | **0.35** |
| `general` | Yes (k=4) | No | No | `"adaptive"` | **0.45** |

### 11.4 Cross-Intent Overlays

After the intent-based decision:
1. If `profile_name` exists but profile isn't being injected → set `greeting_name` (for personalized greeting)
2. If `references_profile` is true AND user has profile data AND profile isn't already injected → force `inject_profile=true`, clear `greeting_name`

### 11.5 Behavior Override Application

After policy, behavior engine overrides are applied:
- `behavior.skip_retrieval=true` → disable RAG + QA injection
- `behavior.reduce_retrieval=true` → apply `rag_k_override` and `rag_min_similarity_override`
- `behavior.boost_retrieval=true` → apply `rag_k_override` and `rag_min_similarity_override`

---

## 12. Context Manager — Token Budgeting

**File**: `backend/context_manager.py`

### Token Estimation

```python
estimate_tokens(text) = max(1, len(text) // 4)     # ~4 chars per token
message_tokens(msg) = estimate_tokens(content) + 10  # 10 token overhead per message
```

### History Budget Calculation

```python
budget = MAX_CONTEXT_WINDOW - MAX_RESPONSE_TOKENS - preamble_tokens
budget = max(budget, min_budget=1000)
effective = min(MAX_HISTORY_TOKENS, budget)           # capped at 8000
```

### Two Strategies

**1. Simple Truncation** (`fit_messages_to_budget`):
- Walk messages from newest to oldest
- Always keep last `min_recent=4` messages
- Add older messages while within budget
- Return in chronological order

**2. Progressive Summarization** (`summarize_old_turns`):
- Split messages into `[overflow | tail(min_recent=6)]`
- If existing summary message found, prepend to overflow transcript
- Build transcript capped at `_MAX_SUMMARIZER_INPUT_TOKENS` (4000 tokens)
- LLM generates 3–8 sentence summary
- Return `[summary_message] + recent_tail`
- On error → fallback to `fit_messages_to_budget`

Controlled by `ENABLE_HISTORY_SUMMARIZATION` (default: true).

---

## 13. Prompt Orchestrator

**File**: `backend/llm/prompt_orchestrator.py`

`build_messages()` assembles the final message array sent to the LLM:

### Message Assembly Order

```
1. SYSTEM_PROMPT (base personality + 8 behavioral rules)
      │
2. GREETING_PERSONALIZATION_FRAME (if greeting_name set)
      │
3. BEHAVIOR_STATE_FRAME
   ├── behavior_context (engine observations)
   ├── PERSONALITY_FRAMES[personality_mode]
   ├── PRECISION_FRAMES[precision_mode]
   └── RESPONSE_LENGTH_HINTS[response_length_hint]
      │
4. THREAD_CONTEXT_FRAME (active thread summary/label/count)
      │
5. RESEARCH_CONTEXT_FRAME
   ├── Up to 5 related insights (type + text + confidence)
   └── Up to 8 concept links
      │
6. PROFILE_CONTEXT_FRAME (user profile key-value pairs)
      │
7. RAG_CONTEXT_FRAME (knowledge base document chunks)
      │
8. PRIVACY_QA_FRAME (if privacy intent) -OR- QA_CONTEXT_FRAME (prior Q&A pairs)
      │
9. Conversation history (budget-fitted or summarized)
      │
10. Current user message
```

### Personality Frames

| Mode | Description |
|---|---|
| `default` | Balanced, helpful, natural |
| `concise` | Direct, minimal, no filler |
| `detailed` | Thorough, explanatory, structured |
| `playful` | Light, witty, use analogies |
| `empathetic` | Warm, patient, validating |

### Precision Frames

| Mode | Description |
|---|---|
| `analytical` | Step-by-step reasoning, cite sources |
| `concise` | Short answers, bullet points |
| `speculative` | Explore possibilities, clear about uncertainty |
| `implementation` | Working code/steps, practical details |
| `adversarial` | Evidence-based, acknowledge limitations |

### Response Length Hints

| Hint | Guidance |
|---|---|
| `brief` | 1–3 sentences, no filler |
| `normal` | Natural length for the topic |
| `detailed` | Comprehensive, use headers/lists, cover edge cases |

---

## 14. LLM Provider Layer

**File**: `backend/llm/providers/`

### Provider Interface (`LLMProvider`)

```python
class LLMProvider(ABC):
    name: str
    @abstractmethod
    def complete(messages, temperature, max_tokens) -> str
    @abstractmethod
    def stream_text_deltas(messages, temperature, max_tokens) -> Generator[str]
```

### Supported Providers

| Provider | SDK | Default Model | Notes |
|---|---|---|---|
| `cerebras` | `cerebras.cloud.sdk.Cerebras` | `gpt-oss-120b` | Default provider |
| `openai` | `openai.OpenAI` | `gpt-4o` | Supports custom `base_url` (Azure, vLLM, Ollama) |
| `anthropic` | `anthropic.Anthropic` | `claude-sonnet-4-20250514` | System messages extracted to top-level `system` param; consecutive same-role messages merged |

### LLM Functions

| Function | Temperature | Max Tokens | Purpose |
|---|---|---|---|
| `classify_intent()` | 0.0 | 50 | Intent classification (JSON) |
| `generate_response()` | 0.3 | 2048 | Non-streaming response |
| `generate_response_stream()` | 0.3 | 2048 | Streaming SSE response |
| `generate_title()` | 0.5 | 20 | Auto-title generation |
| `detect_profile_updates()` | 0.0 | 300 | Profile extraction (JSON) |
| `summarize_old_turns()` | 0.3 | 2048 | History summarization |
| `extract_insights()` | 0.3 | 500 | Insight extraction (JSON) |
| `summarize_thread()` | 0.3 | 300 | Thread summary |
| `generate_thread_label()` | 0.3 | 20 | Thread label |

---

## 15. Vector Store & Embeddings

### Embedding Model

- **Model**: `BAAI/bge-base-en-v1.5` (HuggingFace sentence-transformers)
- **Dimension**: 768
- **Lazy-loaded singleton**: first call triggers download/load
- **Functions**:
  - `get_embedding(text)` → 768-dim float32 (no query prefix)
  - `get_query_embedding(text)` → 768-dim float32 (applies `QUERY_INSTRUCTION` prefix if set)
  - `get_embeddings(texts)` → batch encoding (no prefix)
  - `get_dim()` → actual model dimension

### Vector Store

**Dual-backend** architecture:

1. **PostgreSQL + pgvector** (primary, when `DB_ENABLED`):
   - Stores in `document_chunks` table
   - HNSW index for approximate nearest neighbor search
   - `search()` → `search_document_chunks(embedding, k, min_similarity)`
   
2. **NumPy fallback** (in-memory, when no DB):
   - `_fallback_docs` + `_fallback_embeddings` lists
   - Cosine similarity via `np.dot(query, matrix.T)`

### Similarity Search Functions in `query_db.py`

| Function | Purpose | Default k | Min Sim |
|---|---|---|---|
| `search_document_chunks()` | Knowledge base docs | 4 | 0.0 |
| `retrieve_similar_queries()` | Cross-conversation Q&A | 5 | 0.25 |
| `retrieve_same_conversation_queries()` | Same-conversation Q&A | 4 | 0.2 |
| `get_similar_messages_in_conversation()` | Semantic history retrieval | 3 | 0.4 |
| `find_nearest_thread()` | Thread attachment search | 1 (nearest) | 0.55 |
| `search_similar_insights()` | Research context insights | 5 | 0.0 (filtered to 0.4) |
| `search_similar_concepts()` | Research context concepts | 5 | 0.0 (filtered to 0.4) |

### Cross-Conversation Q&A Ranking Formula

`retrieve_similar_queries()` uses a multi-factor ranking:

```
score = 0.70 × cosine_similarity
      + 0.18 × recency_score
      + 0.05 × tag_overlap
      + 0.05 × same_conversation_boost
```

- **Recency score**: `exp(-age_hours / 72.0)` — half-life of ~72 hours
- **Tag overlap**: `min(overlap_count × 0.2, 0.4)` — capped
- **Same conversation**: flat +0.05 bonus
- Fetches `max(k × 4, 16)` candidates, then re-ranks and returns top k

---

## 16. Document Chunking & Ingestion

**File**: `backend/chunker.py`

### Chunking Strategy (3-pass)

1. **Paragraph split**: Split on `\n{2,}` (double newlines)
2. **Sentence split**: For paragraphs > `CHUNK_SIZE` (500), split on `(?<=[.!?])\s+`
3. **Character window**: For sentences > `CHUNK_SIZE`, sliding window with stride `max(1, size - overlap)`
4. **Merge**: Combine small atoms into chunks up to `CHUNK_SIZE`, carrying overlap from the tail of the previous chunk

### Ingestion Process (`_ingest_knowledge()`)

1. Reads all `.txt` and `.md` files from `KNOWLEDGE_DIR`
2. Chunks each file using `chunk_text(content, CHUNK_SIZE=500, CHUNK_OVERLAP=50)`
3. Stores chunks via `vector_store.add_documents(chunks, source=filename)`
4. The vector store embeds all chunks in batch and inserts into `document_chunks` table

**Startup behavior**: Ingestion runs only if `!has_documents() OR FORCE_REINDEX`.

---

## 17. Caching Layer

**File**: `backend/cache.py`

Optional Redis-backed cache (disabled by default: `ENABLE_CACHE=false`).

### Cache Key Generation

All keys are MD5 hashed: `"rag:{md5(parts_joined)}"`.

### Cached Items

| Item | TTL | Key Truncation | Purpose |
|---|---|---|---|
| Intent classification | **1800s** (30 min) | Query → 200 chars | Avoid re-classifying repeated queries |
| Embedding vectors | **3600s** (1 hour) | Text → 200 chars | Avoid re-embedding repeated text |
| General cache | **3600s** (1 hour) | — | Extensible |

### Graceful Degradation

If Redis is unavailable, all `get` calls return `None` and all `put` calls silently no-op.

---

## 18. Background Worker & Post-Response

**File**: `backend/worker.py`

### Worker Pool

- `ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg-worker")`
- Fire-and-forget pattern: exceptions logged but don't propagate
- Graceful shutdown via `atexit` registration

### Post-Response Persist (`persist_after_response()`)

Runs asynchronously after the streaming response completes:

| Step | Function | Purpose |
|---|---|---|
| 1 | `ensure_conversation_exists()` | Idempotent conversation creation |
| 2 | `store_query()` | Store user query + embedding + response + tags |
| 3 | `store_chat_message()` × 2 | Store user + assistant messages in chat_messages |
| 4 | `update_topic_vector()` | EMA update of conversation topic (α = 0.2) |
| 5 | `increment_message_count()` × 2 | Increment by 2 (user + assistant) |
| 6 | `generate_title()` | Auto-title on first message (if title is "New Chat") |
| 7 | `detect_profile_updates()` | LLM-powered profile extraction from the exchange |
| 8 | `save_conversation_state()` | Persist behavior state to DB |
| 9 | `extract_insights()` | LLM-powered insight extraction |
| 10 | `link_concepts()` | Heuristic concept extraction + embedding + linking |
| 11 | `maybe_summarize()` | Thread summary if message_count % 8 == 0 |

---

## 19. Thread Summarizer

**File**: `backend/thread_summarizer.py`

### Summary Generation

Triggered when `message_count % THREAD_SUMMARY_INTERVAL (8) == 0`:

1. Load thread's messages (last 12) from `chat_messages` where ID in `thread.message_ids`
2. Include `previous_summary` (if exists) as context
3. LLM call with `THREAD_SUMMARY_PROMPT`: "Produce a 3–6 sentence summary"
4. `temperature=0.3`, `max_tokens=300`
5. Store result via `update_thread_summary()`

### Label Generation

Generated on first summary (when thread has no label):

1. Load first 6 messages from thread
2. LLM call with `THREAD_LABEL_PROMPT`: "Generate a concise 3–6 word label"
3. `temperature=0.3`, `max_tokens=20`
4. Strip quotes, truncate to 60 chars
5. Store via `update_thread_label()`

---

## 20. Lifecycle Hooks

**File**: `backend/hooks.py`

### Available Hook Points

| Hook | Decorator | Signature | When |
|---|---|---|---|
| Before generation | `@Hooks.before_generation` | `fn(pipeline_result) → pipeline_result` | After pipeline, before LLM call |
| After generation | `@Hooks.after_generation` | `fn(response, pipeline_result) → str` | After LLM, before sending to user |
| Policy override | `@Hooks.policy_override` | `fn(features, decision) → decision` | After base policy, before application |
| Before persist | `@Hooks.before_persist` | `fn(pipeline_result, response_text) → None` | Before background persist |

Hooks are applied in registration order. Multiple hooks of the same type chain their results.

---

## 21. API Reference — All 23 Endpoints

### Conversations

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/conversations` | `{"title": "New Chat"}` | `{"id", "title", "created_at", "updated_at", "message_count"}` |
| `GET` | `/conversations` | `?limit=50` | `{"conversations": [...], "count"}` |
| `GET` | `/conversations/search` | `?q=text&limit=20` | `{"conversations": [...], "count"}` |
| `GET` | `/conversations/{id}` | — | `{"id", "title", "created_at", "updated_at"}` |
| `GET` | `/conversations/{id}/messages` | `?limit=200` | `{"conversation_id", "messages": [...], "count"}` |
| `PUT` | `/conversations/{id}` | `{"title": "..."}` | `{"id", "title"}` |
| `DELETE` | `/conversations/{id}` | — | `{"deleted": true}` |
| `GET` | `/conversations/{id}/export` | — | `{"id", "title", ..., "messages": [...]}` |
| `GET` | `/conversations/{id}/state` | — | `{"conversation_id", "state", "behavior_engine_enabled"}` |

### Chat

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/chat` | `{"user_query", "conversation_id?", "tags?", "user_id?"}` | `{"response", "conversation_id", "intent", "confidence", "retrieval_info", ...}` |
| `POST` | `/chat/stream` | `{"user_query", "conversation_id?", "tags?", "user_id?"}` | SSE stream (see [Section 22](#22-streaming-protocol)) |
| `POST` | `/chat/regenerate` | `{"conversation_id", "user_id?"}` | Same as `/chat` |

### Threads

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/conversations/{id}/threads` | — | `{"threads": [...], "count"}` |
| `GET` | `/conversations/{id}/threads/{tid}` | — | `{"thread": {...}, "insights": [...]}` |

### Research

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/conversations/{id}/insights` | `?limit=50` | `{"insights": [...], "count"}` |
| `GET` | `/conversations/{id}/concepts` | — | `{"concepts": [...], "count"}` |
| `GET` | `/insights/search` | `?q=text&k=10&type=&conversation_id=` | `{"results": [...], "count"}` |
| `GET` | `/concepts/search` | `?q=text&k=10` | `{"results": [...], "count"}` |

### Profile

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/profile` | `?user_id=public` | `{"entries": [...], "count"}` |
| `POST` | `/profile` | `{"key", "value", "category?", "user_id?"}` | `{"id", "key", "value", "category", "user_id"}` |
| `PUT` | `/profile/{entry_id}` | `{"key", "value", "category?", "user_id?"}` | `{"id", "key", "value", "category", "user_id"}` |
| `DELETE` | `/profile/{entry_id}` | — | `{"deleted": true}` |

### System

| Method | Path | Response |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "database": "connected", "documents": N, "llm_provider": "cerebras", "version": "6.0.0"}` |
| `GET` | `/` | HTML (React build or fallback) |

---

## 22. Streaming Protocol

The `/chat/stream` endpoint uses the **Vercel AI SDK data-stream protocol**:

### Wire Format

Each line is a protocol message. Lines are `\n`-terminated:

| Prefix | Payload | Meaning |
|---|---|---|
| `0:` | JSON-encoded string | **Text token** — append to response |
| `8:` | JSON array | **Annotation** — stage event or metadata |
| `e:` | JSON object | **Finish signal** — `{"finishReason": "stop"}` |
| `d:` | JSON object | **Done signal** — `{"finishReason": "stop"}` |

### Stage Annotations (streamed via `8:`)

Stage annotations are sent in order as the pipeline progresses:

```
8:[{"stage": "classified", "intent": "general", "confidence": 0.97}]
8:[{"stage": "threaded", "thread_resolution": {"thread_id": "xxx", ...}}]
8:[{"stage": "generating"}]
0:"Hello"
0:", how"
0:" can I"
0:" help?"
e:{"finishReason": "stop"}
d:{"finishReason": "stop"}
8:[{"intent": "general", "confidence": 0.97, "retrieval_info": {...}, ...}]
```

### Stage Event Types

| Stage | When | Additional Fields |
|---|---|---|
| `classified` | After intent classification | `intent`, `confidence` |
| `threaded` | After thread resolution | `thread_resolution` (full ThreadResolution) |
| `retrieved` | After retrieval completes | `retrieval_info` |
| `generating` | LLM stream starts | — |
| `complete` | LLM stream ends | — |

### Final Metadata Annotation

After the stream finishes, a final `8:` annotation is sent with the complete pipeline metadata:

```json
{
  "intent": "general",
  "confidence": 0.97,
  "retrieval_info": {
    "intent": "general",
    "route": "adaptive",
    "num_docs": 3,
    "similar_queries": 0,
    "same_conv_qa": 2,
    "topic_similarity": 0.72,
    "profile_injected": false,
    "greeting_personalized": false
  },
  "behavior_mode": "standard",
  "precision_mode": "analytical",
  "thread_resolution": { ... },
  "policy_decision": { ... },
  "research_context": { "related_insights": [...], "concept_links": [...] }
}
```

---

## 23. Frontend Architecture

### Build Configuration

| Tool | Version | Config |
|---|---|---|
| Vite | 6.4.1 | `vite.config.ts` — React plugin, `@` path alias |
| TypeScript | 5.7 | Strict mode, ESNext, bundler module resolution |
| Tailwind CSS | 3.4 | Dark mode (`class`), custom color palette |

### Vite Proxy Routes

All API requests are proxied from `:5173` to `:8000`:
- `/chat` → `http://localhost:8000`
- `/conversations` → `http://localhost:8000`
- `/profile` → `http://localhost:8000`
- `/insights` → `http://localhost:8000`
- `/concepts` → `http://localhost:8000`
- `/health` → `http://localhost:8000`

### Dark Theme Color System

| Token | Hex | Usage |
|---|---|---|
| `sidebar-bg` | `#171717` | Sidebar background |
| `sidebar-hover` | `#2a2a2a` | Sidebar hover state |
| `sidebar-active` | `#343434` | Selected sidebar item |
| `sidebar-text` | `#ececec` | Primary sidebar text |
| `sidebar-muted` | `#8e8e8e` | Secondary text |
| `sidebar-border` | `#2e2e2e` | Dividers |
| `chat-bg` | `#212121` | Main chat background |
| `chat-msg` | `#2f2f2f` | Message bubble background |
| `accent` | `#10a37f` | Primary accent (green) |
| `accent-hover` | `#0d8c6d` | Accent hover state |
| `input-bg` | `#2f2f2f` | Input field background |
| `input-border` | `#424242` | Input field border |
| `danger` | `#ef4444` | Delete actions |

---

## 24. Frontend Component Tree

```
App
├── Sidebar
│   ├── New Chat button
│   ├── Conversation list (with category icons, inline rename, delete)
│   └── Footer buttons (Profile, Research, Debug, Commands)
│
├── ChatArea { chat: useChatStream() }
│   ├── Header bar (sidebar toggle, research dash, thread panel, debug toggle)
│   ├── Messages area (max-w-3xl centered)
│   │   ├── WelcomeScreen (when no messages)
│   │   │   └── 4 suggestion cards (gradient colors, stagger animation)
│   │   └── AIMessage[] { message, isStreaming }
│   │       ├── User message (accent avatar, plain text)
│   │       └── Assistant message
│   │           ├── AIIntentBadge { intent, confidence }
│   │           ├── AIStatusBar { stages, isStreaming }
│   │           │   └── Chip[] (classified, threaded, retrieved, generating/complete)
│   │           ├── Streaming phase indicator ("Thinking…" / "Retrieving…" / "Generating…")
│   │           ├── Markdown body (react-markdown + rehype-highlight)
│   │           │   └── CodeBlock (language label, copy button)
│   │           ├── Actions bar (copy, memory toggle, token meter)
│   │           ├── AIRetrievalPanel { info } (expandable)
│   │           ├── AITokenMeter { used }
│   │           └── AIDebugPanel { metadata } (when debug mode on)
│   ├── Scroll-to-bottom FAB
│   ├── InputArea { chat }
│   │   ├── Auto-resizing textarea (Enter=send, Shift+Enter=newline)
│   │   └── Send / Stop button
│   └── AIThreadPanel (right sidebar, 272px wide)
│       ├── Thread list with expand/collapse
│       └── Thread detail: summary, insights, thread ID
│
├── ProfileModal
│   ├── Entry list with delete buttons
│   └── Add entry form (key, category select, value)
│
├── AIResearchDashboard (full-page overlay)
│   ├── Tab: Threads (2-column grid with summaries)
│   ├── Tab: Insights (search + type filter + insight cards)
│   └── Tab: Concept Graph (bubble cloud + cross-thread matrix)
│
└── CommandPalette (Ctrl+K overlay)
    ├── Search input
    ├── New Chat, View Profile, Toggle Debug commands
    └── Up to 8 recent conversations
```

---

## 25. Frontend State Management

### Zustand Store (`useChatStore`)

| Field | Type | Default | Mutators |
|---|---|---|---|
| `conversations` | `Conversation[]` | `[]` | `addConversation`, `refreshConversations`, `removeConversation`, `renameConversation` |
| `conversationId` | `string \| null` | `null` | `setConversationId` |
| `profileEntries` | `ProfileEntry[]` | `[]` | `refreshProfile`, `addProfileEntry`, `deleteProfileEntry` |
| `sidebarOpen` | `boolean` | `true` | `toggleSidebar` |
| `profileModalOpen` | `boolean` | `false` | `setProfileModalOpen` |
| `memoryPanelOpen` | `Record<string, boolean>` | `{}` | `toggleMemoryPanel` |
| `debugMode` | `boolean` | `false` | `toggleDebugMode` |
| `commandPaletteOpen` | `boolean` | `false` | `setCommandPaletteOpen` |
| `threadPanelOpen` | `boolean` | `false` | `toggleThreadPanel` |
| `researchDashOpen` | `boolean` | `false` | `toggleResearchDash` |
| `threads` | `Thread[]` | `[]` | `refreshThreads` |
| `insights` | `Insight[]` | `[]` | `refreshInsights` |
| `concepts` | `ConceptLink[]` | `[]` | `refreshConcepts` |

---

## 26. Frontend Streaming Hook

**File**: `frontend/src/hooks/use-chat-stream.ts`

### StreamMessage Interface

```typescript
interface StreamMessage {
  id: string;                        // Generated unique ID
  role: 'user' | 'assistant';
  content: string;                   // Accumulated text (grows during streaming)
  annotations: Record<string, any>[];// Accumulated stage events + metadata
  createdAt: Date;
}
```

### Hook Return Value

```typescript
{
  messages: StreamMessage[];
  isLoading: boolean;
  streamingId: string | null;   // ID of currently streaming assistant message
  error: string | null;
  send: (content: string) => Promise<void>;
  stop: () => void;
  setMessages: React.Dispatch<...>;
}
```

### Send Flow

1. If no `conversationId` → create conversation via `api.createConversation()`
2. Optimistically add user message + empty assistant message to state
3. Set `isLoading=true`, `streamingId=assistantId`
4. Fetch `POST /chat/stream` with `AbortController`
5. Read `ReadableStream` chunks, split on `\n`, parse by prefix:
   - `0:` → parse JSON string, append token to assistant message content
   - `8:` → parse JSON array, append to assistant message annotations
   - `e:/d:` → implicit stream end
6. On error: set `error`, append error text to assistant message
7. On complete: `isLoading=false`, `streamingId=null`
8. After 1200ms delay: refresh conversations, threads, insights, concepts

### Conversation Switch

On `conversationId` change:
- Load persisted messages via `api.getMessages(conversationId)`
- Preload threads, insights, concepts

---

## 27. AI Sub-Components

### AIStatusBar

Renders a horizontal timeline of pipeline stage chips:

| Chip | Source | Display |
|---|---|---|
| Classified | `stage == "classified"` | Brain icon + intent label |
| Threaded | `stage == "threaded"` | GitBranch icon + thread label/ID |
| Retrieved | `stage == "retrieved"` | Doc count, Q&A count, topic sim, profile status |
| Generating | `stage == "generating"` | Loader2 (spinning) + pulse |
| Complete | `stage == "complete"` | CheckCircle2 |
| "Details ›" | if retrieval_info present | Toggle for AIRetrievalPanel |

### AIIntentBadge

Color-coded intent indicator with confidence dot:

| Intent | Icon | Color |
|---|---|---|
| `general` | Sparkles | Blue |
| `knowledge_base` | BookOpen | Emerald |
| `continuation` | MessageCircle | Purple |
| `profile` | User | Amber |
| `privacy` | Shield | Red |

Confidence dot: ≥ 0.85 green, ≥ 0.6 yellow, < 0.6 gray.

### AIRetrievalPanel

Expandable panel showing retrieval details:
- Retrieval route (e.g., "adaptive", "rag", "privacy")
- Intent classification
- Knowledge base documents (count)
- Cross-conversation Q&A (count)
- Same-conversation Q&A (count)
- Profile data injection status
- Greeting personalization
- Topic similarity score

### AITokenMeter

Token usage progress bar:
- Estimated via `text.length / 4`
- Default limit: 65,536 (context window)
- Color: green (< 50%), yellow (50–80%), red (> 80%)

### AIDebugPanel

Raw JSON debug view (visible when debug mode is on):
- Policy decision
- Thread resolution  
- Research context
- Retrieval info
- Query tags
- Full metadata dump
- Collapsible sections with monospace rendering

### AIThreadPanel (Right Sidebar)

Shows topic threads for the active conversation:
- Thread list: label, message count, insight count, last active date
- Expandable detail: summary, insights list (with type-colored icons), thread ID
- Refreshes on conversation switch

### AIResearchDashboard (Full-Page Overlay)

3-tab research view:

1. **Threads**: 2-column grid of thread cards with summaries and insight type badges
2. **Insights**: Search + type filter (all/decision/conclusion/hypothesis/open_question/observation) + insight cards with confidence and thread labels
3. **Concept Graph**: 
   - Summary stats (unique concepts × threads)
   - Bubble cloud (font-size scaled by frequency, hover tooltips)
   - Cross-thread link matrix (concepts appearing in multiple threads)

---

## 28. Keyboard Shortcuts

| Shortcut | Scope | Action |
|---|---|---|
| **Ctrl+K / Cmd+K** | Global | Toggle command palette |
| **Enter** | Input textarea | Send message |
| **Shift+Enter** | Input textarea | Insert newline |
| **Enter** | Sidebar rename | Confirm rename |
| **Escape** | Sidebar rename | Cancel rename |
| **Enter** | Profile modal value input | Add entry |
| **Arrow Down** | Command palette | Navigate down |
| **Arrow Up** | Command palette | Navigate up |
| **Enter** | Command palette | Execute selected |
| **Escape** | Command palette | Close |

---

## 29. CSS & Animation System

### Keyframe Animations

| Animation | Class | Duration | Effect |
|---|---|---|---|
| `fadeIn` | `.fade-in` | 0.3s | Opacity 0→1, Y +6→0 |
| `slideIn` | `.slide-in` | 0.25s | Opacity 0→1, Y -10→0 |
| `slideUp` | `.slide-up` | 0.3s | Opacity 0→1, Y +16→0 |
| `scaleIn` | `.scale-in` | 0.2s | Opacity 0→1, Scale 0.95→1 |
| `glowPulse` | `.glow-pulse` | 2s ∞ | Green box-shadow pulse |
| `typingBounce` | `.typing-dot` | 1.2s ∞ | Y 0→-5→0 (staggered) |
| `cursorBlink` | `.streaming-cursor` | 1s ∞ | Step-end opacity toggle |
| `shimmer` | `.skeleton` | 2s ∞ | Gradient sweep |
| `bounce` | — | — | Y 0→-4→0 |
| `bubbleIn` | `.concept-bubble` | 0.4s | Scale 0.6→1 with overshoot |

### Utility CSS Classes

| Class | Effect |
|---|---|
| `.stagger-item` | Fade-in with child-index delay (0–350ms, 50ms increment) |
| `.ai-chip` | Hover lift, gradient overlay, box-shadow |
| `.hover-lift` | translateY(-2px) + shadow on hover |
| `.glass` | Semi-transparent bg + backdrop blur |
| `.accent-gradient` | Green→cyan gradient text |
| `.palette-backdrop` | Backdrop blur(12px) + saturate(120%) |
| `.debug-mono` | JetBrains Mono 10px monospace |
| `.token-progress-bar` | CSS variable `--progress` driven width |
| `.focus-ring` | Accent outline on `:focus-visible` |

---

## 30. Complete Gate Reference Table

All decision points in the system, ordered by pipeline position:

| # | Gate | Condition | Effect | Location |
|---|---|---|---|---|
| 1 | DB Availability | `init_db()` succeeds | All DB features enabled/disabled | `main.py:L74` |
| 2 | Force Reindex | `!has_documents() OR FORCE_REINDEX` | Re-ingest knowledge base | `main.py:L88` |
| 3 | Intent: Cache Hit | Cached classification exists | Skip all heuristics + LLM | `classifier.py` |
| 4 | Intent: Greeting | ≤ 8 words + pattern match | `general @ 0.97`, skip LLM | `classifier.py:L80` |
| 5 | Intent: Profile Statement | No `?`, ≤ 15 words + prefix match | `profile @ 0.92`, skip LLM | `classifier.py:L90` |
| 6 | Intent: Privacy Signal | Contains privacy phrase | `privacy @ 0.95`, skip LLM | `classifier.py:L99` |
| 7 | Intent: Continuation Heuristic | Context ≥ 2, pronoun+? or signal+short | `continuation @ 0.85`, skip LLM | `classifier.py:L104` |
| 8 | Topic Continuation Gate | `continuation AND topic_sim < 0.35` | Force `general @ 0.7` | `main.py:L236` |
| 9 | Behavior Engine Toggle | `BEHAVIOR_ENGINE_ENABLED=false` | Skip behavior system entirely | `main.py:L249` |
| 10 | Behavior: Frustrated | `tone == "frustrated"` | Empathetic + boost retrieval | `behavior_engine.py:L152` |
| 11 | Behavior: Testing | `testing_flag OR pattern == "testing"` | Skip retrieval, playful | `behavior_engine.py:L167` |
| 12 | Behavior: Meta-Aware | `meta_count >= 2 AND !testing` | Reduce retrieval | `behavior_engine.py:L179` |
| 13 | Behavior: High Repetition | `repetition_count >= 3` | Reduce, empathetic | `behavior_engine.py:L190` |
| 14 | Behavior: Mild Repetition | `repetition_count >= 2` | Context overlay | `behavior_engine.py:L202` |
| 15 | Behavior: Greeting Loop | `general + low_entropy + msg≥2 + streak≥2` | Skip retrieval, playful | `behavior_engine.py:L211` |
| 16 | Behavior: Simple Greeting | `general + low_entropy + greeting_like` | Skip retrieval | `behavior_engine.py:L222` |
| 17 | Behavior: Rapid Fire | `pattern == "rapid_fire"` | Reduce (k=2), concise | `behavior_engine.py:L234` |
| 18 | Behavior: Exploratory | `pattern == "exploratory"` | Boost (k=6), detailed | `behavior_engine.py:L245` |
| 19 | Threading Toggle | `THREAD_ENABLED=false OR DB_ENABLED=false` | Skip thread resolution | `main.py:L289` |
| 20 | Thread Attachment | `cosine_similarity >= 0.55` | Attach vs. create new thread | `topic_threading.py:L126` |
| 21 | Thread Max Active | `count >= 12` | Warning log (no block) | `topic_threading.py:L157` |
| 22 | Research Toggle | `RESEARCH_INSIGHTS_ENABLED=false OR DB_ENABLED=false` | Skip research context | `main.py:L302` |
| 23 | Policy: Intent Routing | 5-way intent switch | Determines RAG/QA/profile strategy | `policy.py:L173` |
| 24 | Policy: Name Injection | `profile_name AND !inject_profile` | Set greeting_name | `policy.py:L214` |
| 25 | Policy: Personal Reference | `references_profile AND has_data` | Force inject_profile | `policy.py:L220` |
| 26 | Behavior Override: Skip | `behavior.skip_retrieval` | Disable RAG + QA | `main.py:L316` |
| 27 | Behavior Override: Reduce | `behavior.reduce_retrieval` | Apply k/sim overrides | `main.py:L319` |
| 28 | Behavior Override: Boost | `behavior.boost_retrieval` | Apply k/sim overrides | `main.py:L323` |
| 29 | History Curated | `policy.use_curated_history` | Recency + semantic pruning | `main.py:L339` |
| 30 | Semantic History | `continuation AND history > RECENCY_WINDOW` | Add older semantic matches | `main.py:L342` |
| 31 | RAG Injection | `policy.inject_rag` | Search vector store | `main.py:L358` |
| 32 | QA History Injection | `policy.inject_qa_history AND DB AND msgs > 2` | Same-conv Q&A | `main.py:L372` |
| 33 | Profile Injection | `policy.inject_profile` | Format + inject profile | `main.py:L381` |
| 34 | Privacy Mode Profile | `privacy_mode AND !profile_context` | Force profile load | `main.py:L385` |
| 35 | History Summarization | `ENABLE_HISTORY_SUMMARIZATION=true` | Summarize vs. truncate | `prompt_orchestrator.py:L197` |
| 36 | Insight Short Exchange Skip | `query < 5 words AND response < 20 words` | Skip insight extraction | `research_memory.py:L97` |
| 37 | Insight Confidence Filter | `confidence < 0.6` | Discard low-confidence | `research_memory.py:L119` |
| 38 | Thread Summary Interval | `count > 0 AND count % 8 == 0` | Trigger summarization | `topic_threading.py:L218` |
| 39 | Profile Update Pre-Check | `any(personal_signal in lower)` | Skip LLM if no signals | `profile_detector.py:L37` |
| 40 | Concept Linking Toggle | `CONCEPT_LINKING_ENABLED=false` | Skip concept extraction | `main.py:L471` |
| 41 | EMA Early Stability | `message_count < 4` | Simple mean vs. full EMA | `topic_threading.py:L75` |

---

## 31. Complete Threshold Reference Table

All numeric thresholds, limits, and magic numbers:

### Similarity & Confidence Thresholds

| Threshold | Value | Where Used | Purpose |
|---|---|---|---|
| Topic continuation | **0.35** | Pipeline step 4 | Min similarity to maintain continuation intent |
| Thread attachment | **0.55** | Thread resolution | Min similarity to attach message to thread |
| Semantic history | **0.65** | History pruning | Min similarity for older message retrieval |
| Cross-conv QA | **0.65** | Policy default | Min similarity for cross-conversation Q&A |
| Insight min confidence | **0.6** | Insight extraction | Min LLM confidence to persist insight |
| Repetition detection | **0.7** | State tracker | Jaccard threshold for query repetition |
| Research context insight | **0.4** | Research context | Min similarity for prompt injection |
| Research context concept | **0.4** | Research context | Min similarity for prompt injection |
| Continuation RAG min_sim | **0.35** | Policy | KB floor for continuation intent |
| General RAG min_sim | **0.45** | Policy | KB floor for general intent |
| Frustration boost min_sim | **0.3** | Behavior engine | Broadened KB floor when frustrated |
| Exploratory min_sim | **0.35** | Behavior engine | Broadened KB floor when exploring |
| Cross-conv fetch floor | **0.25** | query_db | DB-level floor for candidate fetch |
| Same-conv fetch floor | **0.2** | query_db | DB-level floor for same-conv Q&A |
| In-conv semantic floor | **0.4** | query_db | Semantic history within conversation |
| Structural followup threshold | **0.5** | Policy | Score to set `is_followup` flag |
| Greeting confidence | **0.97** | Classifier | Pre-heuristic confidence |
| Profile confidence | **0.92** | Classifier | Pre-heuristic confidence |
| Privacy confidence | **0.95** | Classifier | Pre-heuristic confidence |
| Continuation confidence | **0.85** | Classifier | Pre-heuristic confidence |
| Default LLM confidence | **0.5** | Classifier | Fallback on parse error |

### Count & Size Limits

| Limit | Value | Where Used |
|---|---|---|
| Max response tokens | **2,048** | LLM generation |
| Max classifier tokens | **50** | Intent classification |
| Max profile detect tokens | **300** | Profile extraction |
| Max title tokens | **20** | Title generation |
| Context window | **65,536** | Total prompt capacity |
| Max history tokens | **8,000** | History budget hard cap |
| History fetch limit | **100** | Messages loaded from DB |
| Retrieval K (RAG) | **4** | Default KB docs |
| QA K | **4** | Default cross-conv Q&A |
| Recency window | **6** | Always-included recent messages |
| Semantic K | **3** | Older semantic messages |
| Chunk size | **500** chars | Document chunking |
| Chunk overlap | **50** chars | Chunk overlap |
| Thread summary interval | **8** msgs | Summary trigger |
| Thread max active | **12** | Threads per conversation |
| Concept link K | **5** | Concepts per exchange |
| DB pool min/max | **1/10** | Connection pool |
| Cache TTL | **3,600** s | General cache |
| Intent cache TTL | **1,800** s | Intent classification cache |
| State cache max | **200** | In-memory LRU |
| Pattern window | **10** | Intent history length |
| Worker pool | **4** threads | Background tasks |
| Thread EMA alpha | **0.3** | Thread centroid update |
| Topic EMA alpha | **0.2** | Conversation topic update |
| Recency half-life | **72** hours | Q&A recency scoring |
| Summarizer input cap | **4,000** tokens | History summarization |
| Summarizer min_recent | **6** | Messages kept unsummarized |
| Budget fit min_recent | **4** | Messages always kept |
| Budget min floor | **1,000** tokens | Minimum history budget |
| Token estimate | **4** chars/token | Rough approximation |
| Message overhead | **10** tokens | Per-message overhead |
| Thread summarizer last N | **12** | Messages included in summary |
| Thread label first N | **6** | Messages included for label |
| Research query truncation | **500** chars | LLM input limit |
| Research response truncation | **1,000** chars | LLM input limit |
| Insight max_tokens | **500** | LLM budget |
| Thread summary max_tokens | **300** | LLM budget |
| Thread label max_tokens | **20** | LLM budget |
| Prompt max insights injected | **5** | Research prompt |
| Prompt max concepts injected | **8** | Research prompt |
| Q&A fetch multiplier | **4×** (min 16) | Candidate over-fetch for re-ranking |
| Tag overlap cap | **0.4** | Q&A ranking |

### Ranking Weights

**Cross-Conversation Q&A** (`retrieve_similar_queries`):

| Factor | Weight |
|---|---|
| Cosine similarity | **0.70** |
| Recency score | **0.18** |
| Tag overlap | **0.05** |
| Same-conversation bonus | **0.05** |

**Structural Follow-up Score**:

| Signal | Weight |
|---|---|
| Pronoun dependency | **+0.3** |
| Continuation starters | **+0.4** |
| Variable references | **+0.3** |
| Elaboration requests | **+0.4** |
| Short follow-ups | **+0.3** |

---

## 32. Infrastructure & Deployment

### Docker Compose Services

| Service | Image | Port | Notes |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 55432:5432 | User: root, Pass: password, DB: chatapp |
| `redis` | `redis:7-alpine` (commented out) | 6379:6379 | Optional caching |
| `app` | Built from Dockerfile | 8000:8000 | Depends on postgres healthy |

**Volumes**: `postgres_data` (persistent DB), `model_cache` (HuggingFace model cache)

### Dockerfile

- **Base**: `python:3.12.2-slim-bookworm`
- **System deps**: `gcc`, `libpq-dev`
- **PyTorch**: CPU-only (`--index-url https://download.pytorch.org/whl/cpu`)
- **Health check**: `urllib.request` to `/health` every 30s, start-period 60s
- **CMD**: `uvicorn main:app --host 0.0.0.0 --port 8000`

### Startup Sequence

```
1. FastAPI lifespan begins
2. init_db() → create tables, extensions, migrations
3. vector_store.init(db_enabled)
4. _ingest_knowledge() → chunk + embed + store (if needed)
5. App ready on :8000
6. Frontend dev server on :5173 (Vite proxy → :8000)
```

### Production Deployment

```
1. docker-compose up -d          # Start postgres
2. pip install -r requirements.txt
3. cd frontend && npm run build  # Build React → dist/
4. cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
   (serves frontend from dist/ + API on same port)
```

---

*This document covers 18+ backend Python modules, 20+ frontend TypeScript files, 8 database tables, 6 HNSW vector indexes, 23 API endpoints, 41 decision gates, 50+ configurable thresholds, 3 LLM providers, and the complete streaming protocol.*
