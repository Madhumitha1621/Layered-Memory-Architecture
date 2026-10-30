# API Reference — 24 Endpoints, Data Models, Streaming

## Base URL

Default: http://localhost:8000. All endpoints return JSON unless noted. Version: 6.0.0.

## Chat Endpoints (3)

### POST /chat
Non-streaming chat. Runs the full pipeline and returns complete JSON.

Request body:
```json
{
  "user_query": "What is the BehaviorPolicy engine?",
  "conversation_id": "optional-uuid-string",
  "tags": ["optional"],
  "user_id": "default"
}
```

Response body:
```json
{
  "response": "The full response text.",
  "conversation_id": "uuid",
  "intent": "knowledge_base",
  "confidence": 0.92,
  "retrieval_info": {
    "num_docs": 4,
    "similar_queries": 2,
    "behavior_mode": "standard",
    "behavior_triggers": ["standard"]
  },
  "query_tags": ["technical"],
  "behavior_mode": "standard"
}
```

### POST /chat/stream
Streaming chat using Vercel AI SDK data stream protocol over SSE. Same request body as POST /chat.

Response: text/event-stream with stage events → text deltas → final metadata → finish events:
```
8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]
8:[{"stage":"retrieved","retrieval_info":{"num_docs":4,"similar_queries":2}}]
8:[{"stage":"generating"}]
0:"Hello"
0:", here is"
0:" the answer."
8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...},"query_tags":[...]}]
e:{"finishReason":"stop"}
d:{"finishReason":"stop"}
```

If generation fails, the stream emits an error part `3:"message"` followed by `e:`/`d:` with `"finishReason":"error"`, and nothing is persisted. POST /chat returns HTTP 502 in the same situation.

### POST /chat/regenerate
Delete last assistant message and regenerate. Request: `{"conversation_id": "uuid", "user_id": "default"}`.

## Conversation Endpoints (9)

### POST /conversations
Create new conversation. Request: `{"title": "My Chat"}`. Returns conversation dict with UUID.

### GET /conversations?user_id=default
List all conversations ordered by updated_at DESC.

### GET /conversations/search?q=...&user_id=default
Search conversations by title/content using ILIKE.

### GET /conversations/{id}
Get conversation messages.

### GET /conversations/{id}/messages?limit=200
Get all messages in chronological order.

### PUT /conversations/{id}
Rename. Request: `{"title": "New Title"}`.

### DELETE /conversations/{id}
Delete conversation and cascade: messages, queries, state, threads, insights, concepts.

### GET /conversations/{id}/export?format=json
Export full conversation (messages + metadata) as JSON.

### GET /conversations/{id}/state
Inspect behavioral state for debugging. Returns ConversationState data: tone, pattern, repetition, testing, precision_mode, message_count, etc.

## Research Endpoints (6) — v6.0.0

### GET /conversations/{id}/threads
List all topic threads for a conversation. Each thread has: id, label, message_count, summary, last_active.

### GET /conversations/{id}/threads/{thread_id}
Get single thread details including summary, label, message_count, centroid.

### GET /conversations/{id}/insights
List extracted research insights for a conversation. Each: id, insight_type, insight_text, confidence_score, thread_id.

### GET /conversations/{id}/concepts
List concept links for a conversation. Each: id, concept, source_type, source_id, thread_id.

### GET /concepts/search?q=...&conversation_id=...
Semantic search for related concepts. Embeds query q, searches concept_links via pgvector. Optional conversation_id filter.

### GET /insights/search?q=...&k=10&type=...&conversation_id=...
Cross-thread semantic search over extracted insights. Embeds query q, searches research_insights. Optional filters: type (decision/conclusion/hypothesis/open_question/observation), conversation_id.

## Profile Endpoints (4)

### GET /profile?user_id=default
Get all stored user profile entries.

### POST /profile
Add entry. Request: `{"key": "name", "value": "Alex", "category": "personal", "user_id": "default"}`.

### PUT /profile/{entry_id}
Update entry by id.

### DELETE /profile/{entry_id}
Delete entry by id.

## System Endpoints (2)

### GET /health
Returns system status:
```json
{
  "status": "ok",
  "database": "connected",
  "documents": 47,
  "llm_provider": "cerebras",
  "version": "6.0.0"
}
```

### GET /
Serve React frontend from frontend/dist/ if it exists, otherwise fallback HTML. Static assets mounted from frontend/dist/assets/.

## When Database Is Unavailable

All endpoints return graceful responses when DB_ENABLED is False:
- GET /conversations → {"conversations":[], "count":0}
- POST/PUT/DELETE with DB → 503 "Database not available"
- GET /profile → {"entries":[], "count":0}
- /chat and /chat/stream still work (in-memory mode, no persistence)

## Headers

POST /chat/stream sets Cache-Control: no-cache and X-Accel-Buffering: no to prevent proxy buffering.

## CORS

CORSMiddleware with allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]. Allows Vite dev server (port 5173) to call API (port 8000).
