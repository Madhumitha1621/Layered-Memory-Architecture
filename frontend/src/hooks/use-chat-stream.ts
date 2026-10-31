/**
 * Custom streaming chat hook — parses our backend's Vercel AI SDK
 * data‑stream protocol (0: text, 8: annotations, e:/d: finish).
 *
 * Why custom instead of useChat?
 *   • Full control over message state & annotation handling
 *   • No mismatch between AI SDK's request format and our backend
 *   • Reliable conversation creation before first message
 *   • Auto‑refreshes threads/insights/concepts after each response
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { useChatStore } from '../store';
import * as api from '../api';

/* ── Types ──────────────────────────────────────────────────────────────── */

export interface StreamMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  annotations: Record<string, any>[];
  createdAt: Date;
}

let _counter = 0;
const genId = () => `msg-${Date.now()}-${++_counter}`;

/* ── Hook ───────────────────────────────────────────────────────────────── */

export function useChatStream() {
  const [messages, setMessages] = useState<StreamMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Conversation id just created by send(): its messages live only in local
  // state until the stream finishes, so the load effect must not overwrite them.
  const skipLoadRef = useRef<string | null>(null);

  const {
    conversationId,
    setConversationId,
    addConversation,
    refreshConversations,
    refreshThreads,
    refreshInsights,
    refreshConcepts,
  } = useChatStore();

  const cidRef = useRef(conversationId);
  useEffect(() => {
    cidRef.current = conversationId;
  }, [conversationId]);

  /* ── Load persisted messages when the active conversation changes ────── */
  useEffect(() => {
    if (conversationId) {
      if (skipLoadRef.current === conversationId) {
        skipLoadRef.current = null;
      } else {
        api
          .getMessages(conversationId)
          .then(({ messages: saved }) => {
            setMessages(
              saved.map((m, i) => ({
                id: String(m.id ?? `hist-${i}`),
                role: m.role as 'user' | 'assistant',
                content: m.content,
                annotations: [],
                createdAt: m.created_at ? new Date(m.created_at) : new Date(),
              })),
            );
          })
          .catch(console.error);
      }

      // Also preload side‑panel data
      refreshThreads(conversationId);
      refreshInsights(conversationId);
      refreshConcepts(conversationId);
    } else {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  /* ── Send a message ────────────────────────────────────────────────────── */
  const send = useCallback(
    async (content: string) => {
      if (isLoading) return;
      setError(null);

      // 1. Ensure a conversation exists
      let cid = cidRef.current;
      if (!cid) {
        try {
          const conv = await api.createConversation('New Chat');
          cid = conv.id;
          cidRef.current = cid;
          skipLoadRef.current = cid;
          setConversationId(cid);
          addConversation(conv);
        } catch (err) {
          console.error('Conversation creation failed', err);
          setError('Could not create conversation');
          return;
        }
      }

      // 2. Optimistically add user + empty assistant messages
      const userId = genId();
      const asstId = genId();

      setMessages((prev) => [
        ...prev,
        { id: userId, role: 'user', content, annotations: [], createdAt: new Date() },
        { id: asstId, role: 'assistant', content: '', annotations: [], createdAt: new Date() },
      ]);
      setIsLoading(true);
      setStreamingId(asstId);

      // 3. Fetch the streaming response
      try {
        const controller = new AbortController();
        abortRef.current = controller;

        const res = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_query: content, conversation_id: cid }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Server ${res.status}: ${text}`);
        }

        // 4. Read and parse the data‑stream protocol
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop()!; // keep partial last-line

          for (const line of lines) {
            if (!line) continue;
            const prefix = line.slice(0, 2);
            const payload = line.slice(2);

            if (prefix === '0:') {
              // Text delta
              try {
                const token: string = JSON.parse(payload);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === asstId ? { ...m, content: m.content + token } : m,
                  ),
                );
              } catch { /* skip malformed */ }
            } else if (prefix === '8:') {
              // Annotation (array of objects)
              try {
                const arr = JSON.parse(payload);
                const items: Record<string, any>[] = Array.isArray(arr) ? arr : [arr];
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === asstId
                      ? { ...m, annotations: [...m.annotations, ...items] }
                      : m,
                  ),
                );
              } catch { /* skip malformed */ }
            }
            // e: / d: handled implicitly when stream ends
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('Stream error:', err);
          setError(err.message);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === asstId
                ? { ...m, content: m.content || `⚠ ${err.message}` }
                : m,
            ),
          );
        }
      } finally {
        setIsLoading(false);
        setStreamingId(null);
        abortRef.current = null;

        // Refresh conversations list (picks up auto‑generated title) + side panels
        const c = cidRef.current;
        if (c) {
          setTimeout(() => {
            refreshConversations();
            refreshThreads(c);
            refreshInsights(c);
            refreshConcepts(c);
          }, 1200);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isLoading, setConversationId, addConversation],
  );

  /* ── Stop generation ───────────────────────────────────────────────────── */
  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsLoading(false);
    setStreamingId(null);
  }, []);

  /* ── Reload last response ──────────────────────────────────────────────── */
  const reload = useCallback(() => {
    if (isLoading) return;
    const userMessages = messages.filter(m => m.role === 'user');
    const last = userMessages[userMessages.length - 1];
    if (last) send(last.content);
  }, [isLoading, messages, send]);

  return { messages, isLoading, streamingId, error, send, stop, reload, setMessages };
}
