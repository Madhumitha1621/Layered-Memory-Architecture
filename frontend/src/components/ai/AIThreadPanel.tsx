/**
 * AI Thread Panel — active thread label, summary, insights.
 *
 * Shows the current conversation's topic threads with their labels,
 * message counts, and associated insights. Fetches from the backend
 * /conversations/{id}/threads endpoint.
 */

import { motion, AnimatePresence } from 'framer-motion';
import {
  GitBranch,
  MessageSquare,
  Lightbulb,
  Link2,
  Clock,
  ChevronDown,
} from 'lucide-react';
import { useState } from 'react';
import { useChatStore } from '../../store';

export default function AIThreadPanel() {
  const {
    conversationId,
    threads,
    insights,
  } = useChatStore();

  const [expandedThread, setExpandedThread] = useState<string | null>(null);

  if (!conversationId) {
    return (
      <div className="p-4 text-center text-zinc-500 text-xs">
        Start a conversation to see topic threads
      </div>
    );
  }

  if (threads.length === 0) {
    return (
      <div className="p-4 text-center text-zinc-600 text-xs">
        <GitBranch size={16} className="mx-auto mb-2 opacity-30" />
        No threads yet — threads emerge as you chat
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-[288px]">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-surface-2/30">
        <GitBranch size={13} className="text-accent" />
        <span className="text-2xs font-semibold text-zinc-400 tracking-wider uppercase">
          Topic Threads
        </span>
        <span className="ml-auto text-2xs text-zinc-600 bg-surface-1 px-1.5 py-0.5 rounded-full font-medium">
          {threads.length}
        </span>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto py-2 space-y-0.5">
        {threads.map((thread) => {
          const threadInsights = insights.filter(
            (i) => i.thread_id === thread.id,
          );
          const isExpanded = expandedThread === thread.id;

          return (
            <div key={thread.id} className="mx-2">
              <button
                onClick={() =>
                  setExpandedThread(isExpanded ? null : thread.id)
                }
                className={`w-full flex items-start gap-2 px-3 py-2.5 rounded-xl text-left
                  transition-all duration-200 text-sm ${
                    isExpanded
                      ? 'bg-surface-1/80 text-white shadow-inner-glow'
                      : 'text-zinc-400 hover:bg-surface-1/40'
                  }`}
              >
                <GitBranch
                  size={11}
                  className="mt-0.5 flex-shrink-0 text-accent/60"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate text-xs">
                      {thread.label || thread.id.slice(0, 8) + '…'}
                    </span>
                    <ChevronDown
                      size={10}
                      className={`transition-transform duration-200 text-zinc-600 ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    />
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-2xs text-zinc-600">
                    <span className="flex items-center gap-1">
                      <MessageSquare size={9} />
                      {thread.message_count} msgs
                    </span>
                    {threadInsights.length > 0 && (
                      <span className="flex items-center gap-1">
                        <Lightbulb size={9} />
                        {threadInsights.length} insights
                      </span>
                    )}
                    {thread.last_active && (
                      <span className="flex items-center gap-1">
                        <Clock size={9} />
                        {new Date(thread.last_active).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
              </button>

              {/* Expanded detail */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    className="ml-7 mt-1 mb-2 space-y-2 overflow-hidden"
                  >
                    {/* Summary */}
                    {thread.summary && (
                      <div className="px-3 py-2 rounded-lg bg-surface-1/30 text-[11px] text-zinc-400 leading-relaxed">
                        {thread.summary}
                      </div>
                    )}

                    {/* Thread insights */}
                    {threadInsights.length > 0 && (
                      <div className="space-y-1">
                        <div className="px-1 text-2xs font-semibold text-zinc-600 uppercase tracking-wider">
                          Insights
                        </div>
                        {threadInsights.map((insight) => (
                          <div
                            key={insight.id}
                            className="flex items-start gap-2 px-3 py-1.5 rounded-lg bg-surface-1/20 text-2xs"
                          >
                            <InsightTypeIcon type={insight.insight_type} />
                            <div className="flex-1 min-w-0">
                              <span className="text-zinc-400">
                                {insight.insight_text}
                              </span>
                              <div className="flex items-center gap-2 mt-0.5 text-zinc-600">
                                <span className="capitalize">
                                  {insight.insight_type.replace('_', ' ')}
                                </span>
                                {insight.confidence_score > 0 && (
                                  <span>
                                    {Math.round(insight.confidence_score * 100)}%
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Thread ID */}
                    <div className="px-1 text-[9px] text-zinc-700 font-mono">
                      {thread.id}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Insight type icons ─────────────────────────────────────────────────── */

function InsightTypeIcon({ type }: { type: string }) {
  const iconClass = 'flex-shrink-0 mt-0.5';
  switch (type) {
    case 'decision':
      return <Lightbulb size={10} className={`${iconClass} text-amber-400`} />;
    case 'conclusion':
      return <Lightbulb size={10} className={`${iconClass} text-emerald-400`} />;
    case 'hypothesis':
      return <Lightbulb size={10} className={`${iconClass} text-purple-400`} />;
    case 'open_question':
      return <Lightbulb size={10} className={`${iconClass} text-blue-400`} />;
    case 'observation':
      return <Lightbulb size={10} className={`${iconClass} text-zinc-500`} />;
    default:
      return <Link2 size={10} className={`${iconClass} text-zinc-600`} />;
  }
}
