import { create } from 'zustand';
import * as api from './api';
import type { Conversation, ProfileEntry } from './api';

/* ── State shape ────────────────────────────────────────────────────────── */

interface ChatStore {
  /* Conversations */
  conversations: Conversation[];
  conversationId: string | null;
  setConversationId: (id: string | null) => void;
  addConversation: (c: Conversation) => void;
  refreshConversations: () => Promise<void>;
  removeConversation: (id: string) => Promise<void>;
  renameConversation: (id: string, title: string) => Promise<void>;

  /* Profile */
  profileEntries: ProfileEntry[];
  refreshProfile: () => Promise<void>;
  addProfileEntry: (key: string, value: string, category?: string) => Promise<void>;
  deleteProfileEntry: (id: number) => Promise<void>;

  /* UI state */
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  profileModalOpen: boolean;
  setProfileModalOpen: (v: boolean) => void;
  memoryPanelOpen: Record<string, boolean>;
  toggleMemoryPanel: (messageId: string) => void;

  /* AI inspection */
  debugMode: boolean;
  toggleDebugMode: () => void;
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (v: boolean) => void;

  /* Thread & Research panels */
  threadPanelOpen: boolean;
  toggleThreadPanel: () => void;
  researchDashOpen: boolean;
  toggleResearchDash: () => void;
  threads: api.Thread[];
  insights: api.Insight[];
  concepts: api.ConceptLink[];
  refreshThreads: (conversationId: string) => Promise<void>;
  refreshInsights: (conversationId: string) => Promise<void>;
  refreshConcepts: (conversationId: string) => Promise<void>;
}

/* ── Store ───────────────────────────────────────────────────────────────── */

export const useChatStore = create<ChatStore>((set, get) => ({
  /* ── Conversations ── */
  conversations: [],
  conversationId: null,

  // Clear side-panel data so the previous conversation's threads/insights/
  // concepts are not shown while the new conversation's data loads.
  setConversationId: (id) =>
    set({ conversationId: id, threads: [], insights: [], concepts: [] }),

  addConversation: (c) =>
    set((s) => ({ conversations: [c, ...s.conversations] })),

  refreshConversations: async () => {
    try {
      const { conversations } = await api.listConversations();
      set({ conversations });
    } catch (e) {
      console.error('Failed to refresh conversations', e);
    }
  },

  removeConversation: async (id) => {
    try {
      await api.deleteConversation(id);
      set((s) => ({
        conversations: s.conversations.filter((c) => c.id !== id),
        conversationId: s.conversationId === id ? null : s.conversationId,
      }));
    } catch (e) {
      console.error('Failed to delete conversation', e);
    }
  },

  renameConversation: async (id, title) => {
    try {
      await api.renameConversation(id, title);
      set((s) => ({
        conversations: s.conversations.map((c) =>
          c.id === id ? { ...c, title } : c,
        ),
      }));
    } catch (e) {
      console.error('Failed to rename conversation', e);
    }
  },

  /* ── Profile ── */
  profileEntries: [],

  refreshProfile: async () => {
    try {
      const { entries } = await api.getProfile();
      set({ profileEntries: entries });
    } catch (e) {
      console.error('Failed to refresh profile', e);
    }
  },

  addProfileEntry: async (key, value, category = 'general') => {
    try {
      await api.addProfileEntry(key, value, category);
      await get().refreshProfile();
    } catch (e) {
      console.error('Failed to add profile entry', e);
    }
  },

  deleteProfileEntry: async (id) => {
    try {
      await api.deleteProfileEntry(id);
      set((s) => ({
        profileEntries: s.profileEntries.filter((e) => e.id !== id),
      }));
    } catch (e) {
      console.error('Failed to delete profile entry', e);
    }
  },

  /* ── UI state ── */
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  profileModalOpen: false,
  setProfileModalOpen: (v) => set({ profileModalOpen: v }),

  memoryPanelOpen: {},
  toggleMemoryPanel: (messageId) =>
    set((s) => ({
      memoryPanelOpen: {
        ...s.memoryPanelOpen,
        [messageId]: !s.memoryPanelOpen[messageId],
      },
    })),

  /* ── AI inspection ── */
  debugMode: false,
  toggleDebugMode: () => set((s) => ({ debugMode: !s.debugMode })),

  commandPaletteOpen: false,
  setCommandPaletteOpen: (v) => set({ commandPaletteOpen: v }),

  /* ── Thread & Research panels ── */
  threadPanelOpen: false,
  toggleThreadPanel: () => set((s) => ({ threadPanelOpen: !s.threadPanelOpen })),

  researchDashOpen: false,
  toggleResearchDash: () => set((s) => ({ researchDashOpen: !s.researchDashOpen })),

  threads: [],
  insights: [],
  concepts: [],

  refreshThreads: async (conversationId) => {
    try {
      const { threads } = await api.getThreads(conversationId);
      // Ignore responses for a conversation that is no longer active
      if (get().conversationId === conversationId) set({ threads });
    } catch (e) {
      console.error('Failed to refresh threads', e);
    }
  },

  refreshInsights: async (conversationId) => {
    try {
      const { insights } = await api.getInsights(conversationId);
      // Ignore responses for a conversation that is no longer active
      if (get().conversationId === conversationId) set({ insights });
    } catch (e) {
      console.error('Failed to refresh insights', e);
    }
  },

  refreshConcepts: async (conversationId) => {
    try {
      const { concepts } = await api.getConcepts(conversationId);
      // Ignore responses for a conversation that is no longer active
      if (get().conversationId === conversationId) set({ concepts });
    } catch (e) {
      console.error('Failed to refresh concepts', e);
    }
  },
}));
