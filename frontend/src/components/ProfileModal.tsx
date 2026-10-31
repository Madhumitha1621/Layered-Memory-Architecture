import { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Plus, Trash2, Loader2 } from 'lucide-react';
import { useChatStore } from '../store';

const modalVariants = {
  hidden: { opacity: 0, scale: 0.96, y: 12 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.25, ease: [0.16, 1, 0.3, 1] },
  },
  exit: {
    opacity: 0,
    scale: 0.96,
    y: 12,
    transition: { duration: 0.15 },
  },
};

export default function ProfileModal() {
  const {
    profileEntries,
    setProfileModalOpen,
    addProfileEntry,
    deleteProfileEntry,
  } = useChatStore();

  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [category, setCategory] = useState('general');
  const [adding, setAdding] = useState(false);

  const handleAdd = async () => {
    if (!key.trim() || !value.trim()) return;
    setAdding(true);
    await addProfileEntry(key.trim(), value.trim(), category);
    setKey('');
    setValue('');
    setCategory('general');
    setAdding(false);
  };

  const categories = ['general', 'personal', 'professional', 'preferences', 'health', 'education', 'other'];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center modal-backdrop">
      <motion.div
        variants={modalVariants}
        initial="hidden"
        animate="visible"
        exit="exit"
        className="glass border border-surface-2/40 rounded-2xl shadow-elevated-lg w-full max-w-lg
                   max-h-[80vh] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-2/30">
          <h2 className="text-base font-semibold text-white tracking-tight">Profile & Memory</h2>
          <button
            onClick={() => setProfileModalOpen(false)}
            title="Close"
            className="p-1.5 rounded-lg hover:bg-surface-1 text-zinc-500
                       hover:text-white transition-all duration-200"
          >
            <X size={16} />
          </button>
        </div>

        {/* Entries list */}
        <div className="flex-1 overflow-y-auto p-6">
          {profileEntries.length === 0 ? (
            <p className="text-zinc-500 text-sm text-center py-8 leading-relaxed">
              No profile data stored yet.  Share facts about yourself in conversation
              and they'll appear here automatically.
            </p>
          ) : (
            <div className="space-y-1.5">
              {profileEntries.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-center gap-3 p-3 rounded-xl bg-surface-1/30 group
                             hover:bg-surface-1/50 transition-colors duration-200"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2">
                      <span className="text-xs font-medium text-accent">
                        {entry.key.replace(/_/g, ' ')}
                      </span>
                      <span className="text-2xs text-zinc-600">{entry.category}</span>
                    </div>
                    <p className="text-sm text-zinc-300 truncate">{entry.value}</p>
                  </div>
                  <button
                    onClick={() => deleteProfileEntry(entry.id)}
                    title="Delete entry"
                    className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100
                               text-zinc-600 hover:text-red-400 hover:bg-red-500/10
                               transition-all duration-200"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Add entry form */}
        <div className="border-t border-surface-2/30 p-6">
          <h3 className="text-xs font-medium text-zinc-400 mb-3 tracking-tight">Add manually</h3>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <input
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="Key (e.g. name)"
              className="px-3 py-2 text-sm bg-surface-1/50 border border-surface-2/30 rounded-xl
                         text-zinc-300 placeholder:text-zinc-600 outline-none
                         focus:border-accent/30 transition-colors duration-200"
            />
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              title="Category"
              aria-label="Category"
              className="px-3 py-2 text-sm bg-surface-1/50 border border-surface-2/30 rounded-xl
                         text-zinc-300 outline-none focus:border-accent/30
                         transition-colors duration-200"
            >
              {categories.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="Value"
              onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
              className="flex-1 px-3 py-2 text-sm bg-surface-1/50 border border-surface-2/30 rounded-xl
                         text-zinc-300 placeholder:text-zinc-600 outline-none
                         focus:border-accent/30 transition-colors duration-200"
            />
            <button
              onClick={handleAdd}
              disabled={!key.trim() || !value.trim() || adding}
              className="px-4 py-2 text-sm rounded-xl bg-accent text-white font-medium
                         hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed
                         shadow-glow-sm disabled:shadow-none
                         transition-all duration-200 flex items-center gap-1.5"
            >
              {adding ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
              Add
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
