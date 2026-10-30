import type { ThreadSummary } from "../types";

interface Props {
  threads: ThreadSummary[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
}

export function ThreadPicker({ threads, activeId, onSelect, onDelete }: Props) {
  return (
    <div className="flex items-center gap-2">
      <label htmlFor="thread" className="sr-only">Conversation</label>
      <select id="thread" value={activeId ?? ""} onChange={(e) => onSelect(e.target.value || null)}
        className="h-9 min-w-0 flex-1 rounded-lg border border-rule bg-sheet px-2 text-[14px] text-ink">
        <option value="">New conversation</option>
        {threads.map((t) => (
          <option key={t.id} value={t.id}>{t.title} ({t.message_count})</option>
        ))}
      </select>
      <button type="button" onClick={() => onSelect(null)} className="h-9 rounded-lg border border-rule bg-sheet px-3 text-[13px] text-ink hover:border-ink-faint">
        New
      </button>
      {activeId && (
        <button type="button" onClick={() => onDelete(activeId)} className="h-9 rounded-lg px-2 text-[13px] text-miss hover:bg-miss-wash">
          Delete
        </button>
      )}
    </div>
  );
}
