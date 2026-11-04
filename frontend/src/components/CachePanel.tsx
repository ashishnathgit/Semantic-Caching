import { useState } from "react";
import type { CacheEntry } from "../types";
import { sourceLabel } from "./sources";

interface Props {
  entries: CacheEntry[];
  user: string;
  isAdmin: boolean;
  onAdd: (question: string, answer: string, shared: boolean, ttlSeconds: number | null) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onDeleteMany: (ids: number[]) => Promise<void>;
  onReset: () => Promise<void>;
  onResetDemo: () => Promise<void>;
  onResetMetrics: () => Promise<void>;
}

function Confirm({ label, confirm, onConfirm }: { label: string; confirm: string; onConfirm: () => Promise<void> }) {
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return asking ? (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-2">
        <button type="button" className="h-9 rounded-lg bg-miss px-4 text-[14px] font-medium text-sheet"
          onClick={async () => {
            try {
              await onConfirm();
              setAsking(false);
              setError(null);
            } catch (err) {
              setError((err as Error).message);
            }
          }}>
          {confirm}
        </button>
        <button type="button" onClick={() => setAsking(false)} className="h-9 rounded-lg border border-rule px-4 text-[14px] text-ink">
          Cancel
        </button>
      </div>
      {error && <p role="alert" className="m-0 text-[13px] text-miss">{error}</p>}
    </div>
  ) : (
    <button type="button" onClick={() => setAsking(true)}
      className="h-9 rounded-lg border border-rule bg-sheet px-4 text-[14px] text-ink hover:border-ink-faint">
      {label}
    </button>
  );
}

export function CachePanel({ entries, user, isAdmin, onAdd, onDelete, onDeleteMany, onReset, onResetDemo, onResetMetrics }: Props) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [shared, setShared] = useState(false);
  const [ttlHours, setTtlHours] = useState("");
  const [rowError, setRowError] = useState<string | null>(null);
  // Admins may delete any entry (FAQ and other users' private ones too); others only their own.
  const canDelete = (e: CacheEntry) => isAdmin || e.owner === user;
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);

  const shown = filter.trim()
    ? entries.filter((e) => e.query.toLowerCase().includes(filter.trim().toLowerCase()))
    : entries;

  const shownIds = shown.map((e) => e.id);
  const live = new Set(entries.map((e) => e.id));
  const chosen = [...selected].filter((id) => live.has(id));
  const allShown = shownIds.length > 0 && shownIds.every((id) => selected.has(id));
  const toggle = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const toggleShown = () =>
    setSelected((prev) => {
      const next = new Set(prev);
      for (const id of shownIds) {
        if (allShown) next.delete(id);
        else next.add(id);
      }
      return next;
    });

  async function deleteSelected() {
    try {
      await onDeleteMany(chosen);
      setSelected(new Set());
      setConfirmBulk(false);
      setRowError(null);
    } catch (err) {
      setRowError((err as Error).message);
    }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || !answer.trim()) {
      setFormError("Enter both a question and its answer.");
      return;
    }
    try {
      const hours = ttlHours.trim() ? Number(ttlHours) : null;
      if (hours !== null && (!Number.isFinite(hours) || hours <= 0)) {
        setFormError("Expiry must be a number of hours above 0, or empty for never.");
        return;
      }
      await onAdd(question.trim(), answer.trim(), isAdmin && shared, hours === null ? null : hours * 3600);
      setQuestion("");
      setAnswer("");
      setTtlHours("");
      setFormError(null);
    } catch (err) {
      setFormError((err as Error).message);
    }
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_20rem]">
      <div className="min-w-0">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <label className="sr-only" htmlFor="filter">
            Filter cached questions
          </label>
          <input
            id="filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by text"
            className="h-9 w-56 rounded-md border border-rule bg-sheet px-3 text-[14px] outline-none focus-visible:border-query"
          />
          <span className="num text-[13px] text-ink-faint">
            {shown.length} of {entries.length}
          </span>
          {isAdmin && chosen.length > 0 && (
            confirmBulk ? (
              <span className="flex flex-wrap items-center gap-2">
                <span className="text-[13px] text-ink-soft">
                  Delete {chosen.length} cached question{chosen.length === 1 ? "" : "s"} for everyone?
                  {chosen.some((id) => entries.find((e) => e.id === id)?.source === "faq") && " FAQ answers come back with Reset demo."}
                </span>
                <button type="button" onClick={deleteSelected} className="h-9 rounded-lg bg-miss px-3 text-[14px] font-medium text-sheet">
                  Yes, delete
                </button>
                <button type="button" onClick={() => setConfirmBulk(false)} className="h-9 rounded-lg border border-rule px-3 text-[14px] text-ink">
                  Cancel
                </button>
              </span>
            ) : (
              <button type="button" onClick={() => setConfirmBulk(true)}
                className="h-9 rounded-lg border border-miss px-3 text-[14px] font-medium text-miss hover:bg-miss-wash">
                Delete selected ({chosen.length})
              </button>
            )
          )}
          {rowError && (
            <span role="alert" className="text-[13px] text-miss">
              {rowError}
            </span>
          )}
        </div>
        <div className="max-h-[26rem] overflow-y-auto border-t border-rule">
          <table className="w-full border-collapse text-left text-[14px]">
            <thead className="sticky top-0 bg-paper text-[12px] text-ink-faint">
              <tr>
                {isAdmin && (
                  <th className="w-8 py-2 pr-2 font-normal">
                    <input type="checkbox" checked={allShown} onChange={toggleShown}
                      aria-label={allShown ? "Unselect the listed questions" : "Select every listed question"} />
                  </th>
                )}
                <th className="py-2 pr-3 font-normal">Question</th>
                <th className="py-2 pr-3 font-normal">Source</th>
                <th className="py-2 pr-3 font-normal">Seen by</th>
                <th className="num py-2 pr-3 text-right font-normal">Hits</th>
                <th className="py-2 font-normal">
                  <span className="sr-only">Remove</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {shown.map((e) => (
                <tr key={e.id} className={`border-t border-rule align-top ${selected.has(e.id) ? "bg-miss-wash" : ""}`}>
                  {isAdmin && (
                    <td className="py-2 pr-2">
                      <input type="checkbox" checked={selected.has(e.id)} onChange={() => toggle(e.id)}
                        aria-label={`Select “${e.query}”`} />
                    </td>
                  )}
                  <td className="py-2 pr-3 text-ink">
                    <details>
                      <summary className="cursor-pointer list-none">{e.query}</summary>
                      <p className="mt-1 mb-0 whitespace-pre-wrap text-[13px] leading-snug text-ink-soft">{e.answer}</p>
                    </details>
                  </td>
                  <td className="py-2 pr-3 whitespace-nowrap text-ink-soft">
                    {sourceLabel(e.source)}
                    {e.expires_at ? (
                      <span className={`ml-1 text-[12px] ${e.expires_at * 1000 < Date.now() ? "text-miss" : "text-ink-faint"}`}>
                        {e.expires_at * 1000 < Date.now() ? "· expired" : `· expires ${new Date(e.expires_at * 1000).toLocaleString()}`}
                      </span>
                    ) : null}
                  </td>
                  <td className={`py-2 pr-3 whitespace-nowrap ${e.owner ? "text-query" : "text-ink-soft"}`}>
                    {e.owner ? (e.owner === user ? "Only you" : `Only ${e.owner}`) : "Everyone"}
                  </td>
                  <td className="num py-2 pr-3 text-right text-ink-soft">{e.hits}</td>
                  <td className="py-2 text-right">
                    {canDelete(e) && (
                      <button
                        type="button"
                        onClick={async () => {
                          try {
                            await onDelete(e.id);
                            setRowError(null);
                          } catch (err) {
                            setRowError((err as Error).message);
                          }
                        }}
                        className="rounded px-2 py-0.5 text-[13px] text-ink-faint hover:bg-paper hover:text-miss"
                        aria-label={`Remove “${e.query}”`}
                      >
                        Remove
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="space-y-8">
        <form onSubmit={submit} className="space-y-3" noValidate>
          <h3 className="m-0 font-math text-[17px] font-normal">Add an answer</h3>
          <p className="m-0 text-[13px] leading-snug text-ink-soft">
            {isAdmin && shared
              ? "Everyone will get this answer for this question and its paraphrases."
              : "Only you will get this answer; other people keep getting the shared one."}
          </p>
          <div>
            <label htmlFor="new-q" className="mb-1 block text-[13px] text-ink-soft">
              Question
            </label>
            <input
              id="new-q"
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                setFormError(null);
              }}
              placeholder="Where is the exam hall?"
              className="h-10 w-full rounded-md border border-rule bg-sheet px-3 text-[15px] outline-none focus-visible:border-query"
            />
          </div>
          <div>
            <label htmlFor="new-a" className="mb-1 block text-[13px] text-ink-soft">
              Answer
            </label>
            <textarea
              id="new-a"
              value={answer}
              onChange={(e) => {
                setAnswer(e.target.value);
                setFormError(null);
              }}
              rows={3}
              placeholder="Room 204, second floor of the academic block."
              className="w-full rounded-md border border-rule bg-sheet px-3 py-2 text-[15px] outline-none focus-visible:border-query"
            />
          </div>
          <div>
            <label htmlFor="new-ttl" className="mb-1 block text-[13px] text-ink-soft">
              Expires after (hours, optional)
            </label>
            <input id="new-ttl" inputMode="decimal" value={ttlHours} onChange={(e) => setTtlHours(e.target.value)}
              placeholder="never"
              className="h-10 w-32 rounded-md border border-rule bg-sheet px-3 text-[15px] outline-none focus-visible:border-query" />
          </div>
          {isAdmin && (
            <label className="flex items-center gap-2 text-[13px] text-ink-soft">
              <input type="checkbox" checked={shared} onChange={(e) => setShared(e.target.checked)} />
              Share with everyone (admin)
            </label>
          )}
          {formError && (
            <p role="alert" className="m-0 text-[13px] text-miss">
              {formError}
            </p>
          )}
          <button type="submit" className="h-9 rounded-md bg-ink px-4 text-[14px] font-medium text-sheet hover:bg-ink-soft">
            Add to cache
          </button>
        </form>

        <div className="space-y-2">
          <h3 className="m-0 text-[15px] font-semibold">Start over</h3>
          <p className="m-0 text-[13px] leading-snug text-ink-soft">
            {isAdmin
              ? "Removes everything except the FAQ answers, for every user, and sets the hit counters to zero."
              : "Removes the answers you added and sets your counters to zero. Shared answers stay."}
          </p>
          {confirmReset ? (
            <div className="flex gap-2">
              <button
                type="button"
                onClick={async () => {
                  await onReset();
                  setConfirmReset(false);
                }}
                className="h-9 rounded-md bg-miss px-4 text-[14px] font-medium text-sheet"
              >
                {isAdmin ? "Reset the shared cache" : "Remove my answers"}
              </button>
              <button
                type="button"
                onClick={() => setConfirmReset(false)}
                className="h-9 rounded-md border border-rule px-4 text-[14px] text-ink"
              >
                Keep it
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmReset(true)}
              className="h-9 rounded-md border border-rule px-4 text-[14px] text-ink hover:border-ink-faint"
            >
              {isAdmin ? "Reset to the FAQ" : "Start over"}
            </button>
          )}
        </div>

        {isAdmin && (
          <div className="space-y-3 rounded-xl border border-rule p-4">
            <h3 className="m-0 text-[15px] font-semibold">Admin: before a demo</h3>
            <div className="space-y-1">
              <p className="m-0 text-[13px] leading-snug text-ink-soft">
                Reset demo: exactly the 43 FAQ answers in the cache, the metrics archived and started over, and every
                user's counters at zero. Conversations are kept.
              </p>
              <Confirm label="Reset demo" confirm="Yes, reset the demo" onConfirm={onResetDemo} />
            </div>
            <div className="space-y-1">
              <p className="m-0 text-[13px] leading-snug text-ink-soft">Reset metrics only (the old events file is kept).</p>
              <Confirm label="Reset metrics" confirm="Yes, reset the metrics" onConfirm={onResetMetrics} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
