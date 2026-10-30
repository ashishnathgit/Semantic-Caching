import { useState } from "react";
import { DEMO_SCRIPT } from "./demoScript";

interface Props {
  query: string;
  onQuery: (q: string) => void;
  onAsk: () => void;
  pending: boolean;
  forceRefresh: boolean;
  onForceRefresh: (v: boolean) => void;
  followUpHint: string | null;
  hasThread: boolean;
}

export function QuestionBox({ query, onQuery, onAsk, pending, forceRefresh, onForceRefresh, followUpHint, hasThread }: Props) {
  const [menu, setMenu] = useState(false);
  const [next, setNext] = useState(0);

  function load(i: number) {
    onQuery(DEMO_SCRIPT[i].query);
    setNext((i + 1) % DEMO_SCRIPT.length);
    setMenu(false);
    document.getElementById("query")?.focus();
  }

  return (
    <form onSubmit={(e) => { e.preventDefault(); onAsk(); }} className="card space-y-3 p-5" noValidate>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor="query" className="text-[15px] font-medium text-ink">
          {hasThread ? "Your next question" : "Ask a question"}
        </label>
        <div className="relative">
          <button type="button" onClick={() => setMenu((m) => !m)} aria-expanded={menu} aria-haspopup="menu"
            className="rounded-lg px-2 py-1 text-[13px] font-medium text-query hover:bg-paper">
            Demo script ▾
          </button>
          {menu && (
            <div role="menu" className="card absolute right-0 z-20 mt-1 w-[22rem] p-2">
              {DEMO_SCRIPT.map((d, i) => (
                <button key={d.label} type="button" role="menuitem" onClick={() => load(i)}
                  className={`block w-full rounded-lg px-3 py-2 text-left hover:bg-paper ${i === next ? "bg-paper" : ""}`}>
                  <span className="block text-[13px] font-semibold text-ink">
                    {i + 1}. {d.label} <span className="font-normal text-ink-faint">· expect {d.expect}</span>
                  </span>
                  <span className="block truncate text-[13px] text-ink-soft">{d.query}</span>
                </button>
              ))}
              <button type="button" onClick={() => load(next)}
                className="mt-1 w-full rounded-lg bg-ink px-3 py-2 text-[13px] font-medium text-sheet">
                Load next: {DEMO_SCRIPT[next].label}
              </button>
            </div>
          )}
        </div>
      </div>
      <div className="flex gap-2">
        <input id="query" value={query} onChange={(e) => onQuery(e.target.value)} autoComplete="off"
          placeholder={hasThread ? "Ask a follow-up or something new" : "cheap flight tickets to goa"}
          className="h-12 min-w-0 flex-1 rounded-xl border border-rule bg-paper px-4 text-[17px] text-ink outline-none placeholder:text-ink-faint focus-visible:border-blurple focus-visible:ring-2 focus-visible:ring-blurple/30" />
        <button type="submit" disabled={pending}
          className="h-12 rounded-xl bg-blurple px-6 text-[16px] font-semibold text-white shadow-sm hover:bg-query disabled:opacity-60">
          {pending ? "…" : "Ask"}
        </button>
      </div>
      <label className="flex w-fit items-center gap-2 text-[14px] text-ink-soft">
        <input type="checkbox" checked={forceRefresh} onChange={(e) => onForceRefresh(e.target.checked)} className="threshold h-4 w-4" />
        Force refresh (skip cache)
      </label>
      {followUpHint && <p className="m-0 text-[13px] leading-snug text-query">{followUpHint}</p>}
    </form>
  );
}
