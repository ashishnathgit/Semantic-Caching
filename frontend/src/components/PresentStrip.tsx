import { useState } from "react";
import type { AskResult } from "../types";
import { DEMO_SCRIPT } from "./demoScript";
import { formatMs } from "./sources";

// Presentation mode, left side: the question box, Ask, the verdict badge and the answer in
// two lines. Everything else is left to the math panel.
const BADGE: Record<AskResult["answer_source"], string> = {
  CACHE: "bg-hit-fill text-navy",
  LLM: "bg-blurple text-white",
  "FOLLOW-UP": "bg-violet text-navy",
  FALLBACK: "bg-miss-fill text-navy",
  NONE: "bg-rule text-ink",
};

interface Props {
  profile: string | null;
  query: string;
  onQuery: (q: string) => void;
  onAsk: () => void;
  pending: string | null;
  result: AskResult | null;
  error: string | null;
  warming: boolean;
}

export function PresentStrip({ profile, query, onQuery, onAsk, pending, result, error, warming }: Props) {
  const [next, setNext] = useState(0);
  const t = result?.timings_ms;
  return (
    <aside className="card flex h-full min-h-0 flex-col gap-[1.8vh] p-[2vh]" aria-label="Question">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[2vh] font-semibold tracking-tight">
          Semantic cache{profile ? <span className="text-ink-faint"> · {profile.toUpperCase()}</span> : null}
        </span>
        {warming && <span className="animate-pulse text-[1.6vh] text-miss">Model warming up…</span>}
      </div>
      <form onSubmit={(e) => { e.preventDefault(); onAsk(); }} className="space-y-[1.2vh]" noValidate>
        <label htmlFor="query" className="sr-only">Question</label>
        <input id="query" value={query} onChange={(e) => onQuery(e.target.value)} autoComplete="off" placeholder="Ask a question"
          className="h-[6vh] w-full rounded-xl border border-rule bg-paper px-[1.4vh] text-[2.6vh] text-ink outline-none focus-visible:border-blurple focus-visible:ring-2 focus-visible:ring-blurple/30" />
        <div className="flex gap-[1vh]">
          <button type="submit" disabled={pending !== null}
            className="h-[5.6vh] flex-1 rounded-xl bg-blurple text-[2.6vh] font-semibold text-white disabled:opacity-60">
            {pending ? "Asking…" : "Ask"}
          </button>
          <button type="button" title={`Load demo question ${next + 1}: ${DEMO_SCRIPT[next].label}`}
            onClick={() => { onQuery(DEMO_SCRIPT[next].query); setNext((next + 1) % DEMO_SCRIPT.length); }}
            className="h-[5.6vh] rounded-xl border border-rule px-[1.4vh] text-[2vh] text-query">
            Demo {next + 1}
          </button>
        </div>
      </form>

      <div className="min-h-0 space-y-[1vh]" aria-live="polite">
        {error ? (
          <p className="m-0 text-[2.2vh] text-miss">{error}</p>
        ) : pending ? (
          <p className="m-0 line-clamp-2 text-[2.4vh] text-ink-soft">{pending}</p>
        ) : result ? (
          <>
            <div className="flex flex-wrap items-center gap-[1vh]">
              <span className={`rounded-lg px-[1vh] py-[0.4vh] text-[2.2vh] font-bold tracking-wide ${BADGE[result.answer_source]}`}>
                {result.answer_source}
              </span>
              {result.similarity_pct !== null && <span className="num text-[2.2vh]">{result.similarity_pct.toFixed(0)}%</span>}
              {t && (
                <span className="num text-[2vh] text-ink-soft">
                  {result.hit ? `≈${formatMs(t.embed + t.lookup)}` : formatMs(t.total)}
                </span>
              )}
            </div>
            <p className="m-0 line-clamp-2 text-[2.4vh] leading-snug text-ink" title={result.answer}>{result.answer}</p>
          </>
        ) : (
          <p className="m-0 text-[2.2vh] text-ink-soft">Type a question: the math runs on the right.</p>
        )}
      </div>

      <p className="m-0 mt-auto text-[1.6vh] leading-snug text-ink-faint">
        ← → or 1–6: stages · Space: play · R: replay · P: leave
      </p>
    </aside>
  );
}
