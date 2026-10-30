import type { AskResult } from "../types";
import { formatMs, sourceLabel } from "./sources";

const BADGE: Record<AskResult["answer_source"], string> = {
  CACHE: "bg-hit-fill text-navy",
  LLM: "bg-blurple text-white",
  "FOLLOW-UP": "bg-violet text-navy",
  FALLBACK: "bg-miss-fill text-navy",
  NONE: "bg-rule text-ink",
};

export function AnswerCard({ result, pending, error }: { result: AskResult | null; pending: string | null; error: string | null }) {
  if (error) return <div role="alert" className="card border-l-4 border-miss-fill p-5 text-[15px]">{error}</div>;
  if (pending) {
    return (
      <div className="card space-y-2 p-5" aria-live="polite">
        <p className="m-0 text-[13px] text-ink-faint">Asking</p>
        <p className="m-0 text-[17px] text-ink">{pending}</p>
        <div className="h-1 overflow-hidden rounded bg-rule"><div className="h-full w-1/3 animate-pulse bg-blurple" /></div>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="card p-5 text-[15px] leading-relaxed text-ink-soft">
        A question that means the same as one already answered comes straight from the cache; anything new goes to the
        model and is remembered. Watch the math on the right while you type.
      </div>
    );
  }
  const t = result.timings_ms;
  const time = result.hit ? `≈${formatMs(t.embed + t.lookup)}, no model call` : formatMs(t.total);
  const byWords = result.scored_by === "tfidf";
  return (
    <article className="card space-y-3 p-5" aria-live="polite">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className={`rounded-lg px-2.5 py-1 text-[13px] font-bold tracking-wide ${BADGE[result.answer_source]}`}>
          {result.answer_source}
        </span>
        <span className="num text-[14px] text-ink">
          {result.similarity_pct !== null ? `${result.similarity_pct.toFixed(1)}% similar` : "no similarity"}
        </span>
        <span className="num text-[14px] text-ink-soft">{time}</span>
        <span className="num text-[13px] text-ink-faint">ρ {result.rho.toFixed(2)}</span>
      </div>
      {result.hit && result.matched_query && (
        <p className="m-0 text-[14px] text-ink-soft">
          {result.lookup_method === "exact" ? "Exact repeat of" : byWords ? "Same words as" : "Same meaning as"}{" "}
          <q className="text-ink">{result.matched_query}</q>
          {result.matched_source ? ` · ${sourceLabel(result.matched_source)}` : ""}
          {byWords && <span className="text-miss"> · scored by word overlap (TF-IDF)</span>}
        </p>
      )}
      <div className="max-h-56 overflow-y-auto whitespace-pre-wrap text-[16px] leading-[1.6] text-ink">{result.answer}</div>
      {result.note && <p className="m-0 text-[13px] leading-snug text-ink-faint">{result.note}</p>}
      {result.backend_error && <p className="m-0 text-[12px] text-miss">Ollama error: {result.backend_error}</p>}
    </article>
  );
}
