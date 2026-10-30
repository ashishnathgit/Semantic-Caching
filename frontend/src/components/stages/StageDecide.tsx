import type { Explanation, MatchItem } from "../../types";
import { AngleFan } from "../AngleFan";
import { angleOf } from "../sources";
import { short } from "./heat";

export type VerdictKind = "HIT" | "MISS" | "FOLLOW-UP" | "BLOCKED";

/** The decision for the current τ (the slider moves faster than a new explain arrives). */
export function decide(x: Explanation, tau: number): { kind: VerdictKind; text: string } {
  const best = x.matches[0];
  if (x.follow_up && !x.cache_used) {
    return { kind: "FOLLOW-UP", text: `Depends on the conversation (${x.follow_up_reason}): it skips the cache and goes to the model with the history.` };
  }
  if (x.verdict === "exact") return { kind: "HIT", text: "An exact repeat of a cached question: answered from the cache." };
  if (x.verdict === "no_known_words") return { kind: "BLOCKED", text: "None of these words are in the vocabulary, so the question has no vector. Only an exact repeat could hit." };
  if (x.verdict === "guard") {
    return { kind: "BLOCKED", text: `Only ${x.coverage.known} of ${x.coverage.total} words are known (at least ${Math.round(x.coverage.min_ratio * 100)}% are needed), so only an exact repeat can hit.` };
  }
  if (x.verdict === "empty" || !best) return { kind: "MISS", text: "The cache is empty, so the model answers." };
  const how = best.scored_by === "tfidf" ? " (scored by word overlap, TF-IDF)" : "";
  if (x.score >= tau) return { kind: "HIT", text: `${x.score.toFixed(3)} ≥ τ ${tau.toFixed(2)}${how}: the cached answer is reused, no model call.` };
  return { kind: "MISS", text: `${x.score.toFixed(3)} < τ ${tau.toFixed(2)}${how}, ${(tau - x.score).toFixed(3)} short: the model answers, and the answer is cached.` };
}

export const VERDICT_STYLE: Record<VerdictKind, string> = {
  HIT: "bg-hit-fill text-navy",
  MISS: "bg-miss-fill text-navy",
  "FOLLOW-UP": "bg-p-accent text-navy",
  BLOCKED: "bg-miss-fill text-navy",
};

export function StageDecide({ x, tau, onTau, newRowId, compact = false }: {
  x: Explanation; tau: number; onTau: (t: number) => void; newRowId: number | null; compact?: boolean;
}) {
  const v = decide(x, tau);
  const top = x.matches.slice(0, compact ? 3 : 5);
  const fan: MatchItem[] = x.matches.slice(0, 3).map((m) => ({
    id: m.id, query: m.query, score: m.score, source: m.source, owner: m.owner, rho: m.rho,
    pair_rho: m.pair_rho, scored_by: m.scored_by, hidden: m.hidden,
  }));

  return (
    <div className="grid h-full min-h-0 items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
      <div className="space-y-4">
        <div className="flex items-center gap-4" aria-live="polite">
          <span className={`rounded-xl px-5 py-2 text-[30px] leading-none font-bold tracking-wide present:text-[5vh] ${VERDICT_STYLE[v.kind]}`}>{v.kind}</span>
          <p className="m-0 text-[15px] leading-snug text-p-soft present:text-[2.6vh]">{v.text}</p>
        </div>
        <ul className="relative m-0 list-none space-y-2 p-0">
          {top.map((m) => {
            const inside = m.exact || m.score >= tau;
            return (
              <li key={m.id} className="space-y-1">
                <div className="flex items-baseline justify-between gap-3 text-[14px] present:text-[2.6vh]">
                  <span className={`truncate ${m.hidden ? "text-p-faint italic" : "text-p-text"}`}>
                    {short(m.query, 58)}
                    {m.id === newRowId && <span className="text-cyan"> · new row</span>}
                  </span>
                  <span className={`num ${inside ? "text-hit-fill" : "text-p-soft"}`}>{m.score.toFixed(3)}</span>
                </div>
                <div className="relative h-2.5 rounded-full bg-navy-2 present:h-[1.4vh]">
                  <div className={`absolute inset-y-0 left-0 rounded-full transition-[width] duration-500 ${inside ? "bg-hit-fill" : "bg-p-faint/60"}`}
                    style={{ width: `${Math.max(0, Math.min(1, m.score)) * 100}%` }} />
                </div>
              </li>
            );
          })}
          {top.length > 0 && (
            <div className="pointer-events-none absolute -top-1 -bottom-1 w-0.5 bg-p-text transition-[left] duration-300"
              style={{ left: `${tau * 100}%` }} aria-hidden="true" />
          )}
        </ul>
        <div>
          <div className="flex items-baseline justify-between gap-4">
            <label htmlFor="tau" className="text-[14px] text-p-soft present:text-[2.4vh]">Threshold τ</label>
            <span className="num text-[15px] text-p-text present:text-[2.4vh]">
              {tau.toFixed(2)} <span className="text-p-faint">· cone {angleOf(tau).toFixed(1)}°</span>
            </span>
          </div>
          <input id="tau" type="range" min={0.3} max={0.99} step={0.01} value={tau}
            onChange={(e) => onTau(Number(e.target.value))} className="threshold mt-1 w-full" />
        </div>
      </div>
      <div className="stage-fig h-full">
        <AngleFan matches={fan} tau={tau} exact={x.verdict === "exact"} hoveredId={null} onHover={() => undefined} hasQuery />
      </div>
    </div>
  );
}
