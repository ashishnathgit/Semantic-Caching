import type { Metrics } from "../types";
import { formatMs } from "./sources";

const DASH = "—";
const pct = (v: number | null) => (v === null ? DASH : `${(v * 100).toFixed(1)}%`);
const usd = (v: number | null | undefined) =>
  v === null || v === undefined ? DASH : v < 0.01 && v > 0 ? `$${v.toFixed(5)}` : `$${v.toFixed(2)}`;
const ms = (v: number | null) => (v === null ? DASH : formatMs(v));
const duration = (v: number | null) => (v === null ? DASH : v < 60_000 ? formatMs(v) : `${(v / 60_000).toFixed(1)} min`);

function Card({ label, value, sub, title, tone }: { label: string; value: string; sub?: string; title?: string; tone?: "hit" | "miss" }) {
  return (
    <div className="card min-w-0 p-4" title={title}>
      <p className="m-0 text-[13px] text-ink-soft">
        {label}
        {title && <span className="ml-1 text-ink-faint" aria-hidden="true">ⓘ</span>}
      </p>
      <p className={`num m-0 mt-1 truncate text-[26px] leading-tight font-semibold ${tone === "hit" ? "text-hit" : tone === "miss" ? "text-miss" : "text-ink"}`}>
        {value}
      </p>
      {sub && <p className="m-0 mt-1 text-[12px] leading-snug text-ink-faint">{sub}</p>}
    </div>
  );
}

interface Props {
  metrics: Metrics | null;
  scope: "global" | "user";
  onScope: (s: "global" | "user") => void;
  user: string | null;
}

export function StatsCards({ metrics, scope, onScope, user }: Props) {
  const s = metrics?.summary;
  const cost = s?.cost;
  const priceSub = cost ? `${cost.label}${cost.note ? ` (${cost.note})` : ""}` : "set a price in config.toml [metrics]";
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="m-0 text-[22px] font-semibold tracking-tight">Live stats</h2>
        <div role="radiogroup" aria-label="Whose numbers" className="flex rounded-xl bg-sheet p-1 shadow-[var(--shadow-card)]">
          {(["global", "user"] as const).map((v) => (
            <button key={v} type="button" role="radio" aria-checked={scope === v} onClick={() => onScope(v)}
              disabled={v === "user" && !user}
              className={`rounded-lg px-3 py-1 text-[13px] font-medium ${scope === v ? "bg-ink text-sheet" : "text-ink-soft"} disabled:opacity-40`}>
              {v === "global" ? "Everyone" : `Just ${user ?? "me"}`}
            </button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <Card label="Requests" value={s ? String(s.requests) : DASH} sub="questions asked" />
        <Card label="Cache hits" value={s ? String(s.hits) : DASH} tone="hit" />
        <Card label="Cache misses" value={s ? String(s.misses) : DASH} tone="miss" />
        <Card label={s?.hit_rate_label ?? "Hit rate (of cache lookups)"} value={pct(s?.hit_rate ?? null)}
          sub={`of all requests: ${pct(s?.hit_rate_of_requests ?? null)}`}
          title="Follow-ups and forced refreshes skip the cache, so they are left out: hits / (hits + misses)." />
        <Card label="LLM calls" value={s ? String(s.llm_calls) : DASH} sub="misses, follow-ups and refreshes" />
        <Card label="LLM fallbacks" value={s ? String(s.llm_fallbacks) : DASH} sub="Ollama failed, the placeholder answered" />
        <Card label="Est. LLM cost" value={usd(cost?.llm_cost_usd)} sub={priceSub} />
        <Card label="Est. savings" value={usd(cost?.savings_usd)} sub={cost ? "what the hits would have cost" : priceSub} />
        <Card label="Avg latency: hit vs LLM"
          value={s ? `${ms(s.latency_ms.avg_hit)} · ${ms(s.latency_ms.avg_llm)}` : DASH}
          sub={s ? `${s.latency_ms.hit_samples} hits, ${s.latency_ms.llm_samples} model answers` : undefined} />
        <Card label="Time saved" value={duration(s?.time_saved_ms ?? null)}
          title="Cache hits × (average LLM latency − average cache-hit latency)" sub="hits × (LLM − hit latency)" />
      </div>
    </div>
  );
}
