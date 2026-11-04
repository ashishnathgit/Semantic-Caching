import type { Route, RouteStep } from "../types";

// exact? -> guards (known words, rho, negation) -> max score vs tau -> angle -> HIT / LLM / BLOCKED.
const STATUS: Record<RouteStep["status"], string> = {
  yes: "border-hit-fill text-hit",
  pass: "border-hit-fill text-hit",
  no: "border-rule text-ink-soft",
  fail: "border-miss-fill text-miss",
  info: "border-query text-query",
  off: "border-rule text-ink-faint",
  skipped: "border-rule text-ink-faint",
};
const MARK: Record<RouteStep["status"], string> = { yes: "✓", pass: "✓", no: "✗", fail: "✗", info: "i", off: "–", skipped: "–" };
const OUTCOME: Record<Route["outcome"]["kind"], string> = {
  HIT: "bg-hit-fill text-navy",
  LLM: "bg-blurple text-white",
  BLOCKED: "bg-miss-fill text-navy",
};

export function RouteTrace({ route, large = false }: { route: Route; large?: boolean }) {
  const text = large ? "text-[18px]" : "text-[13px]";
  const small = large ? "text-[18px]" : "text-[12px]";
  return (
    <ol className={`m-0 flex list-none flex-wrap items-stretch gap-x-1 gap-y-2 p-0 ${text}`} aria-label="Route of this question">
      {route.steps.map((s) => (
        <li key={s.id} className="flex items-center gap-1">
          <span className={`rounded-lg border px-2 py-1 leading-tight ${STATUS[s.status]}`} title={s.detail}>
            <span className="font-medium">{s.label}</span>{" "}
            <span className="num">{s.value}</span> <span aria-hidden="true">{MARK[s.status]}</span>
            {large && s.detail && <span className={`block text-ink-faint ${small}`}>{s.detail}</span>}
          </span>
          <span aria-hidden="true" className="text-ink-faint">→</span>
        </li>
      ))}
      <li className="flex items-center">
        <span className={`rounded-lg px-2.5 py-1 font-bold tracking-wide ${OUTCOME[route.outcome.kind]}`} title={route.outcome.detail}>
          {route.outcome.kind}
          {route.outcome.guard ? ` (${route.outcome.guard})` : ""}
          {route.outcome.then && route.outcome.kind === "BLOCKED" ? ` → ${route.outcome.then}` : ""}
        </span>
      </li>
    </ol>
  );
}
