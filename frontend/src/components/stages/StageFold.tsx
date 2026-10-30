import { usePhases, useTween } from "../../hooks/useTween";
import type { Explanation } from "../../types";
import { heat, maxAbs, short } from "./heat";

// Stage 3: the question's rows of V_k slide out of a band of the matrix, each is scaled
// by its word's TF-IDF weight (the colours dim), and they merge into one row e.
// Normal layout, and a wider one for presentation mode, where the stage is short and wide.
const NARROW = { W: 680, HEAT_X: 16, HEAT_W: 300, STACK_X: 430, STACK_W: 234 };
const WIDE = { W: 920, HEAT_X: 16, HEAT_W: 440, STACK_X: 590, STACK_W: 314 };
const TOP = 34;
const MAX_ROWS = 9;

function Gauge({ label, rho, cut }: { label: string; rho: number | null; cut: number }) {
  const v = useTween(rho ?? 0, 700);
  const below = rho !== null && cut > 0 && rho < cut;
  return (
    <div className="min-w-0 flex-1 space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-[14px] present:text-[2.6vh]">
        <span className="text-p-soft">{label}</span>
        <span className={`num text-[17px] present:text-[2.8vh] ${rho === null ? "text-p-faint" : below ? "text-miss-fill" : "text-hit-fill"}`}>
          {rho === null ? "—" : `ρ ${v.toFixed(2)}`}
        </span>
      </div>
      <div className="relative h-2.5 rounded-full bg-navy-2" aria-hidden="true">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${below ? "bg-miss-fill" : "bg-hit-fill"}`}
          style={{ width: `${Math.min(1, v) * 100}%` }}
        />
        {cut > 0 && <div className="absolute -top-1.5 h-5.5 w-0.5 bg-p-text" style={{ left: `${cut * 100}%` }} />}
      </div>
    </div>
  );
}

export function StageFold({ x, playKey, wide = false }: { x: Explanation; playKey: number; wide?: boolean }) {
  const { W, HEAT_X, HEAT_W, STACK_X, STACK_W } = wide ? WIDE : NARROW;
  const phase = usePhases(3, 750, `${playKey}:${x.query}`);
  const k = x.model.rank;
  const own = [...x.term_rows].sort((a, b) => b.weight - a.weight).slice(0, MAX_ROWS);
  const ownIdx = new Set(own.map((r) => r.index));
  const band = [...x.vk_band.map((r) => ({ ...r, mine: false })), ...own.map((r) => ({ ...r, mine: true }))].sort(
    (a, b) => a.index - b.index,
  );
  const rh = Math.min(8, 214 / Math.max(1, band.length));
  const scale = maxAbs([...band.map((r) => r.vk)]);
  const cellH = HEAT_W / k;
  const cellS = STACK_W / k;
  const stackH = 15;
  const eY = TOP + own.length * (stackH + 4) + 22;
  const eScale = Math.max(...x.latent.map(Math.abs), 1e-9);
  const cut = x.model.min_latent_coverage;
  const best = x.matches[0];

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="stage-fig">
      <svg viewBox={`0 0 ${W} ${Math.max(eY + 40, TOP + band.length * rh + 30)}`} preserveAspectRatio="xMidYMin meet" role="img"
        aria-label={`The question's ${own.length} rows of V_k, weighted and summed into e with ${k} numbers`}>
        <text x={HEAT_X} y={20} fontSize={12} fill="var(--color-p-faint)">
          V_k: {band.length} of {x.vocab_size.toLocaleString()} rows shown, {k} columns
        </text>
        {band.map((r, i) =>
          r.mine ? null : (
            <g key={r.index} transform={`translate(${HEAT_X} ${TOP + i * rh})`}>
              {r.vk.map((v, j) => (
                <rect key={j} x={j * cellH} width={cellH + 0.3} height={rh - 0.6} fill={heat(v, scale)} />
              ))}
            </g>
          ),
        )}
        {/* the question's rows: in the band, then slid out, weighted, merged */}
        {band.map((r, i) => {
          if (!r.mine) return null;
          const s = own.findIndex((o) => o.index === r.index);
          const row = own[s];
          const out = phase >= 1;
          const weighted = phase >= 2;
          const merged = phase >= 3;
          const tx = out ? STACK_X : HEAT_X;
          const ty = merged ? eY : out ? TOP + s * (stackH + 4) : TOP + i * rh;
          const sx = out ? STACK_W / HEAT_W : 1;
          const sy = out ? stackH / rh : 1;
          return (
            <g key={r.index}>
              <g
                style={{
                  transform: `translate(${tx}px, ${ty}px) scale(${sx}, ${sy})`,
                  transition: "transform 700ms cubic-bezier(.2,.7,.2,1), opacity 500ms ease",
                  opacity: merged ? 0 : 1,
                }}
              >
                {r.vk.map((v, j) => (
                  <rect
                    key={j}
                    x={j * cellH}
                    width={cellH + 0.3}
                    height={rh - 0.6}
                    style={{ fill: heat(weighted ? v * row.weight : v, scale), transition: "fill 600ms ease" }}
                  />
                ))}
                <rect width={HEAT_W} height={rh - 0.6} fill="none" stroke="var(--color-p-accent)" strokeWidth={1 / sx} />
              </g>
              <text
                x={STACK_X - 8}
                y={TOP + s * (stackH + 4) + stackH - 3}
                textAnchor="end"
                fontSize={12.5}
                fill="var(--color-p-text)"
                style={{ opacity: out && !merged ? 1 : 0, transition: "opacity 400ms ease" }}
              >
                {short(row.term, 13)}
                <tspan fill={weighted ? "var(--color-p-accent)" : "var(--color-p-faint)"} className="num">
                  {" "}× {row.weight.toFixed(2)}
                </tspan>
              </text>
            </g>
          );
        })}
        {/* e = normalised sum */}
        <g style={{ opacity: phase >= 3 ? 1 : 0, transition: "opacity 500ms ease 250ms" }}>
          <text x={STACK_X - 8} y={eY + stackH - 2} textAnchor="end" fontSize={15} fontStyle="italic" fill="var(--color-p-accent)">
            e
          </text>
          {x.latent.map((v, j) => (
            <rect key={j} x={STACK_X + j * cellS} y={eY} width={cellS + 0.3} height={stackH + 4} fill={heat(v, eScale)} />
          ))}
          <rect x={STACK_X} y={eY} width={STACK_W} height={stackH + 4} fill="none" stroke="var(--color-p-accent)" strokeWidth={1.2} />
          <text x={STACK_X} y={eY + stackH + 22} fontSize={11.5} fill="var(--color-p-faint)">
            {k} numbers, length 1
          </text>
        </g>
        {own.length < x.term_rows.length && (
          <text x={STACK_X} y={18} fontSize={11.5} fill="var(--color-p-faint)">
            top {own.length} of {x.term_rows.length} terms shown
          </text>
        )}
        {ownIdx.size === 0 && (
          <text x={STACK_X} y={TOP + 20} fontSize={14} fill="var(--color-miss-fill)">
            no known terms: e is zero
          </text>
        )}
      </svg>
      </div>
      <div className="flex shrink-0 flex-wrap gap-6">
        <Gauge label="Question ρ" rho={x.term_rows.length ? x.rho : null} cut={cut} />
        <Gauge label="Cached question ρ (closest)" rho={best ? best.rho : null} cut={cut} />
      </div>
      {cut > 0 && (
        <p className="m-0 shrink-0 text-[13px] text-p-faint present:text-[2vh]">
          The white mark is the guard at {cut.toFixed(2)}: below it, a pair is scored by word overlap (TF-IDF), not by the SVD.
        </p>
      )}
    </div>
  );
}
