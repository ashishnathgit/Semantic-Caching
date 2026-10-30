import { useState } from "react";
import { prefersReducedMotion } from "../../hooks/useTween";
import type { Explanation, WordEffect } from "../../types";
import { signed, sourceLabel } from "../sources";
import { heat, maxAbs, short } from "./heat";

// Stage 4: e above the cache matrix C. A sweep runs down the rows and each row's score
// bar grows as it passes; the three best rows get labels.
// Normal layout, and a wider one for presentation mode, where the stage is short and wide.
const NARROW = { W: 680, HX: 16, HW: 300, BX: 344, BW: 230 };
const WIDE = { W: 1000, HX: 16, HW: 540, BX: 590, BW: 300 };
const TOP = 44;
const SWEEP_MS = 1300;

function WordEffects({ effects }: { effects: WordEffect[] }) {
  const max = Math.max(0.05, ...effects.map((e) => Math.abs(e.effect)));
  return (
    <ul className="m-0 mt-3 list-none space-y-1.5 p-0">
      {effects.slice(0, 8).map((e) => {
        const w = (Math.abs(e.effect) / max) * 50;
        return (
          <li key={e.term} className="grid grid-cols-[minmax(6rem,9rem)_1fr_4.5rem] items-center gap-x-3 text-[14px]"
            title={`Without "${e.term}" the score would be ${e.without.toFixed(3)}`}>
            <span className="truncate text-right text-p-text">
              {e.term}
              <span className="text-p-faint">{e.shared ? " · both" : e.partner && Math.abs(e.effect) >= 0.005 ? ` ↔ ${e.partner}` : ""}</span>
            </span>
            <span className="relative h-3">
              <span className="absolute inset-y-0 left-1/2 w-px bg-navy-line" />
              <span className="absolute inset-y-0 rounded-sm"
                style={{ left: e.effect >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`,
                  background: e.effect >= 0 ? "var(--color-hit-fill)" : "var(--color-miss-fill)" }} />
            </span>
            <span className={`num text-right ${e.effect >= 0 ? "text-hit-fill" : "text-miss-fill"}`}>{signed(e.effect)}</span>
          </li>
        );
      })}
    </ul>
  );
}

export function StageScore({ x, playKey, tau, newRowId, wide = false }: {
  x: Explanation; playKey: number; tau: number; newRowId: number | null; wide?: boolean;
}) {
  const { W, HX, HW, BX, BW } = wide ? WIDE : NARROW;
  const [open, setOpen] = useState(false);
  const C = x.cache_matrix;
  const n = C.rows.length;
  const rh = Math.min(10, 300 / Math.max(1, n));
  const k = C.k;
  const cw = HW / k;
  const scale = maxAbs([x.latent, ...C.rows.map((r) => r.vector)]);
  const height = TOP + n * rh + 16;
  const top3 = [...C.rows].filter((r) => r.rank !== null).sort((a, b) => (a.rank ?? 9) - (b.rank ?? 9));
  const hybrid = x.model.lsa_weight < 1;
  const best = x.matches[0];
  const still = prefersReducedMotion();

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="stage-fig">
      <svg key={`${playKey}:${x.query}`} viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="xMidYMin meet" role="img"
        aria-label={`Cache matrix C with ${C.n} rows; the best scores are ${top3.map((r) => r.score.toFixed(2)).join(", ")}`}>
        <text x={HX - 2} y={TOP - 22} fontSize={13} fontStyle="italic" fill="var(--color-p-accent)">e</text>
        <g transform={`translate(${HX + 12} ${TOP - 32})`}>
          {x.latent.map((v, j) => (
            <rect key={j} x={j * (HW - 12) / k} width={(HW - 12) / k + 0.3} height={12} fill={heat(v, scale)} />
          ))}
        </g>
        <text x={BX} y={TOP - 22} fontSize={12} fill="var(--color-p-faint)">
          score of each row{hybrid ? " (SVD + word overlap)" : ""}
        </text>
        {C.rows.map((r, i) => {
          const y = TOP + i * rh;
          const delay = (i / Math.max(1, n)) * SWEEP_MS;
          const isNew = r.id === newRowId;
          const hitRow = r.score >= tau;
          return (
            <g key={r.id}>
              {r.vector.map((v, j) => (
                <rect key={j} x={HX + j * cw} y={y} width={cw + 0.3} height={rh - 0.8} fill={heat(v, scale)} />
              ))}
              {isNew && <rect x={HX - 2} y={y - 1} width={HW + 4} height={rh + 1} fill="none" stroke="var(--color-cyan)" strokeWidth={1.5} />}
              <rect x={BX} y={y + 0.5} height={Math.max(1, rh - 2)} rx={1.5}
                width={still ? Math.max(0, Math.min(1, r.score)) * BW : 0}
                fill={r.rank === 1 ? (hitRow ? "var(--color-hit-fill)" : "var(--color-miss-fill)") : hitRow ? "var(--color-hit-fill)" : "var(--color-p-faint)"}
                fillOpacity={r.rank ? 1 : 0.55}>
                {!still && (
                  <animate attributeName="width" from="0" to={Math.max(0, Math.min(1, r.score)) * BW}
                    dur="450ms" begin={`${delay}ms`} fill="freeze" />
                )}
              </rect>
              {r.rank && (
                <g transform={`translate(${BX + BW + 14 + (r.rank - 1) * 22} ${y + rh / 2})`}>
                  <circle r={8} fill={r.rank === 1 ? "var(--color-p-accent)" : "var(--color-navy-line)"} />
                  <text y={4} textAnchor="middle" fontSize={11} fontWeight={700}
                    fill={r.rank === 1 ? "var(--color-navy)" : "var(--color-p-text)"}>{r.rank}</text>
                </g>
              )}
              {isNew && (
                <text x={BX + BW + 84} y={y + rh - 1} fontSize={11} fill="var(--color-cyan)">new row</text>
              )}
            </g>
          );
        })}
        <g style={{ transform: `translateX(${BX + tau * BW}px)`, transition: "transform 400ms ease" }}>
          <line x1={0} x2={0} y1={TOP - 6} y2={TOP + n * rh} stroke="var(--color-p-text)" strokeDasharray="3 3" strokeWidth={1.2} />
          <text x={0} y={TOP + n * rh + 13} textAnchor="middle" fontSize={11} fill="var(--color-p-text)">τ</text>
        </g>
        {/* the sweep */}
        {!still && <rect x={HX - 4} y={TOP - 2} width={BX + BW - HX + 8} height={2} fill="var(--color-cyan)"
          style={{ ["--sweep-to" as string]: `${n * rh}px`, animation: `sweep ${SWEEP_MS}ms linear both` } as React.CSSProperties} />}
      </svg>
      </div>
      <div className="shrink-0 space-y-2">
      <p className="num m-0 text-[13px] text-p-faint present:text-[2vh]">
        {C.sampled ? `showing ${C.shown} of ${C.n.toLocaleString()} rows (the top 20 and a sampled band)` : `all ${C.n} rows of C`} ·{" "}
        {k} columns
      </p>
      <ol className="m-0 list-none space-y-2 p-0">
        {top3.map((r) => (
          <li key={r.id} className="grid grid-cols-[1.25rem_3.5rem_1fr] items-baseline gap-x-2 text-[15px] leading-snug present:grid-cols-[2vh_8vh_1fr] present:text-[2.6vh]">
            <span className="num text-p-faint">{r.rank}</span>
            <span className={`num ${r.score >= tau ? "text-hit-fill" : "text-p-soft"}`}>{r.score.toFixed(3)}</span>
            <span className={`truncate ${r.hidden ? "text-p-faint italic" : "text-p-text"}`}>
              {short(r.query, 70)}
              <span className="text-[12px] text-p-faint">
                {" "}· {sourceLabel(r.source)}
                {r.scored_by === "tfidf" ? " · scored by word overlap (TF-IDF)" : ""}
                {r.id === newRowId ? " · the new row" : ""}
              </span>
            </span>
          </li>
        ))}
      </ol>
      {best && best.effects.length > 0 && !best.blocked && (
        <div>
          <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
            className="text-[13px] text-p-accent underline-offset-2 hover:underline present:text-[2.2vh]">
            {open ? "▾" : "▸"} What each word adds (top match)
          </button>
          {open && <WordEffects effects={best.effects} />}
        </div>
      )}
      </div>
    </div>
  );
}
