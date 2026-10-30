import { usePhases } from "../../hooks/useTween";
import type { Explanation } from "../../types";
import { short } from "./heat";

// Stage 2: the TF-IDF row q. One cell per vocabulary term (compressed when the vocabulary
// is large); the question's non-zero cells light up, with leader lines to their labels.
const W = 680;
const X0 = 16;
const X1 = W - 16;
const STRIP_Y = 34;
const STRIP_H = 22;
const LABEL_Y = 150;
const MAX_LABELS = 8;

export function StageTfidf({ x, playKey }: { x: Explanation; playKey: number }) {
  const phase = usePhases(2, 500, `${playKey}:${x.query}`);
  const V = x.vocab_size;
  const width = X1 - X0;
  const cell = width / V;
  const showGrid = V <= 400;
  const weightOf = new Map(x.terms.map((t) => [t.term, t]));
  const rows = [...x.term_rows].sort((a, b) => b.weight - a.weight);
  const labelled = rows.slice(0, MAX_LABELS).sort((a, b) => a.index - b.index);
  const slot = width / Math.max(1, labelled.length);
  const maxW = Math.max(...x.term_rows.map((r) => r.weight), 1e-9);
  const nnz = x.term_rows.length;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="stage-fig">
      <svg viewBox={`0 0 ${W} 250`} preserveAspectRatio="xMidYMin meet" role="img" aria-label={`TF-IDF row with ${nnz} non-zero cells out of ${V}`}>
        <text x={X0} y={20} fontSize={12} fill="var(--color-p-faint)">
          q: one cell per vocabulary term ({V.toLocaleString()} cells{showGrid ? "" : ", compressed"})
        </text>
        <rect x={X0} y={STRIP_Y} width={width} height={STRIP_H} rx={3} fill="var(--color-navy-2)" />
        {showGrid &&
          Array.from({ length: V }, (_, i) => (
            <line
              key={i}
              x1={X0 + i * cell}
              x2={X0 + i * cell}
              y1={STRIP_Y}
              y2={STRIP_Y + STRIP_H}
              stroke="var(--color-navy-line)"
              strokeWidth={0.5}
            />
          ))}
        {x.term_rows.map((r, i) => {
          const cx = X0 + (r.index + 0.5) * cell;
          const on = phase >= 1;
          return (
            <rect
              key={r.term}
              x={cx - Math.max(1.6, cell / 2)}
              y={STRIP_Y - 3}
              width={Math.max(3.2, cell)}
              height={STRIP_H + 6}
              rx={1.5}
              fill="var(--color-p-accent)"
              style={{
                opacity: on ? 0.35 + 0.65 * (r.weight / maxW) : 0,
                transition: `opacity 400ms ease ${i * 70}ms`,
              }}
            />
          );
        })}
        {labelled.map((r, i) => {
          const cx = X0 + (r.index + 0.5) * cell;
          const lx = X0 + slot * (i + 0.5);
          const t = weightOf.get(r.term);
          const visible = phase >= 2;
          return (
            <g key={r.term} style={{ opacity: visible ? 1 : 0, transition: `opacity 400ms ease ${i * 60}ms` }}>
              <path
                d={`M ${cx} ${STRIP_Y + STRIP_H + 3} C ${cx} ${LABEL_Y - 50}, ${lx} ${LABEL_Y - 60}, ${lx} ${LABEL_Y - 18}`}
                fill="none"
                stroke="var(--color-p-accent)"
                strokeOpacity={0.55}
                strokeWidth={1.2}
              />
              <text x={lx} y={LABEL_Y} textAnchor="middle" fontSize={15} fontWeight={600} fill="var(--color-p-text)">
                {short(r.term, 14)}
              </text>
              <text x={lx} y={LABEL_Y + 20} textAnchor="middle" fontSize={11.5} fill="var(--color-p-faint)" className="num">
                tf {t?.tf ?? 1} · idf {t ? t.idf.toFixed(2) : "?"}
              </text>
              <text x={lx} y={LABEL_Y + 42} textAnchor="middle" fontSize={16} fill="var(--color-p-accent)" className="num">
                {r.weight.toFixed(3)}
              </text>
            </g>
          );
        })}
      </svg>
      </div>
      <p className="num m-0 shrink-0 text-[15px] text-p-soft present:text-[2.6vh]">
        {nnz} non-zero {nnz === 1 ? "cell" : "cells"} · {(V - nnz).toLocaleString()} zeros not stored
        {rows.length > MAX_LABELS ? ` · ${rows.length - MAX_LABELS} more terms not labelled` : ""}
      </p>
    </div>
  );
}
