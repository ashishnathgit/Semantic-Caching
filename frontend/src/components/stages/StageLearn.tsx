import { useTween } from "../../hooks/useTween";
import type { Explanation } from "../../types";
import { heat, maxAbs, short } from "./heat";

export interface Learned {
  id: number;
  query: string;
  vector: number[]; // the stored row, as /api/ask returned it
  before: number; // rows of C before
  after: number; // rows of C after
  replaced: boolean; // a forced refresh replaced an LLM answer: C keeps its size
}

// Stage 6: after a miss is answered, its vector becomes a new row at the bottom of C.
const W = 680;
const HX = 16;
const HW = 470;
const SHOW = 9;

export function StageLearn({ learned, x, playKey }: { learned: Learned | null; x: Explanation | null; playKey: number }) {
  const n = useTween(learned ? learned.after : 0, 1400);
  if (!learned) {
    return (
      <p className="m-0 max-w-prose text-[17px] leading-relaxed text-p-soft present:text-[2.8vh]">
        Ask a new question (a miss). Its answer is cached, and you will see its vector join the cache matrix C as a new row,
        so the next paraphrase is a hit.
      </p>
    );
  }
  const k = learned.vector.length;
  const others = (x?.cache_matrix.rows ?? []).filter((r) => r.id !== learned.id).slice(-SHOW);
  const scale = maxAbs([learned.vector, ...others.map((r) => r.vector)]);
  const rh = 16;
  const top = 30;
  const newY = top + others.length * (rh + 3) + 10;
  const cw = HW / k;
  const rows = Math.round(n);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <p className="num m-0 shrink-0 text-[22px] text-p-text present:text-[3.6vh]">
        C grows from {learned.before}×{k} to{" "}
        <span className="text-cyan">{learned.replaced ? learned.before : rows}×{k}</span>
        {learned.replaced && <span className="text-[15px] text-p-soft present:text-[2.4vh]"> (a refresh replaced the old answer: same size)</span>}
      </p>
      <div className="stage-fig">
      <svg key={`${playKey}:${learned.id}`} viewBox={`0 0 ${W} ${newY + rh + 30}`} preserveAspectRatio="xMidYMin meet" role="img"
        aria-label={`The new row for "${learned.query}" is added to the bottom of C`}>
        <text x={HX} y={18} fontSize={12} fill="var(--color-p-faint)">
          C: the last {others.length} of {learned.before} rows, then the new one
        </text>
        {others.map((r, i) => (
          <g key={r.id} transform={`translate(${HX} ${top + i * (rh + 3)})`}>
            {r.vector.map((v, j) => (
              <rect key={j} x={j * cw} width={cw + 0.3} height={rh} fill={heat(v, scale)} />
            ))}
            <text x={HW + 10} y={rh - 4} fontSize={11.5} fill="var(--color-p-faint)">{short(r.query, 22)}</text>
          </g>
        ))}
        <g transform={`translate(${HX} ${newY})`}>
          <g style={{ animation: "fly-in 900ms cubic-bezier(.2,.7,.2,1) both" }}>
            {learned.vector.map((v, j) => (
              <rect key={j} x={j * cw} width={cw + 0.3} height={rh + 4} fill={heat(v, scale)} />
            ))}
            <rect width={HW} height={rh + 4} fill="none" stroke="var(--color-cyan)" strokeWidth={2} />
            <text x={HW + 10} y={rh} fontSize={13} fill="var(--color-cyan)">new: {short(learned.query, 18)}</text>
          </g>
        </g>
      </svg>
      </div>
      <p className="m-0 shrink-0 text-[15px] text-p-soft present:text-[2.6vh]">
        Ask a paraphrase of “{short(learned.query, 60)}” next: stage 4 shows it landing on this row.
      </p>
    </div>
  );
}
