import { useState } from "react";
import type { ModelInfo } from "../types";

// Singular value spectrum: one series, so no legend; hover shows sigma_i.
const W = 360;
const H = 150;
const PAD = { left: 34, right: 10, top: 10, bottom: 24 };

function Spectrum({ values }: { values: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!values.length) return null;
  const max = values[0];
  const x = (i: number) => PAD.left + (i / Math.max(1, values.length - 1)) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - v / max) * (H - PAD.top - PAD.bottom);
  const path = values.map((v, i) => `${i ? "L" : "M"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  const ticks = [0, 0.5, 1].map((f) => f * max);

  const onMove = (event: React.PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - box.left) / box.width;
    setHover(Math.round(Math.min(1, Math.max(0, fraction)) * (values.length - 1)));
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Singular values in decreasing order">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--color-rule)" strokeWidth={1} />
          <text x={PAD.left - 6} y={y(t)} textAnchor="end" dominantBaseline="central" fontSize={10} fill="var(--color-ink-faint)" className="num">
            {t.toFixed(t >= 10 ? 0 : 1)}
          </text>
        </g>
      ))}
      <text x={PAD.left} y={H - 6} fontSize={10} fill="var(--color-ink-faint)">1</text>
      <text x={W - PAD.right} y={H - 6} fontSize={10} textAnchor="end" fill="var(--color-ink-faint)" className="num">
        {values.length}
      </text>
      <path d={path} fill="none" stroke="var(--color-query)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      {hover !== null && (
        <g pointerEvents="none">
          <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--color-ink-faint)" strokeWidth={1} />
          <circle cx={x(hover)} cy={y(values[hover])} r={4} fill="var(--color-query)" stroke="var(--color-sheet)" strokeWidth={2} />
          <text
            x={Math.min(x(hover) + 8, W - 90)}
            y={PAD.top + 12}
            fontSize={11}
            fill="var(--color-ink)"
            className="num"
          >
            σ{hover + 1} = {values[hover].toFixed(3)}
          </text>
        </g>
      )}
      <rect
        x={PAD.left}
        y={PAD.top}
        width={W - PAD.left - PAD.right}
        height={H - PAD.top - PAD.bottom}
        fill="transparent"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      />
    </svg>
  );
}

export function ModelPanel({ model }: { model: ModelInfo | null }) {
  if (!model) return <p className="text-[14px] text-ink-soft">Loading the model…</p>;
  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,30rem)_minmax(0,1fr)]">
      <div className="space-y-5">
        <p className="m-0 max-w-prose text-[15px] leading-relaxed text-ink-soft">
          The TF-IDF matrix of {model.training_documents.toLocaleString()} training questions and{" "}
          {model.vocabulary_size.toLocaleString()} terms is reduced to its top{" "}
          <span className="font-math italic">k</span> = {model.rank} singular directions, which keep{" "}
          {(model.energy_kept * 100).toFixed(1)}% of its energy.
        </p>
        <div>
          <h3 className="mb-1 font-math text-[17px] font-normal text-ink">
            Singular values <span className="italic">σ</span>
            <sub>i</sub>
          </h3>
          <Spectrum values={model.singular_values} />
        </div>
      </div>
      <div>
        <h3 className="mt-0 mb-2 font-math text-[17px] font-normal text-ink">Words grouped by the first directions</h3>
        <p className="mt-0 mb-3 max-w-prose text-[13px] leading-snug text-ink-soft">
          Each direction <span className="font-math italic">v</span>
          <sub>j</sub> is a column of <span className="font-math italic">V</span>
          <sub>k</sub>. These are the words with the largest weights in it.
        </p>
        <ol className="m-0 list-none space-y-1.5 p-0">
          {model.directions.map((d) => (
            <li key={d.direction} className="grid grid-cols-[5.5rem_1fr] gap-x-3 text-[14px] leading-snug">
              <span className="num text-ink-faint">
                <span className="font-math italic">v</span>
                <sub>{d.direction}</sub>, σ {d.sigma.toFixed(2)}
              </span>
              <span className="text-ink">{d.terms.join(", ")}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
