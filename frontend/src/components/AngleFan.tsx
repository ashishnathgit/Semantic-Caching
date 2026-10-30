import type { MatchItem } from "../types";

// Cosine similarity is the cosine of an angle. Every question is a unit vector,
// so we draw the query pointing straight up and each cached question at its angle
// theta = arccos(score) from it. A cached answer is reused when its vector falls
// inside the cone of half-angle arccos(tau).

const W = 400;
const H = 262;
const OX = W / 2;
const OY = 228;
const R = 178;

const toDeg = (rad: number) => (rad * 180) / Math.PI;
const angleOf = (score: number) => toDeg(Math.acos(Math.min(1, Math.max(0, score))));

function polar(deg: number, radius: number): [number, number] {
  const rad = (deg * Math.PI) / 180;
  return [OX + radius * Math.sin(rad), OY - radius * Math.cos(rad)];
}

// Labels for vectors closer than MIN_GAP degrees on the same side would collide,
// so later ones step inward along their own vector.
const MIN_GAP = 8;
function labelRadii(angles: number[]): number[] {
  const radii = angles.map(() => R + 13);
  for (const side of [1, -1]) {
    const idx = angles.map((_, i) => i).filter((i) => Math.sign(angles[i] || 1) === side);
    idx.sort((a, b) => Math.abs(angles[a]) - Math.abs(angles[b]));
    let step = 0;
    for (let n = 1; n < idx.length; n++) {
      const close = Math.abs(Math.abs(angles[idx[n]]) - Math.abs(angles[idx[n - 1]])) < MIN_GAP;
      step = close ? step + 1 : 0;
      radii[idx[n]] = step ? R - 14 * step - 4 : R + 13;
    }
  }
  return radii;
}

function wedge(halfAngle: number, radius: number): string {
  const [x1, y1] = polar(-halfAngle, radius);
  const [x2, y2] = polar(halfAngle, radius);
  const large = halfAngle * 2 > 180 ? 1 : 0;
  return `M ${OX} ${OY} L ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`;
}

interface Props {
  matches: MatchItem[];
  tau: number;
  exact: boolean;
  hoveredId: number | null;
  onHover: (id: number | null) => void;
  hasQuery: boolean;
}

export function AngleFan({ matches, tau, exact, hoveredId, onHover, hasQuery }: Props) {
  const cone = angleOf(tau);
  const thetas = matches.map((m, i) => angleOf(m.score) * (i % 2 === 0 ? 1 : -1));
  const radii = labelRadii(thetas);
  const ticks = Array.from({ length: 19 }, (_, i) => -90 + i * 10);
  const [coneLabelX, coneLabelY] = polar(cone, R + 16);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full"
        role="img"
        aria-labelledby="fan-title fan-desc"
      >
        <title id="fan-title">Angles between your question and the closest cached questions</title>
        <desc id="fan-desc">
          {matches.length
            ? matches
                .map((m, i) => `c${i + 1}: cosine ${m.score.toFixed(2)}, angle ${angleOf(m.score).toFixed(0)} degrees`)
                .join("; ")
            : "No cached questions to compare yet."}
        </desc>

        {/* the unit semicircle with a protractor scale */}
        <path
          d={`M ${OX - R} ${OY} A ${R} ${R} 0 0 1 ${OX + R} ${OY}`}
          fill="none"
          stroke="var(--color-rule)"
          strokeWidth={1}
        />
        <line x1={OX - R - 6} y1={OY} x2={OX + R + 6} y2={OY} stroke="var(--color-rule)" strokeWidth={1} />
        {ticks.map((t) => {
          const major = t % 30 === 0;
          const [x1, y1] = polar(t, R);
          const [x2, y2] = polar(t, R - (major ? 9 : 5));
          const [lx, ly] = polar(t, R - 20);
          return (
            <g key={t}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--color-ink-faint)" strokeWidth={1} />
              {major && Math.abs(t) < 90 && (
                <text x={lx} y={ly} textAnchor="middle" dominantBaseline="central" fontSize={10} fill="var(--color-ink-faint)">
                  {Math.abs(t)}°
                </text>
              )}
            </g>
          );
        })}

        {/* the hit cone: cos(theta) >= tau */}
        <path
          d={wedge(cone, R)}
          fill="var(--color-hit-wash)"
          fillOpacity={0.85}
          stroke="var(--color-hit)"
          strokeWidth={1}
        />
        <text
          x={coneLabelX + 4}
          y={coneLabelY}
          fontSize={11.5}
          fill="var(--color-hit)"
          dominantBaseline="central"
          className="num"
        >
          {cone.toFixed(1)}°
        </text>

        {/* cached question vectors, alternating left and right of the query */}
        {matches.map((m, i) => {
          const theta = thetas[i];
          const inside = exact && i === 0 ? true : m.score >= tau;
          const color = inside ? "var(--color-hit)" : "var(--color-ink-soft)";
          const active = hoveredId === m.id;
          const [tx, ty] = polar(theta, radii[i]);
          return (
            <g
              key={m.id}
              onMouseEnter={() => onHover(m.id)}
              onMouseLeave={() => onHover(null)}
              style={{ cursor: "default" }}
            >
              <g
                style={{
                  transform: `rotate(${theta}deg)`,
                  transformOrigin: `${OX}px ${OY}px`,
                  transition: "transform 450ms cubic-bezier(.2,.7,.2,1)",
                }}
              >
                <line x1={OX} y1={OY} x2={OX} y2={OY - R} stroke="transparent" strokeWidth={14} />
                <line
                  x1={OX}
                  y1={OY}
                  x2={OX}
                  y2={OY - R + 2}
                  stroke={color}
                  strokeWidth={active ? 3 : 2}
                  strokeLinecap="round"
                />
                <circle cx={OX} cy={OY - R} r={active ? 5.5 : 4.5} fill={color} stroke="var(--color-sheet)" strokeWidth={2} />
              </g>
              <text
                x={tx}
                y={ty}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={12.5}
                fontFamily="var(--font-math)"
                fontStyle="italic"
                fill="var(--color-ink)"
                fontWeight={active ? 600 : 400}
              >
                c<tspan fontSize={9} dy={3}>{i + 1}</tspan>
              </text>
              <title>{`${m.query}\ncos θ = ${m.score.toFixed(3)}, θ = ${angleOf(m.score).toFixed(1)}°`}</title>
            </g>
          );
        })}

        {/* the query vector */}
        <line
          x1={OX}
          y1={OY}
          x2={OX}
          y2={OY - R - 4}
          stroke={hasQuery ? "var(--color-query)" : "var(--color-rule)"}
          strokeWidth={3}
          strokeLinecap="round"
        />
        <path
          d={`M ${OX - 6} ${OY - R + 8} L ${OX} ${OY - R - 6} L ${OX + 6} ${OY - R + 8}`}
          fill="none"
          stroke={hasQuery ? "var(--color-query)" : "var(--color-rule)"}
          strokeWidth={3}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <text
          x={OX + 10}
          y={OY - R - 12}
          fontSize={15}
          fontFamily="var(--font-math)"
          fontStyle="italic"
          fill={hasQuery ? "var(--color-query)" : "var(--color-ink-faint)"}
        >
          q
        </text>
        <circle cx={OX} cy={OY} r={3.5} fill="var(--color-ink)" />
      </svg>
      <figcaption className="mt-1 text-[13px] leading-snug text-ink-soft">
        Your question <span className="font-math italic text-query">q</span> points up. Cached questions{" "}
        <span className="font-math italic">c</span> sit at their angle θ from it. Anything inside the green cone,
        where <span className="font-math">cos θ ≥ τ</span>, is answered from the cache.
      </figcaption>
    </figure>
  );
}
