// Diverging colour scale for the matrix heatmaps: blue for positive, amber for negative,
// the panel's dark blue for zero (values are scaled by ``max``).
const POS = [91, 140, 255];
const NEG = [245, 166, 35];
const ZERO = [26, 30, 74];

export function heat(v: number, max: number): string {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  const c = v >= 0 ? POS : NEG;
  const mix = (i: number) => Math.round(ZERO[i] + (c[i] - ZERO[i]) * Math.sqrt(a));
  return `rgb(${mix(0)},${mix(1)},${mix(2)})`;
}

export const maxAbs = (rows: number[][]) =>
  rows.reduce((m, r) => r.reduce((mm, v) => Math.max(mm, Math.abs(v)), m), 1e-9);

export function short(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

export const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");
