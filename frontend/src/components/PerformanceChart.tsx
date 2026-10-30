import { useEffect, useRef } from "react";
import uPlot from "uplot";
import type { Metrics } from "../types";

// Cumulative hit rate (left axis, %) and total requests (right axis) per time bucket.
const BLURPLE = "#635bff";
const AMBER = "#d98a00";

export function PerformanceChart({ metrics }: { metrics: Metrics | null }) {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const points = metrics?.series.points ?? [];
  const bucket = metrics?.series.bucket_seconds ?? 10;

  useEffect(() => {
    if (!box.current) return;
    const opts: uPlot.Options = {
      width: box.current.clientWidth,
      height: 280,
      scales: {
        // Pad the time axis around the data; one bucket alone would otherwise span years.
        // (uPlot widens a single time point by 86400 / 1e-3 s, about 2.7 years, so use the data.)
        x: {
          time: true,
          auto: true,
          range: (u) => {
            const xs = u.data[0] as number[];
            if (!xs.length) return [Date.now() / 1000 - 60, Date.now() / 1000];
            const lo = xs[0];
            const hi = xs[xs.length - 1];
            return lo === hi ? [lo - 60, hi + 30] : [lo, hi + 5];
          },
        },
        pct: { range: [0, 100] }, n: { range: (_u, _min, max) => [0, Math.max(5, max * 1.1)] } },
      series: [
        {},
        { label: "Cumulative hit rate (%)", scale: "pct", stroke: BLURPLE, width: 2.5, points: { show: true, size: 6 }, value: (_u, v) => (v === null ? "—" : `${v.toFixed(1)}%`), spanGaps: true },
        { label: "Total requests", scale: "n", stroke: AMBER, width: 2, dash: [6, 4], points: { show: true, size: 6 }, value: (_u, v) => (v === null ? "—" : String(v)) },
      ],
      axes: [
        { label: "Time", stroke: "#474c6b", grid: { stroke: "#e4e5ee" }, ticks: { stroke: "#e4e5ee" } },
        { label: "Hit rate (%)", scale: "pct", stroke: "#474c6b", grid: { stroke: "#e4e5ee" }, values: (_u, v) => v.map((x) => `${x}%`) },
        { label: "Requests", scale: "n", side: 1, stroke: "#474c6b", grid: { show: false } },
      ],
      legend: { live: true },
      cursor: { points: { size: 7 } },
    };
    plot.current = new uPlot(opts, [[], [], []], box.current);
    const resize = new ResizeObserver(() => {
      if (box.current && plot.current) plot.current.setSize({ width: box.current.clientWidth, height: 280 });
    });
    resize.observe(box.current);
    return () => {
      resize.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
  }, []);

  useEffect(() => {
    if (!plot.current) return;
    plot.current.setData([
      points.map((p) => p.t),
      points.map((p) => (p.hit_rate === null ? null : p.hit_rate * 100)) as (number | null)[],
      points.map((p) => p.total_requests),
    ]);
  }, [points]);

  return (
    <div className="card space-y-2 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 text-[22px] font-semibold tracking-tight">Cache performance over time</h2>
        <span className="text-[13px] text-ink-faint">
          {bucket}-second buckets · hit rate of cache lookups{metrics?.series.truncated ? " · latest buckets only" : ""}
        </span>
      </div>
      {points.length === 0 && <p className="m-0 text-[14px] text-ink-soft">No requests yet: ask something and the chart starts.</p>}
      <div ref={box} className="w-full" />
    </div>
  );
}
