import { useEffect, useRef } from "react";
import uPlot from "uplot";

// A thin uPlot wrapper for the Matrix Lab: the chart is rebuilt when its data or options
// change (the lab's data sets are small) and follows the width of its box.
export const LAB_FONT = '18px "Inter Variable", ui-sans-serif, system-ui, sans-serif';
export const INK_SOFT = "#474c6b";
export const RULE = "#e4e5ee";
export const BLURPLE = "#635bff";
export const AMBER = "#d98a00";

export function axis(label: string, extra: Partial<uPlot.Axis> = {}): uPlot.Axis {
  return { label, stroke: INK_SOFT, font: LAB_FONT, labelFont: LAB_FONT, size: 72, labelSize: 30,
    grid: { stroke: RULE, width: 1 }, ticks: { stroke: RULE }, ...extra };
}

/** A vertical marker line with a label, e.g. at k. */
export function markerPlugin(x: number, label: string, where: "top" | "bottom" = "top"): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const ctx = u.ctx;
        const px = u.valToPos(x, "x", true);
        ctx.save();
        ctx.strokeStyle = "#111439";
        ctx.lineWidth = 2 * devicePixelRatio;
        ctx.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
        ctx.beginPath();
        ctx.moveTo(px, u.bbox.top);
        ctx.lineTo(px, u.bbox.top + u.bbox.height);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#111439";
        ctx.font = `${18 * devicePixelRatio}px "Inter Variable", sans-serif`;
        const right = px + 8 * devicePixelRatio + ctx.measureText(label).width < u.bbox.left + u.bbox.width;
        ctx.textAlign = right ? "left" : "right";
        const y = where === "top" ? u.bbox.top + 22 * devicePixelRatio : u.bbox.top + u.bbox.height - 12 * devicePixelRatio;
        ctx.fillText(label, px + (right ? 8 : -8) * devicePixelRatio, y);
        ctx.restore();
      },
    },
  };
}

export function UChart({ data, options, height = 300, label }: {
  data: uPlot.AlignedData; options: Omit<uPlot.Options, "width" | "height">; height?: number; label: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!box.current) return;
    const chart = new uPlot({ ...options, width: box.current.clientWidth || 600, height }, data, box.current);
    const resize = new ResizeObserver(() => {
      if (box.current) chart.setSize({ width: box.current.clientWidth, height });
    });
    resize.observe(box.current);
    return () => {
      resize.disconnect();
      chart.destroy();
    };
  }, [data, options, height]);
  return <div ref={box} role="img" aria-label={label} className="lab-chart w-full" />;
}
