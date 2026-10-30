import { useEffect, useRef, useState } from "react";

export const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** A number that glides to ``target`` (ease-out) instead of snapping. */
export function useTween(target: number, ms = 600): number {
  const [value, setValue] = useState(target);
  const current = useRef(target);
  useEffect(() => {
    if (!Number.isFinite(target)) return;
    if (prefersReducedMotion() || ms <= 0) {
      current.current = target;
      setValue(target);
      return;
    }
    const from = current.current;
    const start = performance.now();
    let frame = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const v = from + (target - from) * (1 - Math.pow(1 - t, 3));
      current.current = v;
      setValue(v);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, ms]);
  return value;
}

/** Steps 0..n through a short sequence (one per ``ms``), restarting when ``key`` changes. */
export function usePhases(n: number, ms: number, key: unknown, active = true): number {
  const [phase, setPhase] = useState(prefersReducedMotion() ? n : 0);
  useEffect(() => {
    if (!active) return;
    if (prefersReducedMotion()) {
      setPhase(n);
      return;
    }
    setPhase(0);
    const timers = Array.from({ length: n }, (_, i) => setTimeout(() => setPhase(i + 1), ms * (i + 1)));
    return () => timers.forEach(clearTimeout);
  }, [n, ms, key, active]);
  return phase;
}
