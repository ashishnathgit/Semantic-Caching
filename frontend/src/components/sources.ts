// Plain-language names for where a cached answer came from.
export function sourceLabel(source: string): string {
  switch (source) {
    case "faq":
      return "FAQ";
    case "ollama":
      return "Ollama";
    case "stub":
      return "Placeholder";
    case "manual":
      return "Added by hand";
    case "openai_compatible":
      return "Online model";
    default:
      return source;
  }
}

export const angleOf = (score: number) =>
  (Math.acos(Math.min(1, Math.max(0, score))) * 180) / Math.PI;

// "+0.123" / "−0.123", with a real minus sign.
export const signed = (v: number, digits = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export function formatMs(ms: number): string {
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 100) return `${ms.toFixed(1)} ms`;
  if (ms < 10000) return `${Math.round(ms).toLocaleString()} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function backendLabel(backend: string): string {
  return { ollama: "Ollama", openai_compatible: "online model", stub: "placeholder" }[backend] ?? backend;
}
