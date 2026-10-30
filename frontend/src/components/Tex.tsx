import katex from "katex";
import { useMemo } from "react";

/** Escape text for \text{...} inside a KaTeX formula. */
export function texText(s: string): string {
  return s.replace(/[\\{}$&#^_%~]/g, (c) => (c === "\\" ? "\\textbackslash{}" : `\\${c}`));
}

/** A formula typeset by KaTeX (bundled, works offline). */
export function Tex({ tex, block = false, className = "" }: { tex: string; block?: boolean; className?: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: block, throwOnError: false, output: "htmlAndMathml" }),
    [tex, block],
  );
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
