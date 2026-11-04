import { usePhases } from "../../hooks/useTween";
import type { Explanation } from "../../types";

// Stage 1: words appear, glue words fade and drop away, plurals morph into their folded
// form, unknown words get a dashed outline, and the kept word pairs (bigrams) glow.
export function StageClean({ x, playKey }: { x: Explanation; playKey: number }) {
  const phase = usePhases(2, 650, `${playKey}:${x.query}`);
  const dropped = phase >= 1;
  const known = x.tokens.filter((t) => t.status === "known").length;
  const content = x.tokens.filter((t) => t.status !== "stopword").length;

  return (
    <div className="space-y-6">
      <ul className="m-0 flex list-none flex-wrap gap-2.5 p-0" aria-label="Words of the question">
        {x.tokens.map((t, i) => {
          const stop = t.status === "stopword";
          const unknown = t.status === "unknown";
          const morphs = !stop && t.term !== null && t.term !== t.word;
          return (
            <li
              key={`${i}-${t.word}`}
              className={`chip relative rounded-lg px-3 py-1.5 text-[17px] transition-all duration-500 present:text-[3.4vh] ${
                stop
                  ? dropped
                    ? "translate-y-2 scale-95 text-p-faint line-through opacity-40"
                    : "bg-navy-2 text-p-text"
                  : unknown
                    ? dropped
                      ? "border border-dashed border-miss-fill bg-transparent text-miss-fill"
                      : "bg-navy-2 text-p-text"
                    : "bg-navy-2 text-p-text ring-1 ring-p-accent/50"
              }`}
              style={{ animationDelay: `${i * 55}ms` }}
              title={stop ? "A glue word: dropped" : unknown ? "Not in the vocabulary: ignored" : "Kept"}
            >
              {morphs ? (
                <span className="relative inline-grid">
                  <span className={`col-start-1 row-start-1 transition-opacity duration-500 ${dropped ? "opacity-0" : ""}`}>
                    {t.word}
                  </span>
                  <span
                    className={`col-start-1 row-start-1 text-p-accent transition-opacity duration-500 ${dropped ? "" : "opacity-0"}`}
                  >
                    {t.term}
                  </span>
                </span>
              ) : (
                t.word
              )}
            </li>
          );
        })}
      </ul>

      <div className={`space-y-2 transition-opacity duration-500 ${phase >= 2 ? "" : "opacity-0"}`}>
        <p className="m-0 text-[13px] tracking-wide text-p-faint uppercase present:text-[2vh]">Word pairs (bigrams)</p>
        {x.bigrams.length ? (
          <ul className="m-0 flex list-none flex-wrap gap-2.5 p-0">
            {x.bigrams.map((b) => (
              <li
                key={b.term}
                className={`rounded-lg px-3 py-1 text-[15px] present:text-[2.8vh] ${
                  b.known ? "glow bg-navy-2 text-p-text" : "text-p-faint line-through"
                }`}
                title={b.known ? "In the vocabulary: kept" : "Not in the vocabulary: dropped"}
              >
                {b.term}
              </li>
            ))}
          </ul>
        ) : (
          <p className="m-0 text-[14px] text-p-soft">None: fewer than two content words.</p>
        )}
      </div>

      <p className="num m-0 text-[15px] text-p-soft present:text-[2.6vh]">
        {x.tokens.length} words → {content} content words, {known} known to the model
        {x.coverage.total > 0 && (
          <span className={x.coverage.passed ? "text-hit-fill" : "text-miss-fill"}>
            {" "}
            ({Math.round(x.coverage.ratio * 100)}% known
            {x.coverage.relaxed
              ? `; everyday words not counted, matched on ${(x.coverage.topic_known ?? []).join(", ")}`
              : x.coverage.passed ? "" : `, below ${Math.round(x.coverage.min_ratio * 100)}%: blocked`})
          </span>
        )}
      </p>
    </div>
  );
}
