import type { Explanation } from "../types";
import { angleOf } from "./sources";
import { StageClean } from "./stages/StageClean";
import { StageDecide, VERDICT_STYLE, decide } from "./stages/StageDecide";
import { StageFold } from "./stages/StageFold";
import { type Learned, StageLearn } from "./stages/StageLearn";
import { StageScore } from "./stages/StageScore";
import { StageTfidf } from "./stages/StageTfidf";
import { Tex, texText } from "./Tex";

export const STAGE_MS = [2200, 2200, 3200, 2800, 3600, 3600];

interface StageInfo {
  short: string;
  title: string;
  caption: string;
  tex: (x: Explanation, tau: number, learned: Learned | null) => string;
  chip: (x: Explanation, tau: number, learned: Learned | null) => string;
}

const STAGES: StageInfo[] = [
  {
    short: "Clean",
    title: "Clean the text",
    caption: "Throw away the glue words and keep the ones that carry meaning.",
    tex: (x) => {
      const kept = x.terms.filter((t) => !t.bigram).map((t) => `\\text{${texText(t.term)}}`).slice(0, 5);
      const more = x.terms.filter((t) => !t.bigram).length > 5 ? ",\\ \\dots" : "";
      return `\\text{${texText(x.query.length > 34 ? x.query.slice(0, 33) + "…" : x.query)}} \\;\\longrightarrow\\; \\{${kept.join(",\\ ") || "\\;"}${more}\\}`;
    },
    chip: (x) => `${x.tokens.filter((t) => t.status === "known").length} kept`,
  },
  {
    short: "TF-IDF",
    title: "Weigh the words (TF-IDF)",
    caption: "Each kept word lights up one cell of a row as long as the vocabulary. Rare words weigh more.",
    tex: (x) =>
      `q \\in \\mathbb{R}^{1\\times ${x.vocab_size}},\\qquad q_t = \\dfrac{(1+\\ln \\mathrm{tf}_t)\\,\\mathrm{idf}_t}{\\lVert\\,\\cdot\\,\\rVert}`,
    chip: (x) => `${x.term_rows.length} of ${x.vocab_size.toLocaleString()}`,
  },
  {
    short: "Fold",
    title: "Fold into meaning space (SVD)",
    caption: "Each word has a row of meaning coordinates in V_k. Weighted and added up, the question becomes one short arrow e.",
    tex: (x) =>
      `e = \\dfrac{q\\,V_k}{\\lVert q\\,V_k\\rVert} \\in \\mathbb{R}^{${x.model.rank}},\\qquad \\rho = \\dfrac{\\lVert q\\,V_k\\rVert}{\\lVert q\\rVert} = ${x.rho.toFixed(2)}`,
    chip: (x) => `ρ ${x.rho.toFixed(2)}`,
  },
  {
    short: "Score",
    title: "Score the cache",
    caption: "One multiplication compares the question with every cached question at once. Longer bar, closer meaning.",
    tex: (x) => {
      const w = x.model.lsa_weight;
      const hybrid = w < 1 ? `,\\qquad \\text{score} = ${w.toFixed(1)}\\,s + ${(1 - w).toFixed(1)}\\,s_{\\text{words}}` : "";
      return `s = C\\,e,\\quad C \\in \\mathbb{R}^{${x.cache_matrix.n}\\times ${x.model.rank}}${hybrid}`;
    },
    chip: (x) => (x.matches[0] ? `best ${x.matches[0].score.toFixed(2)}` : "empty"),
  },
  {
    short: "Decide",
    title: "Decide",
    caption: "Close enough? Inside the cone the cached answer is reused, with no model call.",
    tex: (_x, tau) => `\\text{hit} \\iff \\max_i s_i \\ge \\tau \\iff \\theta \\le \\arccos\\tau = ${angleOf(tau).toFixed(1)}^\\circ`,
    chip: (x, tau) => decide(x, tau).kind,
  },
  {
    short: "Learn",
    title: "Learn",
    caption: "A new answer is remembered: its arrow becomes a new row of C, so next time a paraphrase hits.",
    tex: (x, _tau, learned) => {
      const n = learned ? learned.before : x.cache_matrix.n;
      return `C \\leftarrow \\begin{bmatrix} C \\\\ e^{\\top} \\end{bmatrix} \\in \\mathbb{R}^{(${n}+1)\\times ${x.model.rank}}`;
    },
    chip: (_x, _tau, learned) => (learned ? (learned.replaced ? "replaced" : `+1 row`) : "—"),
  },
];

interface Props {
  x: Explanation | null;
  loading: boolean;
  error: string | null;
  stage: number; // 1..6
  onStage: (n: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  playKey: number;
  tau: number;
  onTau: (t: number) => void;
  learned: Learned | null;
  presenting: boolean;
}

export function LiveMathPanel({ x, loading, error, stage, onStage, playing, onTogglePlay, playKey, tau, onTau, learned, presenting }: Props) {
  const info = STAGES[stage - 1];
  return (
    <section className="math-panel panel-dark flex flex-col gap-4 p-5 sm:p-6 xl:h-[calc(100vh-8.5rem)] xl:min-h-[640px] present:!h-full present:!min-h-0 present:gap-[1.4vh] present:!p-[2vh]" aria-label="Live linear algebra">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="m-0 text-[13px] font-semibold tracking-[0.14em] text-p-accent uppercase present:text-[1.6vh]">Live linear algebra</h2>
          <p className="m-0 mt-0.5 truncate text-[15px] text-p-soft present:text-[2.4vh]">
            {x ? <>for “{x.query}”</> : "Type a question to watch the math happen."}
            {loading && <span className="ml-2 inline-block h-2 w-2 animate-pulse rounded-full bg-cyan align-middle" aria-label="updating" />}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={onTogglePlay}
            className="rounded-lg border border-navy-line px-3 py-1.5 text-[13px] text-p-text hover:border-p-accent present:text-[1.9vh]"
            aria-label={playing ? "Pause the stages" : "Play the stages"}>
            {playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          {!presenting && <span className="hidden text-[12px] text-p-faint 2xl:inline">Space · ← → · 1–6 · R · P</span>}
        </div>
      </header>

      <div role="tablist" aria-label="Stages" className="grid grid-cols-3 gap-1.5 sm:grid-cols-6">
        {STAGES.map((s, i) => {
          const n = i + 1;
          const active = n === stage;
          const done = n < stage;
          const chip = x ? s.chip(x, tau, learned) : "";
          const verdict = n === 5 && x ? decide(x, tau).kind : null;
          return (
            <button key={s.title} type="button" role="tab" aria-selected={active} onClick={() => onStage(n)}
              disabled={!x && n !== 6}
              className={`group relative rounded-lg px-2.5 py-2 text-left transition-colors ${
                active ? "bg-navy-2 ring-1 ring-p-accent" : "hover:bg-navy-2/60"
              } disabled:opacity-40`}>
              <span className={`block text-[11px] font-semibold tracking-wide present:text-[1.8vh] ${active ? "text-p-accent" : done ? "text-p-soft" : "text-p-faint"}`}>
                {n} · {s.short}
              </span>
              <span className={`num mt-0.5 block truncate text-[13px] present:text-[2.2vh] ${verdict ? "" : "text-p-text"}`}>
                {verdict ? <span className={`rounded px-1.5 py-px text-[11px] font-bold present:text-[1.8vh] ${VERDICT_STYLE[verdict]}`}>{verdict}</span> : chip || " "}
              </span>
              <span className="absolute inset-x-2 bottom-0.5 h-0.5 overflow-hidden rounded bg-navy-line">
                <span className={`block h-full bg-p-accent ${active && playing ? "" : done ? "w-full" : "w-0"}`}
                  style={active && playing ? { animation: `grow ${STAGE_MS[i]}ms linear both` } : undefined} />
              </span>
            </button>
          );
        })}
      </div>

      {error && !x && <p className="m-0 text-[15px] text-miss-fill">{error}</p>}
      {!x && stage !== 6 ? (
        <div className="grid min-h-[320px] place-items-center rounded-xl border border-dashed border-navy-line p-8 text-center">
          <p className="m-0 max-w-md text-[17px] leading-relaxed text-p-soft">
            Start typing a question on the left. Each pause in typing runs the whole pipeline here, from words to the cache decision.
          </p>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-4" role="tabpanel" aria-label={info.title}>
          <div className="shrink-0 space-y-2">
            <h3 className="m-0 text-[22px] font-semibold tracking-tight text-p-text present:text-[3.4vh]">
              {stage}. {info.title}
            </h3>
            <p className="m-0 text-[16px] leading-snug text-p-soft present:text-[2.6vh]">{info.caption}</p>
            {x && (
              <div className="overflow-x-auto rounded-lg bg-navy-2/70 px-4 py-2.5 text-[17px] present:py-[0.8vh] present:text-[2.8vh]">
                <Tex tex={info.tex(x, tau, learned)} />
              </div>
            )}
          </div>
          <div className="stage-body flex min-h-0 flex-1 flex-col overflow-y-auto">
          {x && stage === 1 && <StageClean x={x} playKey={playKey} />}
          {x && stage === 2 && <StageTfidf x={x} playKey={playKey} />}
          {x && stage === 3 && <StageFold x={x} playKey={playKey} wide={presenting} />}
          {x && stage === 4 && <StageScore x={x} playKey={playKey} tau={tau} newRowId={learned?.id ?? null} wide={presenting} />}
          {x && stage === 5 && <StageDecide x={x} tau={tau} onTau={onTau} newRowId={learned?.id ?? null} compact={presenting} />}
          {stage === 6 && <StageLearn learned={learned} x={x} playKey={playKey} />}
          </div>
        </div>
      )}
    </section>
  );
}
