import { useEffect, useMemo, useState } from "react";
import type uPlot from "uplot";
import { api } from "../../api";
import { useDebounced } from "../../hooks/useDebounced";
import type { LabCorrectness, LabHeadline, LabLatencyPoint, LabOps, LabRefit, LabWorked } from "../../types";
import { RouteTrace } from "../RouteTrace";
import { formatMs } from "../sources";
import { Tex, texText } from "../Tex";
import { AMBER, axis, BLURPLE, markerPlugin, UChart } from "./UChart";

// The Matrix Lab: checks and worked numbers behind the served model. Nothing here changes it.
// Every text is at least 18 px (projector-readable).

const fx = (v: number, d = 3) => v.toFixed(d);
const sci = (v: number) => (v === 0 ? "0" : Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(4));
const flops = (v: number) => {
  const units: [number, string][] = [[1e15, "P"], [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"]];
  for (const [n, u] of units) if (v >= n) return `${(v / n).toFixed(1)} ${u}FLOP`;
  return `${Math.round(v)} FLOP`;
};

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="card space-y-4 p-6">
      <div>
        <h2 className="m-0 text-[24px] font-semibold tracking-tight">{title}</h2>
        {sub && <p className="m-0 mt-1 max-w-prose text-[18px] leading-snug text-ink-soft">{sub}</p>}
      </div>
      {children}
    </section>
  );
}

function Check({ ok, label, children }: { ok: boolean; label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <span aria-label={ok ? "passes" : "fails"} className={`mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full text-[18px] font-bold ${ok ? "bg-hit-wash text-hit" : "bg-miss-wash text-miss"}`}>
        {ok ? "✓" : "✗"}
      </span>
      <div className="min-w-0">
        <p className="m-0 font-medium">{label}</p>
        <div className="text-ink-soft">{children}</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 6. correctness
function Correctness({ c }: { c: LabCorrectness }) {
  const ey = c.eckart_young;
  const j = c.jacobi;
  const offData = useMemo<uPlot.AlignedData>(
    () => [j.off_norms.map((_, i) => i), j.off_norms.map((v) => (v > 0 ? v : null))],
    [j.off_norms],
  );
  const offOpts = useMemo<Omit<uPlot.Options, "width" | "height">>(
    () => ({
      scales: { x: { time: false }, y: { distr: 3 } },
      series: [
        { label: "sweep", value: (_u, v) => (v == null ? "—" : String(v)) },
        { label: "‖off(A)‖_F", stroke: BLURPLE, width: 2, points: { show: true, size: 9 }, value: (_u, v) => (v == null ? "—" : v.toExponential(2)) },
      ],
      axes: [axis("sweep", { incrs: [1, 2, 5, 10] }), axis("off-diagonal norm (log scale)", { size: 96, values: (_u, vals) => vals.map((v) => (v == null ? "" : v.toExponential(0))) })],
      legend: { show: true },
    }),
    [],
  );
  return (
    <Section title="Correctness" sub={`The training matrix X is ${c.shape[0].toLocaleString()} × ${c.shape[1].toLocaleString()} (${c.nnz.toLocaleString()} non-zeros), rebuilt from the corpus with the served vocabulary and IDF.`}>
      <div className="grid gap-6 text-[18px] lg:grid-cols-2">
        <div className="space-y-5">
          <Check ok={ey.equal} label="Eckart–Young: what the truncation throws away">
            <Tex block tex={`\\lVert X - X_k \\rVert_F^2 = ${ey.residual_sq.toFixed(9)}`} />
            <Tex block tex={`\\textstyle\\sum_{i > ${c.k}} \\sigma_i^2 = ${ey.dropped_sigma_sq.toFixed(9)}`} />
            <p className="m-0">difference <span className="num">{ey.difference.toExponential(2)}</span></p>
            <p className="m-0 text-ink-faint">Left: {ey.residual_method}. Right: {ey.dropped_method}.</p>
          </Check>
          <Check ok={c.orthogonality.ok} label="V_k has orthonormal columns">
            <Tex block tex={`\\lVert V_k^{\\top} V_k - I \\rVert_F = ${sci(c.orthogonality.value)}`} />
          </Check>
          <Check ok={j.converged} label="Jacobi eigen-solver">
            <p className="m-0">
              {j.path === "exact" ? "Gram matrix" : "B Bᵀ of the randomized SVD"}, {j.size} × {j.size}: <span className="num">{j.sweeps}</span> sweeps,{" "}
              <span className="num">{j.rotations.toLocaleString()}</span> rotations, <span className="num">{formatMs(j.seconds * 1000)}</span>
            </p>
            <p className="m-0 text-ink-faint">Stops when ‖off(A)‖_F ≤ {j.tol.toExponential(0)} · ‖A‖_F = {(j.tol * j.frobenius).toExponential(2)}.</p>
          </Check>
          <p className="m-0 text-[18px] text-ink-soft">
            Energy kept at <span className="font-math italic">k</span> = {c.k}: <span className="num font-semibold text-ink">{(c.spectrum.energy_kept * 100).toFixed(2)}%</span>
          </p>
        </div>
        <div>
          <h3 className="m-0 mb-2 text-[20px] font-medium">Off-diagonal norm per Jacobi sweep</h3>
          <UChart data={offData} options={offOpts} height={320} label="Off-diagonal norm per sweep, log scale" />
        </div>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ 7. spectrum
function Spectrum({ c }: { c: LabCorrectness }) {
  const s = c.spectrum;
  const k = c.k;
  const xs = useMemo(() => s.sigma.map((_, i) => i + 1), [s.sigma]);
  const sigData = useMemo<uPlot.AlignedData>(() => [xs, s.sigma], [xs, s.sigma]);
  const enData = useMemo<uPlot.AlignedData>(() => [xs, s.energy_cumulative.map((v) => v * 100)], [xs, s.energy_cumulative]);
  const sigOpts = useMemo<Omit<uPlot.Options, "width" | "height">>(
    () => ({
      scales: { x: { time: false } },
      series: [{ label: "i" }, { label: "σ_i", stroke: BLURPLE, width: 2, points: { show: false }, value: (_u, v) => (v == null ? "—" : v.toFixed(4)) }],
      axes: [axis("index i"), axis("σ_i")],
      plugins: [markerPlugin(k, `k = ${k}`)],
    }),
    [k],
  );
  const enOpts = useMemo<Omit<uPlot.Options, "width" | "height">>(
    () => ({
      scales: { x: { time: false }, y: { range: [0, 100] } },
      series: [{ label: "k" }, { label: "energy kept", stroke: BLURPLE, width: 2, points: { show: false }, value: (_u, v) => (v == null ? "—" : `${v.toFixed(1)}%`) }],
      axes: [axis("k (directions kept)"), axis("% of ‖X‖_F²", { values: (_u, vals) => vals.map((v) => `${v}%`) })],
      plugins: [markerPlugin(k, `${(s.energy_kept * 100).toFixed(1)}% at k = ${k}`, "bottom")],
    }),
    [k, s.energy_kept],
  );
  return (
    <Section title="Singular values and energy" sub={s.full ? `All ${s.sigma.length} singular values of X; the dashed line is the served k.` : `The ${s.sigma.length} kept singular values (the full spectrum of this matrix is too large to compute).`}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="m-0 mb-2 text-[20px] font-medium">Singular values σ_i</h3>
          <UChart data={sigData} options={sigOpts} label="Singular values in decreasing order with a marker at k" />
        </div>
        <div>
          <h3 className="m-0 mb-2 text-[20px] font-medium">Cumulative energy (σ_1² + … + σ_k²) / ‖X‖_F²</h3>
          <UChart data={enData} options={enOpts} label="Cumulative energy kept by the top k directions" />
        </div>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ 8. worked maths
function Worked({ w }: { w: LabWorked }) {
  const r = w.rarest;
  const m = w.match;
  const idfTex = r
    ? w.smooth_idf
      ? `\\mathrm{idf}(\\text{${texText(r.term)}}) = \\ln\\frac{1+N}{1+\\mathrm{df}} + 1 = \\ln\\frac{1+${w.N}}{1+${r.df}} + 1 = ${r.idf.toFixed(3)}`
      : `\\mathrm{idf}(\\text{${texText(r.term)}}) = \\ln\\frac{N}{\\mathrm{df}} = \\ln\\frac{${w.N}}{${r.df}} = ${r.idf.toFixed(3)}`
    : null;
  const shown = w.terms.slice(0, 6);
  const normTex = `\\lVert w \\rVert = \\sqrt{${shown.map((t) => `${t.raw.toFixed(2)}^2`).join(" + ")}${w.terms.length > shown.length ? " + \\cdots" : ""}} = ${w.norm.toFixed(3)}`;
  const rowTex = `x = \\frac{w}{\\lVert w \\rVert} = \\big(${shown.map((t) => `\\underset{\\text{${texText(t.term)}}}{${t.weight.toFixed(3)}}`).join(",\\ ")}${w.terms.length > shown.length ? ",\\ \\dots" : ""}\\big)`;
  const eTex = `e = \\frac{x V_k}{\\lVert x V_k \\rVert},\\quad \\lVert x V_k \\rVert = ${w.latent_norm.toFixed(3)},\\quad e = (${w.e.slice(0, 5).map((v) => v.toFixed(3)).join(",\\ ")},\\ \\dots)\\ \\in \\mathbb{R}^{${w.k}}`;
  const sumTex = m
    ? `\\cos(e, f) = \\sum_{j=1}^{${m.terms_in_sum}} e_j f_j = ${m.top
        .map((t) => `(${t.e.toFixed(3)})(${t.f.toFixed(3)})`)
        .join(" + ")} + \\underbrace{${m.rest.toFixed(3)}}_{\\text{${m.terms_in_sum - m.top.length} more}} = ${m.lsa.toFixed(3)}`
    : null;
  const blendTex =
    m && m.scored_by === "hybrid" && m.tfidf !== null
      ? `s = w\\cos_{\\text{LSA}} + (1-w)\\cos_{\\text{TF-IDF}} = ${m.lsa_weight}\\cdot ${m.lsa.toFixed(3)} + ${(1 - m.lsa_weight).toFixed(2)}\\cdot ${m.tfidf.toFixed(3)} = ${m.score.toFixed(3)}`
      : m && m.scored_by === "tfidf" && m.tfidf !== null
        ? `s = \\cos_{\\text{TF-IDF}} = ${m.tfidf.toFixed(3)}\\ \\text{(the ρ guard blocked the SVD score)}`
        : null;
  const angTex = m
    ? `\\theta = \\arccos(${m.score.toFixed(3)}) = ${m.theta_deg.toFixed(1)}^\\circ \\ ${m.score >= w.tau ? "\\le" : ">"}\\ \\arccos(${w.tau.toFixed(2)}) = ${w.theta_tau_deg.toFixed(1)}^\\circ`
    : null;
  return (
    <div className="space-y-4 text-[18px]">
      {r && idfTex && (
        <div>
          <p className="m-0 text-ink-soft">1. The rarest word, “{r.term}”, is in {r.df} of N = {w.N} training texts:</p>
          <Tex block tex={idfTex} />
        </div>
      )}
      <div>
        <p className="m-0 text-ink-soft">
          2. The TF-IDF row: w<sub>t</sub> = {w.sublinear_tf ? "(1 + ln tf)" : "tf"} · idf, then scaled to unit length.
        </p>
        <Tex block tex={normTex} />
        <Tex block tex={rowTex} />
      </div>
      <div>
        <p className="m-0 text-ink-soft">3. Fold it into the {w.k}-dimensional space; ρ = ‖x V_k‖ / ‖x‖ = {w.rho.toFixed(3)}.</p>
        <Tex block tex={eTex} />
      </div>
      {m && sumTex && (
        <div>
          <p className="m-0 text-ink-soft">
            4. The best cached question{m.hidden ? " (another user's)" : `, “${m.query}”`}, has latent row f:
            {m.exact && " (an exact repeat: it hits before any similarity is needed)"}
          </p>
          <Tex block tex={sumTex} />
          {blendTex && <Tex block tex={blendTex} />}
        </div>
      )}
      {angTex && (
        <div>
          <p className="m-0 text-ink-soft">5. The decision as an angle:</p>
          <Tex block tex={angTex} />
        </div>
      )}
      {w.refused && (
        <p className="m-0 rounded-lg bg-miss-wash px-3 py-2 text-miss">
          6. Negation guard refused this match: {w.refused.rule}. {w.refused.detail}
        </p>
      )}
      {!m && <p className="m-0 text-ink-soft">No cached question to compare with.</p>}
    </div>
  );
}

// ------------------------------------------------------------------ 10. latency
function Latency({ points }: { points: LabLatencyPoint[] }) {
  const usable = useMemo(() => points.filter((p) => p.total_ms !== null && p.total_ms > 0), [points]);
  const data = useMemo<uPlot.AlignedData>(() => {
    const xs = usable.map((_, i) => i + 1);
    return [
      xs,
      usable.map((p) => (!p.llm ? (p.total_ms as number) : null)),
      usable.map((p) => (p.llm ? (p.total_ms as number) : null)),
    ];
  }, [usable]);
  const opts = useMemo<Omit<uPlot.Options, "width" | "height">>(
    () => ({
      scales: { x: { time: false }, y: { distr: 3 } },
      series: [
        { label: "request" },
        { label: "cache (no LLM call)", stroke: BLURPLE, fill: BLURPLE, paths: () => null, points: { show: true, size: 10, fill: BLURPLE }, value: (_u, v) => (v == null ? "—" : formatMs(v)) },
        { label: "LLM call", stroke: AMBER, fill: AMBER, paths: () => null, points: { show: true, size: 10, fill: AMBER }, value: (_u, v) => (v == null ? "—" : formatMs(v)) },
      ],
      axes: [axis("request (oldest → newest)", { incrs: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] }),
        axis("total time, ms (log scale)", { size: 90 })],
    }),
    [],
  );
  const median = (xs: number[]) => {
    if (!xs.length) return null;
    const s = [...xs].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  };
  const hits = usable.filter((p) => !p.llm).map((p) => p.total_ms as number);
  const llm = usable.filter((p) => p.llm).map((p) => p.total_ms as number);
  if (!usable.length) return <p className="m-0 text-[18px] text-ink-soft">No requests yet: ask a few questions on the Cache tab.</p>;
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <UChart data={data} options={opts} height={340} label="Latency per request on a log scale, cache answers vs LLM calls" />
      <table className="h-fit w-full border-collapse text-left text-[18px]">
        <thead className="text-ink-faint">
          <tr><th className="py-1 font-normal">answered by</th><th className="num py-1 text-right font-normal">n</th><th className="num py-1 text-right font-normal">median</th></tr>
        </thead>
        <tbody>
          <tr className="border-t border-rule"><td className="py-1"><span className="mr-2 inline-block h-3 w-3 rounded-full" style={{ background: BLURPLE }} />cache</td><td className="num text-right">{hits.length}</td><td className="num text-right">{median(hits) === null ? "—" : formatMs(median(hits)!)}</td></tr>
          <tr className="border-t border-rule"><td className="py-1"><span className="mr-2 inline-block h-3 w-3 rounded-full" style={{ background: AMBER }} />LLM call</td><td className="num text-right">{llm.length}</td><td className="num text-right">{median(llm) === null ? "—" : formatMs(median(llm)!)}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ 11. sandbox refit
function HeadlineRow({ name, h, k }: { name: string; h: LabHeadline; k: number }) {
  return (
    <tr className="border-t border-rule">
      <td className="py-1.5 pr-3">{name}</td>
      <td className="num pr-3 text-right">{k}</td>
      <td className="num pr-3 text-right">{fx(h.auc)}</td>
      <td className="num pr-3 text-right">{fx(h.precision)}</td>
      <td className="num pr-3 text-right">{fx(h.recall)}</td>
      <td className="num pr-3 text-right">{h.accepted_hits_correct} of {h.accepted_hits}</td>
      <td className="num text-right">{h.random_false_hits ?? "n/a"}</td>
    </tr>
  );
}

function Refit({ c, question }: { c: LabCorrectness; question: string }) {
  const maxK = Math.min(c.shape[0], c.shape[1]) - 1;
  const [k, setK] = useState(c.k);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<LabRefit | null>(null);
  const [err, setErr] = useState<string | null>(null);
  async function go() {
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.labRefit(k, question.trim() || null));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="space-y-4 text-[18px]">
      <div className="flex flex-wrap items-center gap-4">
        <label htmlFor="lab-k" className="font-medium">
          <span className="font-math italic">k</span> = <span className="num">{k}</span>
        </label>
        <input id="lab-k" type="range" min={1} max={maxK} value={k} onChange={(e) => setK(Number(e.target.value))} className="threshold w-72" />
        <button type="button" onClick={go} disabled={busy} className="h-11 rounded-lg bg-ink px-5 font-medium text-sheet hover:bg-ink-soft disabled:opacity-50">
          {busy ? "Refitting…" : "Refit SVD (sandbox)"}
        </button>
        <span className="text-ink-faint">served model: k = {c.k}, never changed</span>
      </div>
      {err && <p role="alert" className="m-0 text-miss">{err}</p>}
      {res && (
        <div className="space-y-3">
          <p className="m-0 text-ink-soft">
            Sandbox fit in {formatMs(res.seconds * 1000)}: {res.vocabulary_size} terms, energy kept {(res.energy_kept * 100).toFixed(1)}%,{" "}
            Eckart–Young {res.eckart_young.equal ? "✓" : "✗"} (‖X − X_k‖² = {res.eckart_young.residual_sq.toFixed(6)}, Σ dropped σ² = {res.eckart_young.dropped_sigma_sq.toFixed(6)}).
          </p>
          <table className="w-full max-w-4xl border-collapse text-left">
            <thead className="text-ink-faint">
              <tr>
                <th className="py-1 pr-3 font-normal">model</th><th className="num pr-3 text-right font-normal">k</th><th className="num pr-3 text-right font-normal">AUC</th>
                <th className="num pr-3 text-right font-normal">P @ τ {fx(res.headline.tau, 2)}</th><th className="num pr-3 text-right font-normal">R @ τ</th>
                <th className="num pr-3 text-right font-normal">accepted hits correct</th><th className="num text-right font-normal">random false hits / 1,000</th>
              </tr>
            </thead>
            <tbody>
              <HeadlineRow name="served" h={res.served_headline} k={res.served.k} />
              <HeadlineRow name="sandbox" h={res.headline} k={res.k} />
            </tbody>
          </table>
          {res.question && (
            <p className="m-0 text-ink-soft">
              “{res.question.query}”: best score {fx(res.question.served_score)} served ({res.question.served_hit ? "hit" : "miss"}) vs{" "}
              {fx(res.question.sandbox_score)} in the sandbox ({res.question.sandbox_hit ? "hit" : "miss"}).
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ 12. operation count
function Ops({ o }: { o: LabOps }) {
  return (
    <div className="grid gap-6 text-[18px] lg:grid-cols-2">
      <div>
        <h3 className="m-0 mb-2 text-[20px] font-medium">A cache hit: <span className="num">{flops(o.hit.flops)}</span></h3>
        <table className="w-full border-collapse text-left">
          <tbody>
            {o.hit.parts.map((p) => (
              <tr key={p.step} className="border-t border-rule">
                <td className="py-1 pr-3">{p.step}</td>
                <td className="num pr-3 text-ink-faint">{p.formula}</td>
                <td className="num text-right">{p.flops.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="m-0 mt-2 text-ink-faint">nnz = {o.nnz} terms, k = {o.k}, n = {o.n} rows of C.</p>
      </div>
      <div>
        {o.llm ? (
          <>
            <h3 className="m-0 mb-2 text-[20px] font-medium">An LLM answer: <span className="num">{flops(o.llm.flops)}</span> (estimate)</h3>
            <Tex block tex={`2 \\cdot P \\cdot (\\text{prompt} + \\text{output}) = 2 \\cdot ${o.llm.params_billion}\\times10^9 \\cdot (${o.llm.prompt_tokens} + ${o.llm.output_tokens})`} />
            <p className="m-0 text-ink-soft">Model {o.llm.model} ({o.llm.model_basis}); tokens: {o.llm.token_basis}.</p>
            <p className="m-0 text-ink-faint">{o.llm.note}.</p>
            {o.ratio && (
              <p className="m-0 mt-3 text-[22px] font-semibold">
                ≈ {o.ratio >= 1e6 ? `${(o.ratio / 1e6).toFixed(1)} million` : Math.round(o.ratio).toLocaleString()} × more work than a hit (estimate)
              </p>
            )}
          </>
        ) : (
          <p className="m-0 text-ink-soft">{o.missing}</p>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ the tab
export function MatrixLab({ tau, initialQuestion }: { tau: number; initialQuestion: string }) {
  const [c, setC] = useState<LabCorrectness | null>(null);
  const [cErr, setCErr] = useState<string | null>(null);
  const [question, setQuestion] = useState(initialQuestion || "cheap flight tickets to goa");
  const [worked, setWorked] = useState<LabWorked | null>(null);
  const [ops, setOps] = useState<LabOps | null>(null);
  const [qErr, setQErr] = useState<string | null>(null);
  const [latency, setLatency] = useState<LabLatencyPoint[]>([]);
  const q = useDebounced(question.trim(), 300);

  useEffect(() => {
    api.labCorrectness().then(setC).catch((e) => setCErr((e as Error).message));
    const load = () => api.labLatency().then((r) => setLatency(r.points)).catch(() => undefined);
    load();
    const timer = setInterval(() => document.visibilityState === "visible" && load(), 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!q) return;
    const ctl = new AbortController();
    Promise.all([api.labWorked(q, tau, ctl.signal), api.labOps(q, ctl.signal)])
      .then(([w, o]) => {
        setWorked(w);
        setOps(o);
        setQErr(null);
      })
      .catch((e) => (e as Error).name !== "AbortError" && setQErr((e as Error).message));
    return () => ctl.abort();
  }, [q, tau]);

  return (
    <div className="lab space-y-8 text-[18px]">
      <div className="card flex flex-wrap items-center gap-4 p-5">
        <label htmlFor="lab-q" className="font-medium">Question</label>
        <input id="lab-q" value={question} onChange={(e) => setQuestion(e.target.value)}
          className="h-12 min-w-[20rem] flex-1 rounded-lg border border-rule bg-sheet px-4 text-[18px] outline-none focus-visible:border-query" />
        <span className="text-ink-faint">τ = {tau.toFixed(2)} · nothing is stored or counted here</span>
      </div>
      {qErr && <p role="alert" className="m-0 text-miss">{qErr}</p>}
      {worked && (
        <Section title="Route" sub="What the cache does with this question, step by step (hover a step for details).">
          <RouteTrace route={worked.route} large />
        </Section>
      )}
      {worked && (
        <Section title="Worked maths" sub="The real numbers for this question, as the served model computes them.">
          <Worked w={worked} />
        </Section>
      )}
      {cErr && <p role="alert" className="m-0 text-miss">{cErr}</p>}
      {!c && !cErr && <p className="m-0 text-ink-soft">Checking the model…</p>}
      {c && <Correctness c={c} />}
      {c && <Spectrum c={c} />}
      <Section title="Latency per request" sub="Every answered question of this profile: answers from the cache vs answers that needed an LLM call.">
        <Latency points={latency} />
      </Section>
      {c && (
        <Section title="Try another k (sandbox)" sub="Refits TF-IDF + SVD on a copy with the chosen k and scores it on the FAQ held-out pairs. The served model and the cache never change.">
          <Refit c={c} question={question} />
        </Section>
      )}
      {ops && (
        <Section title="Operation count" sub="Floating-point operations to answer this question from the cache vs with the LLM. The LLM figure is an estimate.">
          <Ops o={ops} />
        </Section>
      )}
    </div>
  );
}
