import { useCallback, useEffect, useRef, useState } from "react";
import { api, setApiAdminKey, setApiUser } from "./api";
import { AdminTools } from "./components/AdminTools";
import { AnswerCard } from "./components/AnswerCard";
import { CachePanel } from "./components/CachePanel";
import { Conversation } from "./components/Conversation";
import { MatrixLab } from "./components/lab/MatrixLab";
import { LiveMathPanel, STAGE_MS } from "./components/LiveMathPanel";
import { ModelPanel } from "./components/ModelPanel";
import { PerformanceChart } from "./components/PerformanceChart";
import { PresentStrip } from "./components/PresentStrip";
import { QuestionBox } from "./components/QuestionBox";
import { SignIn } from "./components/SignIn";
import type { Learned } from "./components/stages/StageLearn";
import { StatsCards } from "./components/StatsCards";
import { ThreadPicker } from "./components/ThreadPicker";
import { useDebounced } from "./hooks/useDebounced";
import type {
  AskResult,
  CacheEntry,
  Explanation,
  Health,
  Metrics,
  ModelInfo,
  Profile,
  ThreadDetail,
  ThreadSummary,
} from "./types";

// Per-browser conveniences: which name to sign in with and the open conversation stay in
// localStorage; the admin key lives in sessionStorage, so it is forgotten with the tab.
const USER_KEY = "semcache.user";
const ADMIN_KEY = "semcache.adminKey";
const threadKey = (user: string) => `semcache.thread.${user}`;
function readStored(key: string, session = false): string | null {
  try {
    return (session ? sessionStorage : localStorage).getItem(key);
  } catch {
    return null;
  }
}
function writeStored(key: string, value: string | null, session = false) {
  try {
    const store = session ? sessionStorage : localStorage;
    if (value === null) store.removeItem(key);
    else store.setItem(key, value);
  } catch {
    /* storage unavailable: nothing is remembered */
  }
}

const METRICS_POLL_MS = 5000;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [entries, setEntries] = useState<CacheEntry[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);

  const [user, setUser] = useState<Profile | null>(null);
  const [restoring, setRestoring] = useState(() => readStored(USER_KEY) !== null);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [thread, setThread] = useState<ThreadDetail | null>(null);

  const [query, setQuery] = useState("");
  const [tau, setTau] = useState(0.7);
  const [forceRefresh, setForceRefresh] = useState(false);
  const [result, setResult] = useState<AskResult | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [askError, setAskError] = useState<string | null>(null);

  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [cacheVersion, setCacheVersion] = useState(0);
  const [replay, setReplay] = useState<string | null>(null);

  const [stage, setStage] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [playKey, setPlayKey] = useState(0);
  const [learned, setLearned] = useState<Learned | null>(null);
  const lastQuery = useRef<string | null>(null);

  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [scope, setScope] = useState<"global" | "user">("global");
  const [present, setPresent] = useState(false);
  const [tab, setTab] = useState<"cache" | "lab">("cache");

  // ------------------------------------------------------------------ data
  const refresh = useCallback(async () => {
    try {
      const [h, list] = await Promise.all([api.health(), api.entries()]);
      setHealth(h);
      setEntries(list);
      setApiError(null);
      return h;
    } catch (err) {
      setApiError((err as Error).message);
      return null;
    }
  }, []);

  const loadMetrics = useCallback(async () => {
    try {
      setMetrics(await api.metrics(scope));
    } catch {
      /* the cards keep their last values */
    }
  }, [scope]);

  const loadThreads = useCallback(async () => {
    try {
      setThreads(await api.threads());
    } catch (err) {
      setApiError((err as Error).message);
    }
  }, []);

  const openThread = useCallback(async (id: string | null, name: string) => {
    setThreadId(id);
    writeStored(threadKey(name), id);
    if (id === null) {
      setThread(null);
      return;
    }
    try {
      setThread(await api.thread(id));
    } catch {
      setThreadId(null);
      setThread(null);
      writeStored(threadKey(name), null);
    }
  }, []);

  const signIn = useCallback(
    async (name: string, adminKey: string | null = null) => {
      setApiUser(name);
      setApiAdminKey(adminKey);
      try {
        const profile = await api.signIn(name);
        setUser(profile);
        writeStored(USER_KEY, profile.name);
        writeStored(ADMIN_KEY, adminKey, true);
        await Promise.all([refresh(), loadThreads(), openThread(readStored(threadKey(profile.name)), profile.name)]);
      } catch (err) {
        setApiUser(null);
        setApiAdminKey(null);
        writeStored(ADMIN_KEY, null, true);
        throw err;
      }
    },
    [refresh, loadThreads, openThread],
  );

  function signOut() {
    setApiUser(null);
    setApiAdminKey(null);
    writeStored(ADMIN_KEY, null, true);
    writeStored(USER_KEY, null);
    setUser(null);
    setThreads([]);
    setThread(null);
    setThreadId(null);
    setResult(null);
    setQuery("");
    setScope("global");
    refresh();
  }

  useEffect(() => {
    refresh().then((h) => {
      if (h) setTau(h.defaults.threshold);
    });
    api.model().then(setModel).catch(() => undefined);
    const remembered = readStored(USER_KEY);
    if (remembered) {
      signIn(remembered, readStored(ADMIN_KEY, true))
        .catch(() => writeStored(USER_KEY, null))
        .finally(() => setRestoring(false));
    }
    // Run once on load; signIn is stable for the session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    loadMetrics();
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") loadMetrics();
    }, METRICS_POLL_MS);
    return () => clearInterval(timer);
  }, [loadMetrics]);

  // ------------------------------------------- live explain while typing (250 ms pause)
  const debounced = useDebounced(query.trim(), 250);
  const explainTau = useDebounced(tau, 150);
  const target = query.trim() ? debounced : (replay ?? result?.query ?? "");
  useEffect(() => {
    if (!target || !user) {
      setExplanation(null);
      return;
    }
    const controller = new AbortController();
    setExplaining(true);
    api
      .explain(target, explainTau, threadId, controller.signal)
      .then((x) => {
        setExplanation(x);
        setExplainError(null);
        // A different question replays the stages from the start; the same question
        // (a new τ, or the cache after an answer) updates them in place.
        if (x.query !== lastQuery.current) {
          lastQuery.current = x.query;
          setStage(1);
          setPlaying(true);
          setPlayKey((k) => k + 1);
        }
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError") setExplainError((err as Error).message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setExplaining(false);
      });
    return () => controller.abort();
  }, [target, explainTau, threadId, user, cacheVersion]);

  // ------------------------------------------------------------- stage player
  const lastStage = learned && lastQuery.current === learned.query ? 6 : 5;
  useEffect(() => {
    if (!playing || !explanation || stage >= lastStage) return;
    const timer = setTimeout(() => setStage((s) => Math.min(lastStage, s + 1)), STAGE_MS[stage - 1]);
    return () => clearTimeout(timer);
  }, [playing, explanation, stage, lastStage, playKey]);

  const goTo = useCallback((n: number) => {
    setPlaying(false);
    setStage(Math.max(1, Math.min(6, n)));
    setPlayKey((k) => k + 1);
  }, []);

  const togglePlay = useCallback(() => {
    if (!playing && stage >= lastStage) {
      setStage(1);
      setPlayKey((k) => k + 1);
    }
    setPlaying(!playing);
  }, [playing, stage, lastStage]);

  const replayLast = useCallback(() => {
    const q = result?.query ?? lastQuery.current;
    if (!q) return;
    setQuery("");
    setReplay(q);
    setStage(1);
    setPlaying(true);
    setPlayKey((k) => k + 1);
  }, [result]);

  // ------------------------------------------------------------------ ask
  async function ask() {
    const q = query.trim();
    if (!q) {
      setAskError("Type a question first.");
      return;
    }
    if (!user) return;
    setPendingQuestion(q);
    setAskError(null);
    const before = explanation && explanation.query === q ? explanation.cache_matrix.n : null;
    try {
      const r = await api.ask(q, tau, threadId, forceRefresh);
      setResult(r);
      setReplay(null);
      setQuery("");
      lastQuery.current = r.query;
      if (r.stored_id !== null && r.stored_vector) {
        const replaced = (r.note ?? "").includes("Replaced");
        setLearned({
          id: r.stored_id,
          query: r.query,
          vector: r.stored_vector,
          after: r.cache_rows,
          replaced,
          before: before ?? (replaced ? r.cache_rows : r.cache_rows - 1),
        });
        setStage(6);
      } else {
        setStage(5);
      }
      setPlaying(true);
      setPlayKey((k) => k + 1);
      setCacheVersion((v) => v + 1);
      if (r.thread_id) {
        setThreadId(r.thread_id);
        writeStored(threadKey(user.name), r.thread_id);
        setThread(await api.thread(r.thread_id));
      }
      await Promise.all([refresh(), loadThreads(), api.me().then(setUser), loadMetrics()]);
    } catch (err) {
      setAskError((err as Error).message);
    } finally {
      setPendingQuestion(null);
    }
  }

  // ------------------------------------------------------- presentation mode
  const togglePresent = useCallback(() => setPresent((p) => !p), []);
  // While the LLM loads in the background, poll /health until it is ready.
  const warming = health?.backend.warmup?.state === "warming";
  useEffect(() => {
    if (!warming) return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [warming, refresh]);
  useEffect(() => {
    const root = document.documentElement;
    root.toggleAttribute("data-present", present);
    if (present && !document.fullscreenElement) root.requestFullscreen?.().catch(() => undefined);
    if (!present && document.fullscreenElement) document.exitFullscreen?.().catch(() => undefined);
  }, [present]);
  useEffect(() => {
    const onChange = () => {
      if (!document.fullscreenElement) setPresent(false);
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // Keys: P presentation, Space play/pause, ← → step, 1-6 jump, R replay. Not while typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable;
      if (typing) {
        if (e.key === "Escape") el.blur();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "p" || e.key === "P") togglePresent();
      else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowLeft") goTo(stage - 1);
      else if (e.key === "ArrowRight") goTo(stage + 1);
      else if (/^[1-6]$/.test(e.key)) goTo(Number(e.key));
      else if (e.key === "r" || e.key === "R") replayLast();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePresent, togglePlay, goTo, replayLast, stage]);

  // ------------------------------------------------------------------ view
  const backend = health?.backend;
  const followUpHint =
    query.trim() && explanation && explanation.query === query.trim() && explanation.follow_up && !explanation.cache_used
      ? `Looks like a follow-up (${explanation.follow_up_reason}): it skips the shared cache and goes to the model with this conversation.`
      : null;

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="hero-band hide-in-present" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
        <span />
      </div>

      <div className="app-shell relative z-10 mx-auto max-w-[1880px] px-4 pt-5 pb-16 sm:px-6 xl:px-8 present:max-w-none present:p-[1.2vh]">
        <header className="hide-in-present card mb-6 flex flex-wrap items-center justify-between gap-x-6 gap-y-3 bg-white/85 px-5 py-3 backdrop-blur">
          <div className="flex flex-wrap items-center gap-3">
            <span className="h-8 w-8 rounded-lg bg-[linear-gradient(135deg,var(--color-blurple),var(--color-pink),var(--color-amber))]" aria-hidden="true" />
            <div>
              <h1 className="m-0 text-[20px] leading-tight font-semibold tracking-tight">Semantic cache</h1>
              <p className="m-0 text-[13px] text-ink-soft">Same meaning, same answer, no second model call.</p>
            </div>
            {health && (
              <span
                className="rounded-full bg-ink px-3 py-1 text-[12px] font-semibold tracking-wide text-sheet uppercase"
                title={`Model profile (python -m semcache serve --profile ${health.profile})`}
              >
                {health.profile} · k = {health.model.rank} · {health.model.vocabulary_size.toLocaleString()} terms
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            {health && (
              <span className="flex items-center gap-2 rounded-full border border-rule bg-sheet px-3 py-1">
                <span className={`h-2 w-2 rounded-full ${warming ? "bg-amber" : backend?.available ? "bg-hit-fill" : "bg-miss-fill"}`} aria-hidden="true" />
                {backend?.backend === "stub"
                  ? "Stub backend"
                  : warming
                    ? <span className="animate-pulse">Model warming up…</span>
                    : backend?.available
                      ? `${backend.backend === "openai_compatible" ? "Online" : "Ollama"} · ${backend.model}${backend.chain ? ` (chain: ${backend.chain.join(" → ")})` : ""}`
                      : backend?.backend === "openai_compatible"
                        ? "Online model not configured"
                        : "Ollama off: placeholder answers"}
              </span>
            )}
            {user && (
              <span className="flex items-center gap-2 rounded-full border border-rule bg-sheet px-3 py-1">
                {user.name}
                {user.is_admin && <span className="rounded bg-blurple px-1.5 text-[11px] font-semibold text-white">admin</span>}
                <button type="button" onClick={signOut} className="text-query hover:underline">
                  switch
                </button>
              </span>
            )}
            <button
              type="button"
              onClick={togglePresent}
              className="rounded-full bg-ink px-4 py-1.5 font-medium text-sheet hover:bg-ink-soft"
              title="Presentation mode (P)"
            >
              Presentation mode
            </button>
          </div>
        </header>

        {apiError && (
          <div role="alert" className="card mb-6 border-l-4 border-miss-fill px-4 py-3 text-[15px]">
            {apiError}
          </div>
        )}

        {!user ? (
          restoring ? (
            <p className="text-[15px] text-ink-soft">Signing you back in…</p>
          ) : (
            <SignIn onSignIn={signIn} adminEnabled={health?.admin_enabled ?? false} />
          )
        ) : (
          present ? (
            <div className="grid h-[calc(100vh-2.4vh)] grid-cols-[minmax(250px,22vw)_minmax(0,1fr)] gap-[1.2vh]">
              <PresentStrip profile={health?.profile ?? null} query={query}
                onQuery={(q) => { setQuery(q); setAskError(null); setReplay(null); }}
                onAsk={ask} pending={pendingQuestion} result={result} error={askError} warming={warming} />
              <LiveMathPanel x={explanation} loading={explaining} error={explainError} stage={stage} onStage={goTo}
                playing={playing} onTogglePlay={togglePlay} playKey={playKey} tau={tau} onTau={setTau}
                learned={learned} presenting />
            </div>
          ) : (
          <>
            {/* A radio group, not role=tab: the stage player owns the page's tabs (scripts/screenshots.mjs). */}
            <div role="radiogroup" aria-label="View" className="hide-in-present mb-6 flex w-fit rounded-xl bg-sheet p-1 shadow-[var(--shadow-card)]">
              {([["cache", "Cache"], ["lab", "Matrix Lab"]] as const).map(([id, label]) => (
                <button key={id} type="button" role="radio" aria-checked={tab === id} data-view={id} onClick={() => setTab(id)}
                  className={`rounded-lg px-4 py-1.5 text-[15px] font-medium ${tab === id ? "bg-ink text-sheet" : "text-ink-soft hover:text-ink"}`}>
                  {label}
                </button>
              ))}
            </div>
            {tab === "lab" ? (
              <MatrixLab tau={tau} initialQuestion={result?.query ?? query} />
            ) : (
            <>
            <main className="grid items-start gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
              <div className="min-w-0 space-y-4">
                <QuestionBox
                  query={query}
                  onQuery={(q) => {
                    setQuery(q);
                    setAskError(null);
                    setReplay(null);
                  }}
                  onAsk={ask}
                  pending={pendingQuestion !== null}
                  forceRefresh={forceRefresh}
                  onForceRefresh={setForceRefresh}
                  followUpHint={followUpHint}
                  hasThread={Boolean(thread?.messages.length)}
                />
                <AnswerCard result={result} pending={pendingQuestion} error={askError}
                  onWrong={async (r) => {
                    const out = await api.wrongAnswer(r.query, r.matched_id as number);
                    await refresh();
                    setCacheVersion((v) => v + 1);
                    return out.note;
                  }} />
                <section className="card space-y-3 p-5">
                  <ThreadPicker
                    threads={threads}
                    activeId={threadId}
                    onSelect={(id) => openThread(id, user.name)}
                    onDelete={async (id) => {
                      await api.deleteThread(id);
                      await openThread(null, user.name);
                      await loadThreads();
                    }}
                  />
                  <div className="max-h-[34vh] overflow-y-auto pr-1">
                    <Conversation messages={thread?.messages ?? []} pendingQuestion={pendingQuestion} />
                  </div>
                </section>
              </div>
              <div className="min-w-0 xl:sticky xl:top-4">
                <LiveMathPanel
                  x={explanation}
                  loading={explaining}
                  error={explainError}
                  stage={stage}
                  onStage={goTo}
                  playing={playing}
                  onTogglePlay={togglePlay}
                  playKey={playKey}
                  tau={tau}
                  onTau={setTau}
                  learned={learned}
                  presenting={present}
                />
              </div>
            </main>

            <div className="hide-in-present mt-14 space-y-10">
              <StatsCards metrics={metrics} scope={scope} onScope={setScope} user={user.name} />
              <PerformanceChart metrics={metrics} />
              <section className="card p-5">
                <h2 className="mt-0 mb-4 text-[22px] font-semibold tracking-tight">Cached answers</h2>
                <CachePanel
                  entries={entries}
                  user={user.name}
                  isAdmin={user.is_admin}
                  onAdd={async (q, a, shared, ttl) => {
                    await api.addEntry(q, a, shared, ttl);
                    await refresh();
                    setCacheVersion((v) => v + 1);
                  }}
                  onDelete={async (id) => {
                    await api.deleteEntry(id);
                    await refresh();
                    setCacheVersion((v) => v + 1);
                  }}
                  onDeleteMany={async (ids) => {
                    await api.deleteEntries(ids);
                    await refresh();
                    setCacheVersion((v) => v + 1);
                  }}
                  onReset={async () => {
                    await api.reset();
                    setResult(null);
                    await Promise.all([refresh(), api.me().then(setUser)]);
                    setCacheVersion((v) => v + 1);
                  }}
                  onResetDemo={async () => {
                    await api.resetDemo();
                    setResult(null);
                    setLearned(null);
                    await Promise.all([refresh(), api.me().then(setUser), loadMetrics()]);
                    setCacheVersion((v) => v + 1);
                  }}
                  onResetMetrics={async () => {
                    await api.resetMetrics();
                    await loadMetrics();
                  }}
                />
              </section>
              {user.is_admin && (
                <AdminTools onChanged={async () => {
                  await refresh();
                  api.model().then(setModel).catch(() => undefined);
                  setCacheVersion((v) => v + 1);
                }} />
              )}
              <section className="card p-5">
                <h2 className="mt-0 mb-4 text-[22px] font-semibold tracking-tight">What the SVD learned</h2>
                <ModelPanel model={model} />
              </section>
            </div>
            </>
            )}
          </>
          )
        )}
      </div>
    </div>
  );
}
