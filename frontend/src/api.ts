import type {
  AskResult,
  CacheEntry,
  Explanation,
  Metrics,
  Health,
  ImportResult,
  LabCorrectness,
  LabLatencyPoint,
  LabOps,
  LabRefit,
  LabWorked,
  LookupResult,
  ModelInfo,
  ModelStatus,
  Profile,
  ResetResult,
  ThreadDetail,
  ThreadSummary,
  WrongAnswerResult,
} from "./types";

// Same-origin "/api": in development Vite proxies it to http://127.0.0.1:8000,
// in production FastAPI serves this app and the API together.
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

// The signed-in name, sent as X-User on every request, and the admin passphrase,
// sent as X-Admin-Key when the user entered one (see apps/api.py).
let currentUser: string | null = null;
let adminKey: string | null = null;
export function setApiUser(name: string | null) {
  currentUser = name;
}
export function setApiAdminKey(key: string | null) {
  adminKey = key && key.trim() ? key.trim() : null;
}

async function request<T>(path: string, init?: RequestInit, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (currentUser) headers["X-User"] = currentUser;
  if (adminKey) headers["X-Admin-Key"] = adminKey;
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, headers, signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError("Can't reach the API. Start it with: python -m semcache serve");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),
  model: () => request<ModelInfo>("/model"),
  signIn: (name: string) => request<Profile>("/session", post({ name })),
  me: () => request<Profile>("/me"),

  threads: () => request<ThreadSummary[]>("/threads"),
  thread: (id: string) => request<ThreadDetail>(`/threads/${id}`),
  renameThread: (id: string, title: string) =>
    request<ThreadSummary>(`/threads/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteThread: (id: string) => request<{ deleted: string }>(`/threads/${id}`, { method: "DELETE" }),

  ask: (query: string, threshold: number, threadId: string | null, forceRefresh = false) =>
    request<AskResult>(
      "/ask",
      post({ query, threshold, thread_id: threadId, ...(forceRefresh ? { use_cache: false } : {}) }),
    ),
  lookup: (query: string, threshold: number, threadId: string | null, signal?: AbortSignal) =>
    request<LookupResult>("/lookup", post({ query, threshold, thread_id: threadId }), signal),
  explain: (query: string, threshold: number, threadId: string | null, signal?: AbortSignal) =>
    request<Explanation>("/explain", post({ query, threshold, thread_id: threadId }), signal),
  metrics: (scope: "global" | "user", bucket = "10s") =>
    request<Metrics>(`/metrics?scope=${scope}&bucket=${bucket}`),
  resetMetrics: () => request<{ removed_events: number }>("/metrics/reset", { method: "POST" }),
  resetDemo: () => request<{ entries: number }>("/reset-demo", { method: "POST" }),

  entries: (limit = 200) => request<CacheEntry[]>(`/entries?limit=${limit}`),
  addEntry: (question: string, answer: string, shared = false, ttlSeconds: number | null = null) =>
    request<CacheEntry>("/entries", post({ question, answer, shared, ...(ttlSeconds !== null ? { ttl_seconds: ttlSeconds } : {}) })),
  importCsv: (profile: string, csv: string) => request<ImportResult>("/admin/import-csv", post({ profile, csv })),
  deleteEntries: (ids: number[]) =>
    request<{ removed: number; missing: number[]; entries: number }>("/admin/entries/delete", post({ ids })),
  forgetLlm: () => request<{ removed: number; entries: number }>("/admin/forget-llm", { method: "POST" }),
  wrongAnswer: (query: string, entryId: number) =>
    request<WrongAnswerResult>("/feedback/wrong", post({ query, entry_id: entryId })),
  deleteEntry: (id: number) => request<{ deleted: number }>(`/entries/${id}`, { method: "DELETE" }),
  reset: () => request<ResetResult>("/reset", { method: "POST" }),

  modelStatus: () => request<ModelStatus>("/model/status"),
  rebuildModel: () => request<ModelStatus>("/admin/rebuild", { method: "POST" }),

  // Matrix Lab: none of these change the served model
  labCorrectness: () => request<LabCorrectness>("/lab/correctness"),
  labWorked: (query: string, threshold: number, signal?: AbortSignal) =>
    request<LabWorked>("/lab/worked", post({ query, threshold }), signal),
  labOps: (query: string, signal?: AbortSignal) => request<LabOps>("/lab/ops", post({ query }), signal),
  labRefit: (k: number, query: string | null) => request<LabRefit>("/lab/refit", post({ k, query })),
  labLatency: (limit = 500) => request<{ points: LabLatencyPoint[] }>(`/lab/latency?limit=${limit}`),
};
