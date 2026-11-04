// Shapes of the FastAPI responses (see apps/api.py).

export interface MatchItem {
  id: number;
  query: string;
  score: number; // the score the hit decision uses
  source: string; // "faq" | "ollama" | "stub" | "manual"
  owner: string | null; // null = shared with everyone
  lsa?: number; // present when hybrid scoring is on
  tfidf?: number;
  rho: number; // latent coverage of the cached question
  pair_rho: number; // min(question rho, cached question rho)
  scored_by: ScoredBy;
  hidden?: boolean; // another user's question: its text is not shown
}

// How a score was computed: the SVD (LSA), the LSA + TF-IDF blend, word overlap
// alone (the rho guard blocked the pair), or an exact repeat.
export type ScoredBy = "svd" | "hybrid" | "tfidf" | "exact";

export interface Timings {
  embed: number;
  lookup: number;
  llm: number;
  total: number;
}

export interface AskResult {
  query: string;
  hit: boolean;
  answer: string;
  source: string; // "cache" | "ollama" | "stub" | "none"
  score: number;
  threshold: number;
  matched_query: string | null;
  matched_id: number | null;
  matched_source: string | null;
  stored_id: number | null;
  matches: MatchItem[];
  timings_ms: Timings;
  backend: string | null;
  model: string | null;
  known_terms: string[];
  lookup_method: string; // "exact" | "brute" | "lsh" | "empty"
  compared: number;
  note: string | null;
  backend_error: string | null;
  user: string | null;
  thread_id: string | null;
  thread_title: string | null;
  follow_up: boolean; // depends on earlier messages of the conversation
  follow_up_reason: string | null;
  cache_used: boolean; // false: the shared cache was skipped
  history_used: number; // earlier messages sent to the model
  rho: number; // how much of the question survives in the SVD space
  scored_by: ScoredBy | null;
  answer_source: "CACHE" | "LLM" | "FOLLOW-UP" | "FALLBACK" | "NONE";
  similarity_pct: number | null;
  refreshed: boolean; // forced refresh: the cache was skipped on purpose
  prompt_tokens: number | null;
  output_tokens: number | null;
  stored_vector: number[] | null; // the new row of C after a miss
  cache_rows: number; // rows of C this user's lookups see, after this answer
  route: Route | null; // exact? -> guards -> score vs tau -> angle -> outcome
}

export interface LookupResult {
  hit: boolean;
  score: number;
  threshold: number;
  method: string;
  matches: MatchItem[];
  known_terms: string[];
  cache_used: boolean;
  follow_up: boolean;
  follow_up_reason: string | null;
  rho: number;
  scored_by: ScoredBy | null;
  note: string | null;
}

export interface UserStats {
  questions: number;
  hits: number;
  misses: number;
  follow_ups: number;
  hit_rate: number | null;
}

export interface CacheStats {
  entries: number;
  shared_entries: number;
  private_entries: number;
  max_entries: number;
  hits: number;
  misses: number;
  hit_rate: number | null;
  evictions: number;
  threshold: number;
  lsa_weight: number;
  lsh: boolean;
  by_source: Record<string, number>;
  memory_bytes: number;
  user?: UserStats; // present when signed in
}

export interface BackendInfo {
  backend: string;
  available: boolean;
  model?: string | null;
  url?: string;
  error?: string;
  fallback?: string;
  chain?: string[];
  api_key_set?: boolean;
  installed_models?: string[];
  warmup?: { state: "not needed" | "warming" | "ready" | "failed"; ms: number | null; error: string | null };
}

export interface ModelSummary {
  fingerprint: string;
  rank: number;
  vocabulary_size: number;
  training_documents: number;
  energy_kept: number;
}

export interface Health {
  status: string;
  profile: string; // "faq" or "quora"
  profiles: string[];
  model: ModelSummary;
  cache: CacheStats;
  backend: BackendInfo;
  defaults: { threshold: number; top_k: number; min_latent_coverage: number };
  follow_up_detection: boolean;
  admin_enabled: boolean; // SEMCACHE_ADMIN_KEY is set on the server
  model_status?: ModelStatus;
}

// GET /api/model/status: the growing model of profiles with corpus.include_cached (faq-live)
export interface ModelStatus {
  state: "idle" | "running" | "done" | "failed";
  reason: string | null;
  started_at: number | null;
  finished_at: number | null;
  seconds: number | null;
  error: string | null;
  generation: number;
  grows: boolean;
  rebuild_every: number;
  new_since_build: number;
  k: number;
  vocabulary_size: number;
  documents: number;
}

export interface Direction {
  direction: number;
  sigma: number;
  terms: string[];
}

export interface ModelInfo extends ModelSummary {
  singular_values: number[];
  energy_cumulative: number[];
  directions: Direction[];
}

export interface CacheEntry {
  id: number;
  query: string;
  answer: string;
  source: string;
  created_at: number;
  last_used_at: number;
  hits: number;
  model: string | null;
  owner: string | null; // null = shared with everyone
  expires_at?: number | null; // unix time after which it is a miss (null = never)
}

export interface ImportResult {
  profile: string;
  questions: number;
  entries: number;
  k: number;
  vocabulary_size: number;
  training_documents: number;
  folder: string;
  evaluable: boolean;
  serve: string;
}

export interface WrongAnswerResult {
  evicted: boolean;
  note: string;
  score: number | null;
  tau: number;
}

export interface Profile {
  name: string;
  is_admin: boolean;
  created_at: number;
  last_seen_at: number;
  stats: UserStats;
}

export interface ThreadSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  last_message: string | null;
}

export interface MessageMeta {
  hit?: boolean;
  source?: string;
  score?: number;
  matched_query?: string | null;
  follow_up?: boolean;
  follow_up_reason?: string | null;
  cache_used?: boolean;
  history_used?: number;
  stored_id?: number | null;
  model?: string | null;
}

export interface ThreadMessage {
  role: "user" | "assistant";
  content: string;
  created_at: number;
  meta: MessageMeta;
}

export interface ThreadDetail extends ThreadSummary {
  messages: ThreadMessage[];
}

export interface ResetResult {
  scope: "all" | "private";
  entries: number;
  faq_entries: number | null;
  removed: number | null;
}

// POST /api/explain: every number behind one hit-or-miss decision (see semcache/explain.py).
export interface ExplainToken {
  word: string;
  term: string | null; // the word after plural folding; null for a dropped stopword
  status: "stopword" | "known" | "unknown";
}

export interface ExplainTerm {
  term: string;
  tf: number;
  idf: number;
  weight: number; // unit-length TF-IDF weight
  bigram: boolean;
}

export interface LatentTop {
  dim: number;
  value: number;
  words: string[]; // the direction's strongest words overall
  from: string[]; // the question's words that put it there
}

export interface WordEffect {
  term: string;
  effect: number; // score minus the score without this word
  without: number;
  shared: boolean; // also in the cached question
  partner: string | null; // the cached word it connects to most
  partner_cos: number;
  row_total: number;
}

export interface DimPart {
  dim: number;
  value: number; // e_j * f_j
  q: number;
  c: number;
  words: string[];
  from_q: string[];
  from_c: string[];
}

export interface Reason {
  kind: "plus" | "minus" | "info";
  value: number | null;
  text: string;
}

export interface PairGrid {
  rows: string[]; // terms of your question
  cols: string[]; // terms of the cached question
  contribution: number[][]; // part of the cosine from each pair; all cells sum to the LSA cosine
  similarity: number[][]; // cosine between the two words in latent space
}

export interface MatchExplanation {
  rank: number;
  id: number;
  query: string;
  hidden: boolean;
  rho: number; // cached question rho
  pair_rho: number;
  rho_blocked: boolean; // scored by word overlap (TF-IDF), not the SVD
  scored_by: ScoredBy;
  source: string;
  owner: string | null;
  score: number;
  exact: boolean;
  hit: boolean;
  terms: { term: string; weight: number }[];
  lsa: number;
  tfidf: number;
  computed?: number;
  blocked: string | null;
  pairs: PairGrid | null;
  effects: WordEffect[];
  dims: DimPart[];
  c_latent: number[];
  reasons: Reason[];
}

export interface Explanation {
  query: string;
  normalized: string;
  threshold: number;
  verdict: "exact" | "empty" | "no_known_words" | "guard" | "hit" | "miss" | "refused";
  refused?: { rule: string; detail: string; entry_id: number; score: number }; // negation guard (off on faq)
  hit: boolean;
  score: number;
  method: string;
  compared: number;
  model: { rank: number; vocabulary_size: number; sigma_power: number; lsa_weight: number; min_latent_coverage: number };
  rho: number; // question rho
  rho_blocked: boolean;
  tokens: ExplainToken[];
  bigrams: { term: string; known: boolean }[];
  coverage: {
    known: number; total: number; ratio: number; min_ratio: number; passed: boolean;
    relaxed?: boolean; // passed only by the topic-word rule: matched on its known topic words
    topic_known?: string[]; topic_unknown?: string[]; everyday_unknown?: string[];
  };
  terms: ExplainTerm[];
  latent: number[];
  latent_top: LatentTop[];
  matches: MatchExplanation[];
  vocab_size: number;
  term_rows: TermRow[]; // the question's own rows of V_k
  vk_band: { term: string; index: number; vk: number[] }[]; // a fixed sample of other rows
  cache_matrix: CacheMatrix;
  follow_up: boolean; // how ask would treat it in the open conversation
  follow_up_reason: string | null;
  cache_used: boolean;
}

export interface TermRow {
  term: string;
  index: number; // column in the TF-IDF row / row in V_k
  weight: number;
  in_svd: boolean;
  vk: number[];
}

export interface CacheRow {
  id: number;
  query: string;
  hidden: boolean;
  source: string;
  score: number; // decision score
  lsa: number; // c_i . e
  scored_by: ScoredBy;
  rank: number | null; // 1-3 for the three best rows
  vector: number[];
}

export interface CacheMatrix {
  n: number; // rows of C this user's lookups see
  k: number;
  shown: number;
  sampled: boolean;
  rows: CacheRow[]; // in their order in C
}

export interface MetricsSummary {
  scope: "global" | "user";
  requests: number;
  hits: number;
  misses: number;
  hit_rate: number | null;
  hit_rate_label: string;
  hit_rate_basis: string;
  hit_rate_of_requests: number | null;
  llm_calls: number;
  llm_fallbacks: number;
  follow_ups: number;
  forced_refreshes: number;
  latency_ms: { avg: number | null; avg_hit: number | null; avg_llm: number | null; hit_samples: number; llm_samples: number };
  time_saved_ms: number | null;
  time_saved_basis: string;
  tokens: { input: number; output: number; calls_with_counts: number };
  by_backend?: Record<string, number>; // LLM calls per backend that answered
  cost: null | {
    estimated: true;
    label: string;
    note: string | null;
    llm_cost_usd: number | null;
    savings_usd: number | null;
    hits_priced_from_entry: number;
    hits_priced_from_average: number;
  };
}

export interface MetricsPoint {
  t: number;
  requests: number;
  hits: number;
  misses: number;
  total_requests: number;
  hit_rate: number | null;
}

export interface Metrics {
  profile: string;
  pricing: { configured: boolean; label: string | null; note: string | null };
  summary: MetricsSummary;
  series: { bucket_seconds: number; points: MetricsPoint[]; truncated: boolean };
}

// ------------------------------------------------------------------ route trace (semcache/route.py)
export interface RouteStep {
  id: string; // exact | known | rho | negation | score | angle | cache
  label: string;
  value: string;
  status: "yes" | "no" | "pass" | "fail" | "info" | "off" | "skipped";
  detail: string;
}

export interface Route {
  steps: RouteStep[];
  outcome: { kind: "HIT" | "LLM" | "BLOCKED"; guard: string | null; detail: string; then: string | null };
}

// ------------------------------------------------------------------ Matrix Lab (semcache/lab.py)
export interface LabCorrectness {
  profile: string;
  shape: [number, number];
  nnz: number;
  k: number;
  frobenius_sq: number;
  eckart_young: {
    residual_sq: number;
    dropped_sigma_sq: number;
    difference: number;
    relative: number;
    equal: boolean;
    residual_method: string;
    dropped_method: string;
  };
  orthogonality: { value: number; ok: boolean };
  sigma_check: { max_abs_diff: number | null };
  jacobi: {
    path: string;
    size: number;
    sweeps: number;
    rotations: number;
    seconds: number;
    converged: boolean;
    tol: number;
    frobenius: number;
    off_norms: number[];
    svd_seconds: number;
  };
  spectrum: { sigma: number[]; energy_cumulative: number[]; full: boolean; k: number; energy_kept: number };
}

export interface LabTerm {
  term: string;
  tf: number;
  df: number;
  idf: number;
  tf_weight: number;
  raw: number;
  weight: number;
  bigram: boolean;
}

export interface LabWorked {
  query: string;
  N: number;
  smooth_idf: boolean;
  sublinear_tf: boolean;
  rarest: LabTerm | null;
  terms: LabTerm[];
  norm: number;
  k: number;
  latent_norm: number;
  rho: number;
  e: number[];
  e_top: { j: number; value: number }[];
  tau: number;
  theta_tau_deg: number;
  hit: boolean;
  refused: { rule: string; detail: string } | null;
  match: null | {
    query: string;
    hidden: boolean;
    exact: boolean;
    score: number;
    lsa: number;
    tfidf: number | null;
    scored_by: ScoredBy;
    lsa_weight: number;
    top: { j: number; e: number; f: number; product: number }[];
    rest: number;
    terms_in_sum: number;
    theta_deg: number;
  };
  route: Route;
}

export interface LabOps {
  query: string;
  nnz: number;
  k: number;
  n: number;
  hit: { parts: { step: string; formula: string; flops: number }[]; flops: number };
  llm: null | {
    model: string;
    model_basis: string;
    params_billion: number;
    prompt_tokens: number;
    output_tokens: number;
    token_basis: string;
    flops: number;
    formula: string;
    note: string;
  };
  ratio: number | null;
  missing: string | null;
}

export interface LabHeadline {
  auc: number;
  precision: number;
  recall: number;
  accepted_hits: number;
  accepted_hits_correct: number;
  random_false_hits: number | null;
  random_asked: number | null;
  best_f1: number;
  best_f1_tau: number;
  tau: number;
}

export interface LabRefit {
  k: number;
  requested_k: number;
  seconds: number;
  vocabulary_size: number;
  energy_kept: number;
  sigma: number[];
  eckart_young: { residual_sq: number; dropped_sigma_sq: number; equal: boolean };
  headline: LabHeadline;
  served_headline: LabHeadline;
  served: { k: number; fingerprint: string };
  question: null | { query: string; sandbox_score: number; sandbox_hit: boolean; served_score: number; served_hit: boolean };
}

export interface LabLatencyPoint {
  t: number;
  kind: "hit" | "miss" | "follow_up" | "refresh";
  total_ms: number | null;
  llm_ms: number | null;
  llm: boolean;
  fallback: boolean;
  backend: string | null;
}
