"""The service every interface uses (CLI, FastAPI, Gradio, notebook).

    ask(query):
        1. embed      e = LSAEmbedder.embed_one(query)       -> linalg: TF-IDF, fold-in, normalise
        2. lookup     scores = C e, best >= tau ?            -> linalg: cosine_scores / LSH
        3a. hit       return the cached answer
        3b. miss      ask the LLM backend, store (query, answer, e) in the cache

Many users share one service. ``ask(query, user=..., thread_id=...)`` also keeps
the conversation: the question and answer are added to the user's thread, and a
follow-up ("what about the second one?") skips the shared cache and goes to the
LLM with the thread's recent history; its answer is not cached, because it only
makes sense inside that conversation. Standalone questions use the cache as
above, and their LLM answers are generated without history so they are safe to
share with everyone.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .backends import StubBackend, build_backend
from .cache import ALL_USERS, CacheEntry, LookupResult, Match, SemanticCache
from .config import ADMIN_KEY_ENV, Settings
from .context import STANDALONE, FollowUp, build_history, detect_follow_up
from .data import faq_corpus, load_faq, load_question_pairs, unique_questions
from .embedder import LSAEmbedder
from .explain import explain_query
from .metrics import MetricsStore, Pricing, parse_bucket
from .users import Message, Thread, UserStore, normalize_username

MAX_QUERY_CHARS = 1000
HIDDEN_QUESTION = "another user's question"


def admin_key_configured() -> bool:
    return bool(os.environ.get(ADMIN_KEY_ENV, "").strip())


def check_admin_key(provided: str | None) -> bool:
    """True when ``provided`` matches SEMCACHE_ADMIN_KEY; False when no key was sent.

    Raises PermissionError when a key is sent but admin actions are turned off (the
    variable isn't set) or the key is wrong.
    """
    if provided is None or not provided.strip():
        return False
    expected = os.environ.get(ADMIN_KEY_ENV, "").strip()
    if not expected:
        raise PermissionError(ADMIN_DISABLED)
    if not hmac.compare_digest(provided.strip().encode(), expected.encode()):
        raise PermissionError("wrong admin key")
    return True


ADMIN_DISABLED = (f"admin actions are turned off on this server: start it with the {ADMIN_KEY_ENV} "
                  "environment variable set to a passphrase")


def admin_required(action: str) -> PermissionError:
    if not admin_key_configured():
        return PermissionError(f"{action} needs an admin, and {ADMIN_DISABLED}")
    return PermissionError(f"{action} needs an admin: send the admin key (X-Admin-Key)")


@dataclass
class AskResult:
    query: str
    hit: bool
    answer: str
    source: str                         # "cache", "ollama", "stub" or "none"
    score: float                        # the best match's score (see scored_by)
    threshold: float
    matched_query: str | None = None
    matched_id: int | None = None
    matched_source: str | None = None
    stored_id: int | None = None        # id of the new entry after a miss
    matches: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    backend: str | None = None
    model: str | None = None
    known_terms: list[str] = field(default_factory=list)
    lookup_method: str = "brute"
    compared: int = 0
    note: str | None = None             # why a semantic match was or wasn't allowed
    backend_error: str | None = None    # set when Ollama failed and the stub answered
    user: str | None = None             # who asked (None = anonymous, shared entries only)
    thread_id: str | None = None        # the conversation the question was added to
    thread_title: str | None = None
    follow_up: bool = False             # depends on earlier messages of the thread
    follow_up_reason: str | None = None
    cache_used: bool = True             # False: the cache was skipped (follow-up or turned off)
    history_used: int = 0               # earlier messages sent to the LLM
    rho: float = 1.0                    # latent coverage of the question: ||q V_k|| / ||q||
    scored_by: str | None = None        # how the best match was scored: svd, hybrid, tfidf, exact
    answer_source: str = "NONE"         # CACHE, LLM, FOLLOW-UP, FALLBACK or NONE (model not called)
    similarity_pct: float | None = None # the best match's score as a percentage
    refreshed: bool = False             # forced refresh: the cache was skipped on purpose
    prompt_tokens: int | None = None    # reported by Ollama for this answer
    output_tokens: int | None = None
    stored_vector: list[float] | None = None   # the new row of C after a miss (for the math view)
    cache_rows: int = 0                 # rows of C this user's lookups see, after this answer

    def to_dict(self) -> dict:
        return asdict(self)


def build_corpus(settings: Settings, verbose: bool = False) -> tuple[list[str], dict]:
    """Texts the SVD is fitted on: FAQ questions + paraphrases, plus Quora questions if present."""
    corpus: list[str] = []
    info = {"faq_texts": 0, "quora_questions": 0, "quora_file": None}
    if settings.corpus.use_faq:
        texts = faq_corpus(load_faq(settings.path(settings.paths.faq_csv)))
        corpus += texts
        info["faq_texts"] = len(texts)
    quora_path = settings.path(settings.paths.quora_train)
    if settings.corpus.use_quora and settings.corpus.require_quora and not quora_path.exists():
        raise FileNotFoundError(f"the {settings.profile!r} profile needs the Quora data at {quora_path}; "
                                "run: pip install datasets && python scripts/download_qqp.py")
    if settings.corpus.use_quora and quora_path.exists():
        pairs = load_question_pairs(quora_path)
        questions = unique_questions(pairs, settings.corpus.quora_max_questions, settings.corpus.seed)
        corpus += questions
        info.update(quora_questions=len(questions), quora_file=str(quora_path))
    elif verbose and settings.corpus.use_quora:
        print(f"  (no Quora file at {quora_path}; fitting on the FAQ only. "
              "Run scripts/download_qqp.py for the full corpus.)")
    if not corpus:
        raise ValueError("the corpus is empty: enable use_faq or add the Quora data")
    return corpus, info


def cache_params(settings: Settings) -> dict:
    c = settings.cache
    return {"threshold": c.threshold, "max_entries": c.max_entries, "use_lsh": c.use_lsh,
            "lsh_bits": c.lsh_bits, "lsh_tables": c.lsh_tables, "lsh_min_entries": c.lsh_min_entries,
            "lsa_weight": c.lsa_weight, "min_latent_coverage": c.min_latent_coverage}


def new_user_store(settings: Settings) -> UserStore:
    u = settings.users
    return UserStore(settings.users_dir, u.max_threads_per_user, u.max_messages_per_thread)


def new_cache(settings: Settings, embedder: LSAEmbedder) -> SemanticCache:
    return SemanticCache(embedder.dim, model_fingerprint=embedder.fingerprint, **cache_params(settings))


def warm_cache(cache: SemanticCache, settings: Settings, embedder: LSAEmbedder) -> int:
    """Preload the FAQ question/answer pairs."""
    if not settings.cache.warm_with_faq:
        return 0
    items = load_faq(settings.path(settings.paths.faq_csv))
    vectors, rows, rho = embedder.encode_with_rho([i.question for i in items], settings.cache.min_known_ratio)
    cache.add_many([(i.question, i.answer) for i in items], vectors, source="faq", tfidf_rows=rows, rhos=rho)
    return len(items)


def build_artifacts(settings: Settings | None = None, verbose: bool = True,
                    fresh: bool = False) -> tuple[LSAEmbedder, SemanticCache]:
    """Fit the embedder on the corpus and save it to artifacts/.

    An existing cache keeps its answers: each stored question is re-embedded with the new
    model. ``fresh=True`` starts over with the FAQ answers only.
    """
    settings = settings or Settings.load()
    if verbose:
        print(f"Building the semantic cache (profile {settings.profile!r})")
    corpus, info = build_corpus(settings, verbose)
    if verbose:
        print(f"  corpus: {len(corpus):,} texts (FAQ {info['faq_texts']:,}, "
              f"Quora {info['quora_questions']:,})")
    embedder = LSAEmbedder(settings.embedder).fit(corpus, verbose=verbose)
    for note in embedder.report.notes:
        if verbose:
            print(f"  note: {note}")
    embedder.save(settings.model_dir)
    kept = 0
    if not fresh and SemanticCache.exists(settings.cache_dir):
        cache = SemanticCache.load(settings.cache_dir, embedder, settings.cache.min_known_ratio,
                                   **cache_params(settings))
        kept = len(cache)
    else:
        cache = new_cache(settings, embedder)
    n = warm_cache(cache, settings, embedder)
    cache.save(settings.cache_dir)
    if verbose:
        if kept:
            print(f"  kept {kept} cached answers, re-embedded with the new model (--fresh to start over)")
        print(f"  cache has {len(cache)} entries ({n} from the FAQ)")
        print(f"  saved to {settings.artifacts_dir}")
    return embedder, cache


class SemanticCacheService:
    def __init__(self, embedder: LSAEmbedder, cache: SemanticCache, backend, settings: Settings,
                 users: UserStore | None = None):
        self.embedder, self.cache, self.backend, self.settings = embedder, cache, backend, settings
        self.users = users or new_user_store(settings)
        self.metrics = MetricsStore(settings.metrics_dir, settings.metrics.enabled)
        # "not needed" (no model to load), "warming", "ready" or "failed"
        self.warmup: dict = {"state": "not needed", "ms": None, "error": None}
        self._lock = threading.RLock()
        self._unsaved = 0

    @classmethod
    def from_settings(cls, settings: Settings | None = None, build_if_missing: bool = True,
                      verbose: bool = True) -> "SemanticCacheService":
        settings = settings or Settings.load()
        if not LSAEmbedder.exists(settings.model_dir):
            if not build_if_missing:
                raise FileNotFoundError(f"no built model for profile {settings.profile!r}; run "
                                        f"`python -m semcache build --profile {settings.profile}`")
            embedder, cache = build_artifacts(settings, verbose)
        else:
            embedder = LSAEmbedder.load(settings.model_dir)
            if SemanticCache.exists(settings.cache_dir):
                cache = SemanticCache.load(settings.cache_dir, embedder, settings.cache.min_known_ratio,
                                           **cache_params(settings))
            else:
                cache = new_cache(settings, embedder)
                warm_cache(cache, settings, embedder)
        return cls(embedder, cache, build_backend(settings.llm), settings)

    def start_warm_up(self) -> threading.Thread | None:
        """Load the LLM in the background (the API calls this at startup). The header shows
        "Model warming up…" while ``warmup["state"]`` is "warming"."""
        if not callable(getattr(self.backend, "warm_up", None)) or isinstance(self.backend, StubBackend):
            return None
        self.warmup = {"state": "warming", "ms": None, "error": None}

        def run() -> None:
            t0 = time.perf_counter()
            try:
                self.backend.warm_up()
                self.warmup = {"state": "ready", "ms": round((time.perf_counter() - t0) * 1000), "error": None}
            except Exception as exc:          # Ollama down: the fallback still answers
                self.warmup = {"state": "failed", "ms": None, "error": str(exc)}

        thread = threading.Thread(target=run, name="llm-warm-up", daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------ users
    def profile(self, user: str, admin: bool = False) -> dict:
        """Sign in: creates the user on first use. ``admin`` comes from the admin key."""
        p = self.users.profile(user)
        return {**p.to_dict(), "is_admin": bool(admin)}

    # ------------------------------------------------------------ privacy
    @staticmethod
    def can_read(entry: CacheEntry, user: str | None, admin: bool = False) -> bool:
        """May ``user`` see the text of this cached question? FAQ entries, shared answers an
        admin added, and their own entries (private, or shared answers to their own
        questions) yes; other users' questions no. Admins see everything."""
        if admin or entry.source == "faq" or (entry.source == "manual" and entry.owner is None):
            return True
        return user is not None and (entry.owner == user or entry.asked_by == user)

    def _match_dict(self, m: Match, user: str | None, admin: bool) -> dict:
        d = m.to_dict()
        d["hidden"] = not self.can_read(m.entry, user, admin)
        if d["hidden"]:
            d["query"] = HIDDEN_QUESTION
        return d

    def follow_up_for(self, query: str, user: str | None = None, thread_id: str | None = None,
                      use_cache: bool | None = None) -> tuple[bool, FollowUp]:
        """How ``ask`` would treat this question: (use the cache?, follow-up verdict)."""
        thread = self._thread(user, thread_id) if thread_id else None
        cache_used, follow, _ = self._plan(query, thread, use_cache)
        return cache_used, follow

    # ------------------------------------------------------------------ core
    def lookup(self, query: str, threshold: float | None = None, top_k: int | None = None,
               record: bool = False, user: str | None = None) -> tuple[LookupResult, list[str]]:
        """Score a query against the entries ``user`` can see, without calling the LLM
        or storing anything."""
        query = self._validate(query)
        user = self._user(user)
        vector, row, rho = self._encode(query)
        with self._lock:
            result = self.cache.lookup(vector, top_k or self.settings.cache.top_k, threshold, record,
                                       query_text=query, query_tfidf=row, user=user, query_rho=rho)
        return result, self.embedder.known_terms(query)

    def match_dicts(self, result: LookupResult, user: str | None = None, admin: bool = False) -> list[dict]:
        """The matches as JSON-ready dicts, with other users' questions hidden."""
        return [self._match_dict(m, self._user(user), admin) for m in result.matches]

    def explain(self, query: str, threshold: float | None = None, top_k: int | None = None,
                user: str | None = None, admin: bool = False, thread_id: str | None = None) -> dict:
        """Every number behind the hit-or-miss decision for ``query`` (see semcache/explain.py),
        plus how ``ask`` would treat it in the conversation ``thread_id``. Nothing is
        stored, nothing is counted and the LLM is not called."""
        query = self._validate(query)
        user = self._user(user)
        cache_used, follow = (self.follow_up_for(query, user, thread_id)
                              if user is not None and thread_id else (True, STANDALONE))
        with self._lock:
            out = explain_query(self.embedder, self.cache, query, threshold=threshold,
                                top_k=top_k or self.settings.cache.top_k, user=user,
                                min_known_ratio=self.settings.cache.min_known_ratio,
                                can_read=lambda e: self.can_read(e, user, admin))
        out.update(follow_up=follow.is_follow_up, follow_up_reason=follow.reason, cache_used=cache_used)
        return out

    def rho_note(self, lookup: LookupResult) -> str:
        """Plain-language note about the latent-coverage guard for this lookup."""
        rho, cut, k = lookup.rho, self.cache.min_latent_coverage, self.embedder.dim
        best = lookup.matches[0] if lookup.matches else None
        if lookup.method == "exact":
            return f"ρ = {rho:.2f} (not needed: an exact repeat hits without any similarity score)."
        if cut > 0 and rho < cut:
            return (f"ρ = {rho:.2f}: only {rho:.0%} of this question survives in the {k}-dimensional "
                    f"SVD space (below {cut:.2f}), so it is scored by word overlap (TF-IDF), not by the SVD.")
        if best is not None and best.blocked:
            return (f"ρ = {rho:.2f} for this question, but only {best.rho:.0%} of the closest cached "
                    f"question survives the SVD (below {cut:.2f}), so that pair is scored by word "
                    "overlap (TF-IDF), not by the SVD.")
        return f"ρ = {rho:.2f}: {rho:.0%} of this question survives in the {k}-dimensional SVD space."

    def ask(self, query: str, threshold: float | None = None, top_k: int | None = None,
            use_llm: bool = True, *, user: str | None = None, thread_id: str | None = None,
            use_cache: bool | None = None, admin: bool = False) -> AskResult:
        """Answer one question.

        Without ``user`` this is a one-off, anonymous question against the shared cache.
        With ``user`` the question joins the thread ``thread_id`` (a new thread when it is
        None), and follow-ups are detected. ``use_cache=True`` always uses the cache;
        ``use_cache=False`` is a forced refresh: the cache is skipped and the model answers
        (see _store_refresh for what is kept). ``admin`` (or a trusted caller such as the
        CLI) sees other users' question text in the matches.
        """
        query = self._validate(query)
        user = self._user(user)
        if thread_id is not None and user is None:
            raise ValueError("a conversation thread needs a user")
        # A new thread is created only once there is an answer to put in it (see _remember);
        # until then it has no history, so the question is planned as standalone.
        thread = self._thread(user, thread_id) if thread_id else None
        cache_used, follow, refresh = self._plan(query, thread, use_cache)
        history = (build_history([{"role": m.role, "content": m.content} for m in thread.messages],
                                 self.settings.users.history_messages, self.settings.users.history_chars)
                   if thread is not None and follow.is_follow_up and not cache_used else [])

        ratio = self.settings.cache.min_known_ratio
        t0 = time.perf_counter()
        vector, row, rho = self._encode(query)
        t1 = time.perf_counter()
        with self._lock:
            # A follow-up or refresh is still scored (so the UI can show the neighbours)
            # but not recorded: it never counts as a hit or a miss.
            lookup = self.cache.lookup(vector, top_k or self.settings.cache.top_k, threshold,
                                       record=cache_used, query_text=query, query_tfidf=row, user=user,
                                       query_rho=rho)
        t2 = time.perf_counter()
        known, total = self.embedder.coverage(query)
        hit = lookup.hit and cache_used

        result = AskResult(
            query=query, hit=hit, answer="", source="cache", score=round(lookup.score, 4),
            threshold=lookup.threshold, matches=self.match_dicts(lookup, user, admin),
            known_terms=self.embedder.known_terms(query), lookup_method=lookup.method,
            compared=lookup.compared, user=user, follow_up=follow.is_follow_up,
            follow_up_reason=follow.reason, cache_used=cache_used, history_used=len(history),
            rho=round(float(rho), 4), refreshed=refresh,
        )
        if lookup.matches:
            top = lookup.matches[0]
            result.matched_query, result.matched_id = result.matches[0]["query"], top.entry.id
            result.matched_source, result.scored_by = top.entry.source, top.scored_by
            result.similarity_pct = round(100 * max(0.0, lookup.score), 1)
        if follow.is_follow_up and not cache_used:
            context = (f"with {len(history)} earlier message{'s' if len(history) != 1 else ''} of this "
                       "conversation" if history else "without conversation history")
            result.note = (f"Follow-up ({follow.reason}): answered by the model {context}. The shared cache "
                           "was not used, and this answer is not stored in it.")
        elif refresh:
            result.note = "Force refresh: the cache was skipped and the model answered."
        elif lookup.method == "exact":
            result.note = "Exact repeat of a cached question."
        elif total == 0 or not result.known_terms:
            result.note = "None of these words are in the model's vocabulary, so only an exact repeat can hit."
        elif known / total < ratio:
            result.note = (f"Only {known} of {total} words are known to the model, so a semantic match "
                           f"isn't trusted (min_known_ratio = {ratio}); only an exact repeat can hit.")
        if total and result.known_terms:
            result.note = " ".join(x for x in (result.note, self.rho_note(lookup)) if x)

        llm_ms, gen = 0.0, None
        if hit:
            result.answer = lookup.best.answer
            result.answer_source = "CACHE"
        elif use_llm:
            # Standalone questions call generate(prompt) exactly as before, so backends
            # that don't know about history keep working.
            gen = self.backend.generate(query, history) if history else self.backend.generate(query)
            llm_ms = gen.latency_ms
            result.answer, result.source = gen.text, gen.backend
            result.backend, result.model = gen.backend, gen.model
            result.prompt_tokens, result.output_tokens = gen.prompt_tokens, gen.output_tokens
            result.answer_source = ("FALLBACK" if gen.fallback else
                                    "FOLLOW-UP" if follow.is_follow_up and not cache_used else "LLM")
            if refresh:
                result.stored_id, outcome = self._store_refresh(query, gen, vector, row, rho, user)
                result.note = " ".join(x for x in (result.note, outcome) if x)
            elif gen.cacheable and cache_used:
                result.stored_id = self._store_shared(query, gen, vector, row, rho, user)
            if gen.error:
                result.backend_error = gen.error
        else:
            result.source = "none"
            result.answer_source = "NONE"
        t3 = time.perf_counter()
        if result.stored_id is not None:
            vec = self.cache.vector_of(result.stored_id)
            result.stored_vector = None if vec is None else [round(float(x), 4) for x in vec]
        with self._lock:
            result.cache_rows = len(self.cache.entries(user=user))
        result.timings_ms = {
            "embed": round((t1 - t0) * 1000, 3),
            "lookup": round((t2 - t1) * 1000, 3),
            "llm": round(llm_ms, 1),
            "total": round((t3 - t0) * 1000, 3),
        }
        self._record(result, lookup, gen, user)
        if user is not None:
            self._remember(user, thread, result)
        return result

    def _record(self, r: AskResult, lookup: LookupResult, gen, user: str | None) -> None:
        """One metrics event per answered question (see semcache/metrics.py)."""
        kind = ("follow_up" if r.follow_up and not r.cache_used else "refresh" if r.refreshed
                else "hit" if r.hit else "miss")
        entry_tokens = None
        if r.hit and lookup.best is not None:
            best = lookup.best
            if best.prompt_tokens is not None or best.output_tokens is not None:
                entry_tokens = [best.prompt_tokens, best.output_tokens]
        self.metrics.record({
            "user": user, "kind": kind, "source": r.answer_source, "llm": gen is not None,
            "fallback": bool(gen is not None and gen.fallback), "backend": gen.backend if gen else None,
            "total_ms": r.timings_ms["total"], "llm_ms": r.timings_ms["llm"],
            "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
            "entry_tokens": entry_tokens, "score": r.score, "rho": r.rho,
        })

    def _store_shared(self, query: str, gen, vector, row, rho: float = 1.0,
                      asked_by: str | None = None) -> int:
        # The lock is released while the LLM runs, so a concurrent request for the same
        # question may have stored an answer meanwhile (the lookup would then have hit an
        # exact repeat). Keep that first answer instead of replacing it.
        with self._lock:
            entry = self.cache.find_exact(query)
            if entry is None:
                entry = self.cache.add(query, gen.text, vector, source=gen.backend, model=gen.model,
                                       tfidf_row=row, rho=float(rho), asked_by=asked_by,
                                       prompt_tokens=gen.prompt_tokens, output_tokens=gen.output_tokens)
                self._autosave()
            return entry.id

    def _store_refresh(self, query: str, gen, vector, row, rho: float,
                       user: str | None) -> tuple[int | None, str]:
        """What a forced refresh keeps.

        * No cached answer for this exact question: the fresh one is stored, like a miss.
        * The cached answer came from the LLM: it is replaced by the fresh one.
        * It is an FAQ, admin-added or placeholder answer: it is never overwritten; the
          fresh answer is shown but not stored.
        A fallback (placeholder) answer is never stored.
        """
        if not gen.cacheable:
            return None, "The fresh answer is a placeholder, so it is not stored."
        with self._lock:
            existing = self.cache.find_exact(query, user)
            if existing is not None and not (existing.source == "ollama" and existing.owner is None):
                kind = {"faq": "an FAQ", "manual": "an admin-added" if existing.owner is None else "your own",
                        "stub": "a placeholder"}.get(existing.source, f"a {existing.source}")
                return None, (f"Not stored: the cached answer to this exact question is {kind} answer, "
                              "which a refresh never overwrites.")
            entry = self.cache.add(query, gen.text, vector, source=gen.backend, model=gen.model,
                                   tfidf_row=row, rho=float(rho), asked_by=user,
                                   prompt_tokens=gen.prompt_tokens, output_tokens=gen.output_tokens)
            self._autosave()
        if existing is not None:
            return entry.id, f"Replaced the cached LLM answer #{existing.id} with this one."
        return entry.id, "Stored as a new cached answer."

    def _plan(self, query: str, thread: Thread | None,
              use_cache: bool | None) -> tuple[bool, FollowUp, bool]:
        """(use the cache?, follow-up verdict, forced refresh?)"""
        has_history = bool(thread and thread.messages)
        if use_cache is True:
            return True, STANDALONE, False
        follow = (detect_follow_up(query, has_history)
                  if has_history and self.settings.users.detect_follow_ups else STANDALONE)
        if use_cache is False:
            # A refreshed follow-up still needs its conversation; a standalone one is a refresh.
            return False, follow, not follow.is_follow_up
        return not follow.is_follow_up, follow, False

    def _remember(self, user: str, thread: Thread | None, r: AskResult) -> None:
        """Add the exchange to the thread and count it in the user's stats."""
        self.users.record(user, hit=r.hit if r.cache_used else None, follow_up=r.follow_up)
        if r.source == "none":                  # nothing was answered: leave the thread alone
            if thread is not None:
                r.thread_id, r.thread_title = thread.id, thread.title
            return
        if thread is None:
            thread = self.users.create_thread(user)
        r.thread_id = thread.id
        now = time.time()
        meta: dict[str, Any] = {"hit": r.hit, "source": r.source, "score": r.score, "rho": r.rho,
                                "scored_by": r.scored_by,
                                "matched_query": r.matched_query if r.hit else None,
                                "follow_up": r.follow_up, "follow_up_reason": r.follow_up_reason,
                                "cache_used": r.cache_used, "history_used": r.history_used,
                                "stored_id": r.stored_id, "model": r.model}
        updated = self.users.append(user, thread.id, Message("user", r.query, now),
                                    Message("assistant", r.answer, now, meta))
        r.thread_title = updated.title

    # ------------------------------------------------------------ management
    def visible_entries(self, user: str | None = None, limit: int | None = None,
                        everyone: bool = False) -> list[CacheEntry]:
        """Entries ``user`` can see (shared + their own), newest first. ``everyone=True``
        lists all entries, private ones included (for admins and the CLI)."""
        with self._lock:
            items = self.cache.entries(user=ALL_USERS if everyone else self._user(user))
        return items[:limit] if limit else items

    def listed_entries(self, user: str | None = None, admin: bool = False,
                       limit: int | None = None) -> list[CacheEntry]:
        """What /api/entries shows: everything for an admin; otherwise the FAQ entries, shared
        answers an admin added, and the user's own entries, newest first."""
        user = self._user(user)
        with self._lock:
            items = [e for e in self.cache.entries(user=ALL_USERS) if self.can_read(e, user, admin)]
        return items[:limit] if limit else items

    def add_entry(self, question: str, answer: str, source: str = "manual", user: str | None = None,
                  shared: bool = False, admin: bool = False) -> CacheEntry:
        """Add an answer by hand. With a ``user`` it is private to them, unless ``shared``
        is set by an admin. Without a user (CLI, scripts) it is shared."""
        question = self._validate(question)
        user = self._user(user)
        if not answer or not answer.strip():
            raise ValueError("answer must not be empty")
        if user is not None and shared and not admin:
            raise admin_required("adding an answer shared with everyone")
        owner = None if user is None or shared else user
        vector, row, rho = self._encode(question)
        with self._lock:
            entry = self.cache.add(question, answer.strip(), vector, source=source, tfidf_row=row,
                                   owner=owner, rho=float(rho))
        self._autosave()
        return entry

    def remove_entry(self, entry_id: int, user: str | None = None, admin: bool = False) -> bool:
        """Delete an entry. A user may delete their own private entries; shared entries
        need an admin. Without a user (CLI) any entry can be deleted."""
        user = self._user(user)
        with self._lock:
            entry = self.cache.get(entry_id)
            if entry is None or (user is not None and not admin and entry.owner not in (None, user)):
                return False                     # someone else's private entry: not visible
            if user is not None and entry.owner is None and not admin:
                raise admin_required("removing a shared answer")
            removed = self.cache.remove(entry_id)
        if removed:
            self._autosave()
        return removed

    def reset(self) -> int:
        """Empty the whole cache (every user's private entries too) and preload the FAQ."""
        with self._lock:
            self.cache.clear()
            n = warm_cache(self.cache, self.settings, self.embedder)
            self.cache.save(self.settings.cache_dir)
            self._unsaved = 0
        return n

    def warm(self) -> dict:
        """Put back any FAQ answer that is missing from the shared cache, without touching
        the other entries (an entry with the same question text is left as it is)."""
        items = load_faq(self.settings.path(self.settings.paths.faq_csv))
        missing = [i for i in items if self.cache.find_exact(i.question) is None]
        if missing:
            vectors, rows, rho = self.embedder.encode_with_rho([i.question for i in missing],
                                                               self.settings.cache.min_known_ratio)
            with self._lock:
                self.cache.add_many([(i.question, i.answer) for i in missing], vectors, source="faq",
                                    tfidf_rows=rows, rhos=rho)
            self.save()
        return {"added": len(missing), "already_there": len(items) - len(missing), "entries": len(self.cache)}

    def reset_for(self, user: str, admin: bool = False) -> dict:
        """What "reset" means for one user: an admin resets the shared cache; anyone else
        removes their own private answers and zeroes their counters."""
        user = self._user(user)
        if admin:
            n = self.reset()
            return {"scope": "all", "entries": len(self.cache), "faq_entries": n, "removed": None}
        with self._lock:
            removed = self.cache.remove_owned_by(user)
        self.users.reset_stats(user)
        if removed:
            self.save()
        return {"scope": "private", "entries": len(self.cache), "faq_entries": None, "removed": removed}

    def stats(self, user: str | None = None) -> dict:
        with self._lock:
            out = self.cache.stats()
        if user is not None:
            out["user"] = self.users.profile(self._user(user)).stats()
        return out

    @property
    def pricing(self) -> Pricing:
        m = self.settings.metrics
        return Pricing(m.price_per_1k_input_tokens, m.price_per_1k_output_tokens,
                       m.cost_per_llm_call_usd, m.price_note)

    def metrics_report(self, bucket: str | int | None = None, user: str | None = None) -> dict:
        """Counters, latency, estimated cost and the time series, for everyone (``user=None``)
        or for one user."""
        seconds = parse_bucket(bucket if bucket is not None else self.settings.metrics.bucket_seconds)
        user = self._user(user)
        return {"profile": self.settings.profile, "enabled": self.metrics.enabled,
                "pricing": self.pricing.describe(),
                "summary": self.metrics.summary(self.pricing, user),
                "series": self.metrics.series(seconds, user)}

    def reset_demo(self, admin: bool = False) -> dict:
        """Back to a clean stage: exactly the FAQ answers in the cache (everything else is
        removed, private entries too), the metrics archived and started over, and every
        user's counters at zero. Conversations are kept."""
        if not admin:
            raise admin_required("resetting the demo")
        n = self.reset()
        metrics = self.metrics.reset()
        users = self.users.list_users()
        for name in users:
            self.users.reset_stats(name)
        return {"entries": len(self.cache), "faq_entries": n, "users_reset": len(users), **metrics}

    def reset_metrics(self, admin: bool = False) -> dict:
        if not admin:
            raise admin_required("resetting the metrics")
        return self.metrics.reset()

    def describe(self) -> dict:
        return {"embedder": self.embedder.describe(), "cache": self.stats(),
                "backend": self.backend.describe()}

    def save(self) -> None:
        with self._lock:
            self.cache.save(self.settings.cache_dir)
            self._unsaved = 0

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _user(user: str | None) -> str | None:
        return None if user is None else normalize_username(user)

    def _thread(self, user: str | None, thread_id: str) -> Thread:
        if user is None:
            raise ValueError("a conversation thread needs a user")
        return self.users.get_thread(self._user(user), thread_id)

    def _encode(self, text: str):
        """(latent vector, TF-IDF row, rho) for one question."""
        vectors, rows, rho = self.embedder.encode_with_rho([text], self.settings.cache.min_known_ratio)
        return vectors[0], rows[0], float(rho[0])

    def _validate(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        query = " ".join(query.split())
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")
        return query

    def _autosave(self) -> None:
        with self._lock:
            self._unsaved += 1
            every = self.settings.cache.autosave_every
            if every and self._unsaved >= every:
                self.save()
