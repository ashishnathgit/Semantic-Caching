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
import json
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
from .embedder import KnownWordPolicy, LSAEmbedder
from .explain import explain_query
from .metrics import MetricsStore, Pricing, parse_bucket
from .text import exact_key
from .route import route_trace
from .users import Message, Thread, UserStore, normalize_username

MAX_QUERY_CHARS = 1000
HIDDEN_QUESTION = "another user's question"


def _words(words: list[str]) -> str:
    return ", ".join(f'"{w}"' for w in dict.fromkeys(words))


def blocked_note(adm, ratio: float, policy) -> str:
    """Why the known-word guard sent a question to the model, naming the words it didn't know."""
    note = (f"Only {adm.known} of {adm.total} words are known to the model (at least {ratio:.0%} are needed), "
            "so a meaning-based match isn't trusted; only an exact repeat can hit.")
    if policy is not None and adm.topic_unknown:
        note += (f" Topic words it doesn't know: {_words(adm.topic_unknown)}"
                 + (f"; everyday words like {_words(adm.everyday_unknown)} were not counted against it" if
                    adm.everyday_unknown else "")
                 + f". A second chance needs {policy.min_topic_words} known topic words and "
                   f"{policy.min_topic_ratio:.0%} of them known; it had {len(adm.topic_known)} of "
                   f"{len(adm.topic_known) + len(adm.topic_unknown)}.")
    elif adm.topic_unknown:
        note += f" Words it doesn't know: {_words(adm.topic_unknown)}."
    return note


def relaxed_note(adm) -> str:
    return (f"Only {adm.known} of {adm.total} words are known, but {_words(adm.everyday_unknown)} are everyday "
            f"words that say nothing about the topic, so they weren't counted. Matched on its known topic words "
            f"({_words(adm.topic_known)}) alone"
            + (f"; it doesn't know {_words(adm.topic_unknown)}." if adm.topic_unknown else "."))


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
    route: dict | None = None           # exact? -> guards -> score vs tau -> angle -> outcome (route.py)

    def to_dict(self) -> dict:
        return asdict(self)


PLACEHOLDER_SOURCES = {"stub"}     # placeholder answers: their text would only teach "placeholder", "ollama"


def cached_documents(settings: Settings, entries: list[CacheEntry] | None) -> list[str]:
    """With ``corpus.include_cached``: one training text per shared cached entry, its question
    joined with its answer (placeholder answers add only their question). The FAQ answers the
    cache is warmed with count too, also before the cache exists. Private entries are left out:
    their words would show up in the shared model's directions."""
    if not settings.corpus.include_cached:
        return []
    docs: dict[str, str] = {}
    for e in entries or []:
        if e.owner is not None:
            continue
        docs[exact_key(e.query)] = e.query if e.source in PLACEHOLDER_SOURCES else f"{e.query} {e.answer}"
    if settings.cache.warm_with_faq:
        for item in load_faq(settings.path(settings.paths.faq_csv)):
            docs.setdefault(exact_key(item.question), f"{item.question} {item.answer}")
    return list(docs.values())


def build_corpus(settings: Settings, verbose: bool = False,
                 entries: list[CacheEntry] | None = None) -> tuple[list[str], dict]:
    """Texts the SVD is fitted on: FAQ questions + paraphrases, plus Quora questions if present,
    plus (``corpus.include_cached``) every shared cached question joined with its answer."""
    corpus: list[str] = []
    info = {"faq_texts": 0, "quora_questions": 0, "quora_file": None, "cached_texts": 0}
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
    cached = cached_documents(settings, entries)
    corpus += cached
    info["cached_texts"] = len(cached)
    if not corpus:
        raise ValueError("the corpus is empty: enable use_faq or add the Quora data")
    return corpus, info


def save_corpus_snapshot(settings: Settings, corpus: list[str]) -> None:
    """The exact texts a growing model was fitted on (the Matrix Lab rebuilds X from them)."""
    if settings.corpus.include_cached:
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        path = settings.model_dir / "corpus.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"texts": corpus}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def cache_params(settings: Settings) -> dict:
    c = settings.cache
    return {"threshold": c.threshold, "max_entries": c.max_entries, "use_lsh": c.use_lsh,
            "lsh_bits": c.lsh_bits, "lsh_tables": c.lsh_tables, "lsh_min_entries": c.lsh_min_entries,
            "lsa_weight": c.lsa_weight, "min_latent_coverage": c.min_latent_coverage,
            "negation_guard": c.negation_guard, "antonyms": list(c.antonyms), "ttl_seconds": c.ttl_seconds,
            "per_entry_thresholds": c.per_entry_thresholds, "entry_precision_target": c.entry_precision_target}


def configure_embedder(embedder: LSAEmbedder, settings: Settings) -> LSAEmbedder:
    """Apply the query-time settings of [cache] to a loaded or freshly fitted embedder: the
    known-word guard's topic-word rule (KnownWordPolicy)."""
    c = settings.cache
    embedder.policy = (KnownWordPolicy(int(c.min_topic_words), float(c.min_topic_ratio))
                       if c.relaxed_known_words else None)
    embedder._everyday_keep = None
    return embedder


def load_embedder(settings: Settings) -> LSAEmbedder:
    return configure_embedder(LSAEmbedder.load(settings.model_dir), settings)


def new_user_store(settings: Settings) -> UserStore:
    u = settings.users
    return UserStore(settings.users_dir, u.max_threads_per_user, u.max_messages_per_thread)


def new_cache(settings: Settings, embedder: LSAEmbedder) -> SemanticCache:
    return SemanticCache(embedder.dim, model_fingerprint=embedder.fingerprint,
                         query_policy=embedder.policy_signature, **cache_params(settings))


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
    existing = None
    if settings.corpus.include_cached and not fresh and SemanticCache.exists(settings.cache_dir):
        existing = SemanticCache.load(settings.cache_dir).entries(newest_first=False)
    corpus, info = build_corpus(settings, verbose, existing)
    if verbose:
        print(f"  corpus: {len(corpus):,} texts (FAQ {info['faq_texts']:,}, "
              f"Quora {info['quora_questions']:,}"
              + (f", cached question + answer {info['cached_texts']:,}" if settings.corpus.include_cached else "")
              + ")")
    embedder = configure_embedder(LSAEmbedder(settings.embedder).fit(corpus, verbose=verbose), settings)
    for note in embedder.report.notes:
        if verbose:
            print(f"  note: {note}")
    embedder.save(settings.model_dir)
    save_corpus_snapshot(settings, corpus)
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
        # Live rebuild (corpus.include_cached): the model generation bumps on every swap, so a
        # vector computed outside the lock with an older model is recomputed inside it.
        self._model_gen = 0
        self._new_since_build = self._entries_after_build()
        self.rebuild_state: dict = {"state": "idle", "reason": None, "started_at": None, "finished_at": None,
                                    "seconds": None, "error": None, "generation": 0}
        self._rebuild_thread: threading.Thread | None = None

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
            embedder = load_embedder(settings)
            if SemanticCache.exists(settings.cache_dir):
                cache = SemanticCache.load(settings.cache_dir, embedder, settings.cache.min_known_ratio,
                                           **cache_params(settings))
            else:
                cache = new_cache(settings, embedder)
                warm_cache(cache, settings, embedder)
        return cls(embedder, cache, build_backend(settings.llm), settings)

    # ------------------------------------------------------------ live rebuild
    @property
    def grows(self) -> bool:
        """True when the model's corpus includes the cached answers (faq-live)."""
        return bool(self.settings.corpus.include_cached)

    def _entries_after_build(self) -> int:
        meta = self.settings.model_dir / "embedder.json"
        if not self.grows or not meta.exists():
            return 0
        built = meta.stat().st_mtime
        return sum(e.created_at > built and e.source != "faq" for e in self.cache.entries())

    def model_status(self) -> dict:
        every = self.settings.cache.rebuild_every
        return {**self.rebuild_state, "grows": self.grows, "rebuild_every": every if self.grows else 0,
                "new_since_build": self._new_since_build, "k": self.embedder.dim,
                "vocabulary_size": len(self.embedder.vocab), "documents": self.embedder.report.n_docs,
                "fingerprint": self.embedder.fingerprint}

    def request_rebuild(self, admin: bool = False, reason: str = "admin",
                        wait: bool = False) -> dict:
        """Refit TF-IDF + SVD on the FAQ texts plus every shared cached question + answer, in
        a background thread; re-embed all stored questions; then swap the model atomically.
        Only for profiles whose corpus includes the cache (it would change nothing elsewhere)."""
        if reason == "admin" and not admin:
            raise admin_required("rebuilding the model")
        if not self.grows:
            raise ValueError(f"profile {self.settings.profile!r} does not fit on cached answers "
                             "(corpus.include_cached = false), so a rebuild would give the same model")
        with self._lock:
            if self.rebuild_state["state"] == "running":
                raise RuntimeError("a rebuild is already running")
            self.rebuild_state = {**self.rebuild_state, "state": "running", "reason": reason,
                                  "started_at": time.time(), "finished_at": None, "seconds": None, "error": None}
            thread = threading.Thread(target=self._rebuild, name="model-rebuild", daemon=True)
            self._rebuild_thread = thread
        thread.start()
        if wait:
            thread.join()
        return self.model_status()

    def _rebuild(self) -> None:
        t0 = time.perf_counter()
        ratio = self.settings.cache.min_known_ratio
        try:
            with self._lock:
                snapshot = self.cache.entries(newest_first=False)
            corpus, _ = build_corpus(self.settings, entries=snapshot)
            new = configure_embedder(LSAEmbedder(self.settings.embedder).fit(corpus), self.settings)  # slow: no lock
            V, X, rho = new.encode_with_rho([e.query for e in snapshot], ratio)
            from .cache import as_sparse_row
            pre = {e.id: (V[i], as_sparse_row(X[i]), float(rho[i])) for i, e in enumerate(snapshot)}
            new.save(self.settings.model_dir)
            save_corpus_snapshot(self.settings, corpus)
            with self._lock:                                                # the swap: fast
                late = self.cache.reembed(new, ratio, precomputed=pre)
                self.embedder = new
                self._model_gen += 1
                self._new_since_build = late
                self.cache.save(self.settings.cache_dir)
                self._unsaved = 0
                self.rebuild_state.update(state="done", finished_at=time.time(),
                                          seconds=time.perf_counter() - t0, generation=self._model_gen,
                                          reembedded=len(pre) + late, documents=len(corpus))
        except Exception as exc:                                            # noqa: BLE001 - report it
            with self._lock:
                self.rebuild_state.update(state="failed", finished_at=time.time(), error=str(exc),
                                          seconds=time.perf_counter() - t0)

    def _note_new_entry(self) -> None:
        """Count a new cached entry; start a background rebuild every ``rebuild_every`` of them."""
        if not self.grows:
            return
        with self._lock:
            self._new_since_build += 1
            due = (self.settings.cache.rebuild_every > 0
                   and self._new_since_build >= self.settings.cache.rebuild_every
                   and self.rebuild_state["state"] != "running")
        if due:
            try:
                self.request_rebuild(reason="auto")
            except RuntimeError:
                pass

    def _current(self, text: str, enc: tuple, gen: int) -> tuple:
        """``enc`` = (vector, row, rho) computed with model generation ``gen``; recomputed if the
        model was swapped since. Call with the lock held."""
        return enc if gen == self._model_gen else self._encode(text)

    @property
    def lab(self):
        """The Matrix Lab (semcache/lab.py): checks and worked numbers; never changes the model."""
        if getattr(self, "_lab", None) is None:
            from .lab import Lab
            self._lab = Lab(self)
        return self._lab

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
        gen = self._model_gen
        enc = self._encode(query)
        with self._lock:
            vector, row, rho = self._current(query, enc, gen)
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

    def route_for(self, lookup: LookupResult, known: int, total: int, *, cache_used: bool = True,
                  follow: FollowUp = STANDALONE, refreshed: bool = False, use_llm: bool = True,
                  admission=None) -> dict:
        """The step-by-step route of one lookup (see semcache/route.py)."""
        c = self.settings.cache
        return route_trace(lookup, cache_used=cache_used, follow_reason=follow.reason, refreshed=refreshed,
                           known=known, total=total, min_known_ratio=c.min_known_ratio,
                           min_latent_coverage=self.cache.min_latent_coverage,
                           text_guard_on=bool(getattr(c, "negation_guard", False)), use_llm=use_llm,
                           admission=admission)

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
        model_gen = self._model_gen
        enc = self._encode(query)
        t1 = time.perf_counter()
        with self._lock:
            vector, row, rho = enc = self._current(query, enc, model_gen)
            model_gen = self._model_gen
            # A follow-up or refresh is still scored (so the UI can show the neighbours)
            # but not recorded: it never counts as a hit or a miss.
            lookup = self.cache.lookup(vector, top_k or self.settings.cache.top_k, threshold,
                                       record=cache_used, query_text=query, query_tfidf=row, user=user,
                                       query_rho=rho)
        t2 = time.perf_counter()
        adm = self.embedder.admission(query, ratio)
        known, total = adm.known, adm.total
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
        elif lookup.refused:
            result.note = (f"Negation guard ({lookup.refused['rule']}): {lookup.refused['detail']}. The closest "
                           f"cached question scored {lookup.refused['score']:.3f}, but its answer would be wrong here, "
                           "so the question goes to the model.")
        elif total == 0 or not result.known_terms:
            result.note = "None of these words are in the model's vocabulary, so only an exact repeat can hit."
        elif adm.status == "blocked":
            result.note = blocked_note(adm, ratio, self.embedder.policy)
        elif adm.status == "relaxed":
            result.note = relaxed_note(adm)
        if total and result.known_terms:
            result.note = " ".join(x for x in (result.note, self.rho_note(lookup)) if x)
        result.route = self.route_for(lookup, known, total, cache_used=cache_used, follow=follow,
                                      refreshed=refresh, use_llm=use_llm, admission=adm)

        llm_ms, gen = 0.0, None
        if hit:
            result.answer = lookup.best.answer
            result.answer_source = "CACHE"
            if self.cache.per_entry_thresholds and lookup.method != "exact":
                with self._lock:                  # an unreported hit counts as correct
                    self.cache.observe(lookup.best.id, lookup.score, True)
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
                result.stored_id, outcome = self._store_refresh(query, gen, vector, row, rho, user,
                                                                model_gen=model_gen)
                result.note = " ".join(x for x in (result.note, outcome) if x)
            elif gen.cacheable and cache_used:
                result.stored_id = self._store_shared(query, gen, vector, row, rho, user, model_gen=model_gen)
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
                      asked_by: str | None = None, model_gen: int | None = None) -> int:
        # The lock is released while the LLM runs, so a concurrent request for the same
        # question may have stored an answer meanwhile (the lookup would then have hit an
        # exact repeat). Keep that first answer instead of replacing it. The model may also
        # have been rebuilt meanwhile; then the vector is recomputed with the new one.
        added = False
        with self._lock:
            if model_gen is not None:
                vector, row, rho = self._current(query, (vector, row, rho), model_gen)
            entry = self.cache.find_exact(query)
            if entry is None:
                entry = self.cache.add(query, gen.text, vector, source=gen.backend, model=gen.model,
                                       tfidf_row=row, rho=float(rho), asked_by=asked_by,
                                       prompt_tokens=gen.prompt_tokens, output_tokens=gen.output_tokens)
                self._autosave()
                added = True
        if added:
            self._note_new_entry()
        return entry.id

    def _store_refresh(self, query: str, gen, vector, row, rho: float,
                       user: str | None, model_gen: int | None = None) -> tuple[int | None, str]:
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
            if model_gen is not None:
                vector, row, rho = self._current(query, (vector, row, rho), model_gen)
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
        self._note_new_entry()
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
                  shared: bool = False, admin: bool = False, ttl_seconds: float | None = None) -> CacheEntry:
        """Add an answer by hand. With a ``user`` it is private to them, unless ``shared``
        is set by an admin. Without a user (CLI, scripts) it is shared."""
        question = self._validate(question)
        user = self._user(user)
        if not answer or not answer.strip():
            raise ValueError("answer must not be empty")
        if user is not None and shared and not admin:
            raise admin_required("adding an answer shared with everyone")
        owner = None if user is None or shared else user
        gen = self._model_gen
        enc = self._encode(question)
        with self._lock:
            vector, row, rho = self._current(question, enc, gen)
            entry = self.cache.add(question, answer.strip(), vector, source=source, tfidf_row=row,
                                   owner=owner, rho=float(rho), ttl_seconds=ttl_seconds)
        self._autosave()
        if owner is None:
            self._note_new_entry()
        return entry

    # ------------------------------------------------------------ data and admin
    LEARNED_KEEP = frozenset({"faq", "manual"})      # sources that are not LLM output

    def forget_llm_rows(self, admin: bool = False) -> dict:
        """Admin: remove every entry an LLM backend produced (Ollama, an online model or the
        placeholder), keeping the FAQ and answers added by hand."""
        if not admin:
            raise admin_required("forgetting the LLM-learned answers")
        with self._lock:
            removed = self.cache.remove_where(lambda e: e.source not in self.LEARNED_KEEP)
            self.cache.save(self.settings.cache_dir)
            self._unsaved = 0
        return {"removed": removed, "entries": len(self.cache)}

    def import_csv(self, name: str, text: str, admin: bool = False) -> dict:
        """Admin: build a NEW profile from an FAQ CSV (see semcache/importer.py). The running
        profile, and faq, are never touched."""
        if not admin:
            raise admin_required("importing an FAQ CSV")
        from .importer import import_faq_csv
        if name.strip().lower() == self.settings.profile:
            raise ValueError("importing into the profile that is being served is not allowed; pick a new name")
        return import_faq_csv(self.settings, name, text)

    @property
    def feedback_path(self):
        return self.settings.artifacts_dir / "feedback" / "wrong_answers.jsonl"

    def report_wrong(self, query: str, entry_id: int, user: str | None = None, admin: bool = False) -> dict:
        """"Wrong answer" on a hit: log (question, cached question, score, tau) for threshold tuning,
        and evict the entry. FAQ answers and shared answers added by hand are evicted only by an
        admin (anyone else's report is logged and the answer stays); private entries only by
        their owner."""
        query = self._validate(query)
        user = self._user(user)
        gen = self._model_gen
        enc = self._encode(query)
        with self._lock:
            entry = self.cache.get(entry_id)
            if entry is None or (entry.owner is not None and entry.owner != user and not admin):
                raise LookupError(f"no entry with id {entry_id}")
            vector, row, rho = self._current(query, enc, gen)
            every = self.cache.score_all(vector, row, rho, user)
            score = next((float(sc) for e, sc in zip(every["entries"], every["scores"]) if e.id == entry_id), None)
            protected = (entry.source == "faq" or (entry.source == "manual" and entry.owner is None)) and not admin
            # Experimental per-entry thresholds: a semantic hit tightens the entry's own tau instead
            # of evicting it (an exact repeat that is wrong is still evicted: no tau can fix it).
            exact = self.cache.find_exact(query, user) is not None and self.cache.find_exact(query, user).id == entry_id
            learn = self.cache.per_entry_thresholds and score is not None and not exact
            new_tau = self.cache.observe(entry_id, score, False, replace_positive=True) if learn else None
            evict = not protected and not learn
            record = {"t": time.time(), "user": user, "query": query, "entry_id": entry_id,
                      "cached_query": entry.query, "source": entry.source,
                      "score": None if score is None else round(score, 4), "tau": self.cache.threshold,
                      "evicted": evict, "entry_threshold": new_tau, "model": self.embedder.fingerprint}
            if learn:
                self.cache.save(self.settings.cache_dir)
            if evict:
                self.cache.remove(entry_id)
                self.cache.save(self.settings.cache_dir)
                self._unsaved = 0
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.feedback_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        note = (f"Thanks: this answer now needs a score of at least {new_tau or self.cache.threshold:.2f} "
                "to be served (per-entry threshold); the pair is logged." if learn else
                "Thanks: the answer was removed from the cache and the pair logged for threshold tuning."
                if evict else
                "Thanks: logged for threshold tuning. FAQ and admin answers can only be removed by an admin.")
        return {**record, "note": note}

    def remove_entries(self, entry_ids: list[int], admin: bool = False) -> dict:
        """Admin: delete several cached entries at once (any owner, FAQ included), saving once."""
        if not admin:
            raise admin_required("deleting cached questions")
        ids = list(dict.fromkeys(int(i) for i in entry_ids))
        with self._lock:
            removed = [i for i in ids if self.cache.remove(i)]
            if removed:
                self.cache.save(self.settings.cache_dir)
                self._unsaved = 0
        return {"removed": len(removed), "missing": [i for i in ids if i not in set(removed)],
                "entries": len(self.cache)}

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
            with self._lock:
                vectors, rows, rho = self.embedder.encode_with_rho([i.question for i in missing],
                                                                   self.settings.cache.min_known_ratio)
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
