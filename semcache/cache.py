"""The semantic cache: stores (query, answer, vector) and decides hit or miss.

Vectors live in one contiguous matrix C (one unit-length row per cached query),
so a lookup is a single matrix-vector product (linalg.cosine_scores):

    scores = C q,     hit  <=>  max(scores) >= tau

Five refinements, each optional:

* Exact layer. A dictionary keyed by the normalised query text answers exact
  repeats instantly, even when their words are unknown to the model (such
  entries get a zero vector, which can never produce a semantic match).
* Hybrid score. With ``lsa_weight < 1`` every entry is scored as
  w * cos_LSA + (1 - w) * cos_TFIDF (linalg.blend_scores), using its sparse TF-IDF row.
* Latent-coverage guard. rho = ||q V_k|| / ||q|| measures how much of a question
  survives in the k-dimensional space. When the smaller rho of a (question, cached
  question) pair is below ``min_latent_coverage``, their latent cosine means little
  (both may have collapsed onto "what" or "how"), so that pair is scored by word
  overlap alone, the TF-IDF cosine (linalg.guarded_scores).
* Negation guard. With ``negation_guard=True`` a semantic hit is refused when the
  question and the cached question differ in negation, an antonym, a number or a name
  (semcache/guards.py); the lookup becomes a miss and ``refused`` names the rule.
* LSH. With ``use_lsh=True`` only rows sharing an LSH bucket with q are scored
  (linalg.RandomHyperplaneLSH), which keeps lookups fast for large caches. The LSH
  tables are keyed by row position in C (not entry id), so candidates index C directly.

Several users share one cache. An entry with ``owner=None`` is shared (FAQ
answers and LLM answers to standalone questions); an entry with an owner is
private to that user (answers they added by hand). A lookup made for a user sees
the shared entries plus that user's own; ``user=None`` sees only the shared ones.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import scipy.sparse as sp

from . import linalg as la
from .text import exact_key

if TYPE_CHECKING:  # pragma: no cover
    from .embedder import LSAEmbedder

SparseRow = tuple[np.ndarray, np.ndarray]          # (sorted term indices, weights)
_EMPTY_ROW: SparseRow = (np.empty(0, dtype=np.int32), np.empty(0))


def as_sparse_row(row) -> SparseRow:
    """Accept a 1 x V scipy row, an (indices, values) pair or None."""
    if row is None:
        return _EMPTY_ROW
    if sp.issparse(row):
        r = sp.csr_matrix(row)
        r.sort_indices()
        return r.indices.astype(np.int32), r.data.astype(np.float64)
    idx, val = row
    order = np.argsort(idx)
    return np.asarray(idx, dtype=np.int32)[order], np.asarray(val, dtype=np.float64)[order]


@dataclass
class CacheEntry:
    id: int
    query: str
    answer: str
    source: str                      # "faq", "ollama", "stub" or "manual"
    created_at: float
    last_used_at: float
    hits: int = 0
    model: str | None = None         # which LLM produced the answer, if any
    owner: str | None = None         # None = shared with everyone, else private to this user
    asked_by: str | None = None      # who asked the question a shared LLM answer was made for
    prompt_tokens: int | None = None # tokens the LLM call that made the answer used (if reported)
    output_tokens: int | None = None
    expires_at: float | None = None  # unix time after which the entry is a miss (None = never)
    threshold: float | None = None   # own tau (per_entry_thresholds; None = the global tau)
    feedback: list | None = None     # recent [score, correct 0/1] observations behind ``threshold``

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Match:
    entry: CacheEntry
    score: float                     # the score the hit decision uses
    lsa: float | None = None         # its parts (tfidf only when it was computed)
    tfidf: float | None = None
    rho: float = 1.0                 # latent coverage of the cached question
    pair_rho: float = 1.0            # min(rho of the question, rho of the cached question)
    scored_by: str = "svd"           # "svd", "hybrid", "tfidf" (blocked by the guard) or "exact"

    @property
    def blocked(self) -> bool:
        return self.scored_by == "tfidf"

    def to_dict(self) -> dict:
        d = {"id": self.entry.id, "query": self.entry.query, "score": round(self.score, 4),
             "source": self.entry.source, "owner": self.entry.owner,
             "rho": round(self.rho, 4), "pair_rho": round(self.pair_rho, 4),
             "scored_by": self.scored_by}
        if self.lsa is not None:
            d["lsa"] = round(self.lsa, 4)
        if self.tfidf is not None:
            d["tfidf"] = round(self.tfidf, 4)
        return d


@dataclass
class LookupResult:
    hit: bool
    score: float
    threshold: float
    best: CacheEntry | None
    matches: list[Match] = field(default_factory=list)
    compared: int = 0                # how many cached vectors were scored
    method: str = "brute"            # "exact", "brute", "lsh" or "empty"
    rho: float = 1.0                 # latent coverage of the question
    refused: dict | None = None      # the text guard refused the best match: {rule, detail, ...}
    threshold_used: float | None = None  # the best entry's own tau when per-entry thresholds raised it


ALL_USERS = object()
"""Pass as ``user`` to ``entries()`` to list every entry, private ones included."""


class SemanticCache:
    def __init__(self, dim: int, threshold: float = 0.8, max_entries: int = 5000,
                 use_lsh: bool = False, lsh_bits: int = 10, lsh_tables: int = 24,
                 lsh_min_entries: int = 50000, seed: int = 0, model_fingerprint: str = "",
                 lsa_weight: float = 1.0, min_latent_coverage: float = 0.0,
                 negation_guard: bool = False, antonyms: list[str] | None = None,
                 ttl_seconds: float = 0.0, per_entry_thresholds: bool = False,
                 entry_precision_target: float = 0.95, query_policy: str = ""):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if not 0.0 <= lsa_weight <= 1.0:
            raise ValueError("lsa_weight must be in [0, 1]")
        self.dim = dim
        self.threshold = threshold
        self.max_entries = max_entries
        self.model_fingerprint = model_fingerprint
        self.query_policy = query_policy        # the embedder's known-word policy the vectors were made under
        self.lsa_weight = lsa_weight
        self.min_latent_coverage = float(min_latent_coverage)
        self.lsh_settings = {"n_bits": lsh_bits, "n_tables": lsh_tables, "seed": seed}
        self.lsh_min_entries = lsh_min_entries
        self.ttl_seconds = float(ttl_seconds or 0.0)      # default lifetime of new entries; 0 = forever
        self.per_entry_thresholds = bool(per_entry_thresholds)   # experimental (semcache/thresholds.py)
        self.entry_precision_target = float(entry_precision_target)
        self.lsh = la.RandomHyperplaneLSH(dim, lsh_bits, lsh_tables, seed) if use_lsh else None
        # Text rules checked before a semantic hit is served (semcache/guards.py); None = off.
        self.text_guard = None
        if negation_guard:
            from .guards import TextGuard
            self.text_guard = TextGuard(antonyms or [])

        capacity = 64
        self._vectors = np.zeros((capacity, dim))
        self._last_used = np.zeros(capacity)
        self._rho = np.ones(capacity)              # latent coverage of each cached question
        self._expires = np.full(capacity, np.inf)  # expiry time of each row (inf = never)
        self._n_expiring = 0                       # rows with a finite expiry; 0 = no mask needed
        self._tfidf_matrix: sp.csr_matrix | None = None   # rows of _tfidf, built on demand
        self._owner_codes = np.zeros(capacity, dtype=np.int32)   # 0 = shared, else _owner_code[owner]
        self._owner_code: dict[str, int] = {}
        self._n_private = 0
        self._entries: list[CacheEntry] = []
        self._tfidf: list[SparseRow] = []          # slot-aligned sparse TF-IDF rows
        self._slot_of: dict[int, int] = {}
        self._exact: dict[tuple[str | None, str], int] = {}   # (owner, exact_key(query)) -> entry id
        self._next_id = 1
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def vectors(self) -> np.ndarray:
        """The cache matrix C, one row per entry (read-only view)."""
        view = self._vectors[: len(self)]
        view.flags.writeable = False
        return view

    def rho_of(self, entry_id: int) -> float:
        slot = self._slot_of.get(entry_id)
        return 1.0 if slot is None else float(self._rho[slot])

    def get(self, entry_id: int) -> CacheEntry | None:
        slot = self._slot_of.get(entry_id)
        return None if slot is None else self._entries[slot]

    def find_exact(self, query: str, user: str | None = None, include_expired: bool = False) -> CacheEntry | None:
        """The entry ``user`` would get for exactly this question (case, spacing and
        punctuation ignored): their own private entry first, else the shared one.
        Expired entries are skipped unless ``include_expired``."""
        key = exact_key(query)
        now = time.time()
        for owner in ((user, None) if user is not None else (None,)):
            entry_id = self._exact.get((owner, key))
            entry = None if entry_id is None else self.get(entry_id)
            if entry is not None and (include_expired or not self.expired(entry, now)):
                return entry
        return None

    @staticmethod
    def expired(entry: CacheEntry, now: float | None = None) -> bool:
        return entry.expires_at is not None and entry.expires_at <= (time.time() if now is None else now)

    def entries(self, newest_first: bool = True, user=ALL_USERS) -> list[CacheEntry]:
        """Entries sorted by creation time. ``user`` limits them to what that user can see
        (``None`` = shared entries only); by default every entry is listed."""
        items = self._entries if user is ALL_USERS else [e for e in self._entries if self._visible(e, user)]
        return sorted(items, key=lambda e: e.created_at, reverse=newest_first)

    @staticmethod
    def _visible(entry: CacheEntry, user: str | None) -> bool:
        return entry.owner is None or entry.owner == user

    def _visible_mask(self, user: str | None) -> np.ndarray | None:
        """Boolean mask over the slots ``user`` may see (and that have not expired), or None
        when every slot counts. O(n) only when there are private or expiring entries."""
        mask = None
        if self._n_private:
            codes = self._owner_codes[: len(self)]
            code = self._owner_code.get(user) if user is not None else None
            mask = (codes == 0) if code is None else ((codes == 0) | (codes == code))
        if self._n_expiring:
            alive = self._expires[: len(self)] > time.time()
            mask = alive if mask is None else (mask & alive)
        return mask

    # ------------------------------------------------------------------ lookup
    def lookup(self, vector: np.ndarray, top_k: int = 5, threshold: float | None = None,
               record: bool = True, query_text: str | None = None,
               query_tfidf=None, user: str | None = None, query_rho: float = 1.0) -> LookupResult:
        """Exact match on ``query_text`` first, then similarity against C.

        Pass ``query_tfidf`` (the query's TF-IDF row) for hybrid scoring and for the
        word-overlap score of pairs the latent-coverage guard blocks, and ``query_rho``
        (its latent coverage) for the guard. Only entries visible to ``user`` count.
        """
        tau = self.threshold if threshold is None else float(threshold)
        if not 0.0 < tau <= 1.0:              # tau <= 0 would let every query hit
            raise ValueError(f"threshold must be in (0, 1], got {tau}")
        n = len(self)
        q = la.l2_normalize_rows(np.asarray(vector, dtype=np.float64))

        matches: list[Match] = []
        compared, method = 0, "empty" if n == 0 else "brute"
        query_rho = float(query_rho)
        # A question the guard blocks is scored by word overlap, which LSH can't search.
        blocked_query = query_rho < self.min_latent_coverage and query_tfidf is not None
        if n and np.any(q):
            if self.lsh is not None and n >= self.lsh_min_entries and not blocked_query:
                # The LSH tables store row positions of C, so no per-candidate id lookup.
                slots = self.lsh.candidates(q).astype(np.intp)
                method = "lsh"
            else:
                slots = np.arange(n)
            visible = self._visible_mask(user)
            if visible is not None and method == "lsh":
                slots = slots[visible[slots]]
                visible = None
            if slots.size:
                # Brute force scores a slice of C (a view); fancy-indexing all n rows
                # would copy the whole matrix on every lookup. Rows other users own are
                # masked out of the scores instead of being cut out of the matrix.
                rows = self._vectors[:n] if method == "brute" else self._vectors[slots]
                matches = self._score(slots, rows, q, top_k, query_tfidf, visible, query_rho)
            compared = int(slots.size if visible is None else visible.sum())

        refused = threshold_used = None
        exact_entry = self.find_exact(query_text, user) if query_text is not None else None
        if exact_entry is not None:
            exact_match = Match(exact_entry, 1.0, rho=self.rho_of(exact_entry.id),
                                pair_rho=min(query_rho, self.rho_of(exact_entry.id)), scored_by="exact")
            matches = [exact_match] + [m for m in matches if m.entry.id != exact_entry.id]
            matches = matches[:top_k]
            best, hit, method = matches[0], True, "exact"
        elif matches:
            best = matches[0]
            own = best.entry.threshold if self.per_entry_thresholds else None
            if own is not None and own > tau:
                threshold_used = own
            hit = best.score >= max(tau, own or 0.0)
            if hit and self.text_guard is not None and query_text is not None:
                refused = self.text_guard.check(query_text, best.entry.query)
                if refused is not None:
                    hit = False
                    refused = {**refused, "entry_id": best.entry.id, "score": round(best.score, 4)}
        else:
            if record:
                self.misses += 1
            return LookupResult(False, 0.0, tau, None, [], compared, method, query_rho)

        if record:
            if hit:
                self.hits += 1
                best.entry.hits += 1
                best.entry.last_used_at = time.time()
                self._last_used[self._slot_of[best.entry.id]] = best.entry.last_used_at
            else:
                self.misses += 1
        return LookupResult(hit, best.score, tau, best.entry if hit else None, matches,
                            compared, method, query_rho, refused, threshold_used)

    def _score(self, slots: np.ndarray, rows: np.ndarray, q: np.ndarray, top_k: int,
               query_tfidf, visible: np.ndarray | None = None, query_rho: float = 1.0) -> list[Match]:
        """Score ``rows`` (the vectors stored in ``slots``) against q, skipping rows
        where ``visible`` is False.

            lsa_i     = c_i . q                    (one product C q)
            tfidf_i   = x_i . x_q                  (one sparse product X_C x_q^T)
            score_i   = tfidf_i                    if min(rho_q, rho_i) < min_latent_coverage
                        w lsa_i + (1 - w) tfidf_i  otherwise
        """
        lsa, lexical, scores, blocked = self._score_arrays(slots, rows, q, query_tfidf, query_rho)
        pair_rho = np.minimum(query_rho, self._rho[slots])
        n_visible = slots.size
        if visible is not None:
            scores = np.where(visible, scores, -np.inf)
            n_visible = int(visible.sum())
        mode = "svd" if self.lsa_weight >= 1.0 or lexical is None else "hybrid"
        return [Match(self._entries[slots[i]], float(scores[i]), float(lsa[i]),
                      None if lexical is None else float(lexical[i]),
                      rho=float(self._rho[slots[i]]), pair_rho=float(pair_rho[i]),
                      scored_by="tfidf" if blocked[i] else mode)
                for i in la.top_k(scores, min(top_k, n_visible))]

    def _score_arrays(self, slots: np.ndarray, rows: np.ndarray, q: np.ndarray, query_tfidf,
                      query_rho: float):
        """(lsa, lexical or None, decision scores, blocked) for the rows in ``slots``."""
        lsa = la.cosine_scores(rows, q)
        pair_rho = np.minimum(query_rho, self._rho[slots])
        guard = self.min_latent_coverage > 0 and bool(np.any(pair_rho < self.min_latent_coverage))
        need_lexical = query_tfidf is not None and (self.lsa_weight < 1.0 or guard)
        if not need_lexical:
            return lsa, None, lsa, np.zeros(slots.size, dtype=bool)
        q_idx, q_val = as_sparse_row(query_tfidf)
        q_row = sp.csr_matrix((q_val, q_idx, [0, q_idx.size]),
                              shape=(1, int(q_idx.max()) + 1 if q_idx.size else 1))
        lexical = la.sparse_row_scores(self._tfidf_rows()[slots], q_row)
        scores, blocked = la.guarded_scores(lsa, lexical, pair_rho, self.lsa_weight, self.min_latent_coverage)
        return lsa, lexical, scores, blocked

    def score_all(self, vector: np.ndarray, query_tfidf=None, query_rho: float = 1.0,
                  user: str | None = None) -> dict:
        """Every visible row of C scored against the question, in slot (row) order: what
        the math view draws as s = C e. Keys: slots, entries, lsa, scores, blocked."""
        n = len(self)
        q = la.l2_normalize_rows(np.asarray(vector, dtype=np.float64))
        slots = np.arange(n)
        visible = self._visible_mask(user)
        if visible is not None:
            slots = slots[visible]
        if not slots.size or not np.any(q):
            empty = np.zeros(slots.size)
            return {"slots": slots, "entries": [self._entries[i] for i in slots], "lsa": empty,
                    "scores": empty, "blocked": np.zeros(slots.size, dtype=bool)}
        lsa, _, scores, blocked = self._score_arrays(slots, self._vectors[slots], q, query_tfidf,
                                                     float(query_rho))
        return {"slots": slots, "entries": [self._entries[i] for i in slots], "lsa": lsa,
                "scores": scores, "blocked": blocked}

    def vector_of(self, entry_id: int) -> np.ndarray | None:
        slot = self._slot_of.get(entry_id)
        return None if slot is None else self._vectors[slot].copy()

    def _tfidf_rows(self) -> sp.csr_matrix:
        """The cached TF-IDF rows as one sparse matrix (rebuilt after the cache changes)."""
        if self._tfidf_matrix is None or self._tfidf_matrix.shape[0] != len(self):
            n = len(self)
            lengths = np.array([i.size for i, _ in self._tfidf], dtype=np.int64)
            indptr = np.concatenate([[0], np.cumsum(lengths)]) if n else np.zeros(1, np.int64)
            indices = np.concatenate([i for i, _ in self._tfidf]) if n else np.empty(0, np.int32)
            data = np.concatenate([v for _, v in self._tfidf]) if n else np.empty(0)
            width = int(indices.max()) + 1 if indices.size else 1
            self._tfidf_matrix = sp.csr_matrix((data, indices, indptr), shape=(n, width))
        return self._tfidf_matrix

    # -------------------------------------------------------------- insertion
    def add(self, query: str, answer: str, vector: np.ndarray, source: str = "manual",
            model: str | None = None, tfidf_row=None, owner: str | None = None,
            rho: float = 1.0, asked_by: str | None = None, prompt_tokens: int | None = None,
            output_tokens: int | None = None, ttl_seconds: float | None = None) -> CacheEntry:
        """Store an answer. ``owner=None`` shares it with everyone; otherwise only that
        user's lookups can see it. ``rho`` is the question's latent coverage. ``ttl_seconds``
        (default: the cache's ``ttl_seconds``; 0 = never) makes it expire: then it is a miss."""
        vector = la.l2_normalize_rows(np.asarray(vector, dtype=np.float64))
        if vector.shape != (self.dim,):
            raise ValueError(f"vector must have shape ({self.dim},), got {vector.shape}")
        key = exact_key(query)
        if (owner, key) in self._exact:           # same question again: replace the old entry
            self.remove(self._exact[(owner, key)])
        while len(self) >= self.max_entries:
            self._evict_lru()
        if len(self) == self._vectors.shape[0]:
            self._grow()

        now = time.time()
        ttl = self.ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        entry = CacheEntry(self._next_id, query, answer, source, now, now, model=model, owner=owner,
                           asked_by=asked_by, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                           expires_at=now + ttl if ttl > 0 else None)
        self._next_id += 1
        slot = len(self)
        self._vectors[slot] = vector
        self._last_used[slot] = now
        self._rho[slot] = float(rho)
        self._set_expiry(slot, entry.expires_at)
        self._tfidf_matrix = None
        self._set_owner(slot, owner)
        self._entries.append(entry)
        self._tfidf.append(as_sparse_row(tfidf_row) if np.any(vector) else _EMPTY_ROW)
        self._slot_of[entry.id] = slot
        if key:
            self._exact[(owner, key)] = entry.id
        if self.lsh is not None:
            self.lsh.add([slot], vector[None, :])
        return entry

    def add_many(self, items: Iterable[tuple[str, str]], vectors: np.ndarray,
                 source: str = "faq", tfidf_rows=None, rhos=None) -> list[CacheEntry]:
        rows = [None] * len(vectors) if tfidf_rows is None else [tfidf_rows[i] for i in range(len(vectors))]
        rhos = np.ones(len(vectors)) if rhos is None else rhos
        return [self.add(q, a, v, source, tfidf_row=r, rho=float(p))
                for (q, a), v, r, p in zip(items, vectors, rows, rhos)]

    def _set_expiry(self, slot: int, expires_at: float | None) -> None:
        self._expires[slot] = np.inf if expires_at is None else float(expires_at)
        self._n_expiring += expires_at is not None

    def _set_owner(self, slot: int, owner: str | None) -> None:
        if owner is None:
            self._owner_codes[slot] = 0
            return
        code = self._owner_code.setdefault(owner, len(self._owner_code) + 1)
        self._owner_codes[slot] = code
        self._n_private += 1

    def _grow(self) -> None:
        capacity = self._vectors.shape[0] * 2
        vectors = np.zeros((capacity, self.dim))
        vectors[: len(self)] = self._vectors[: len(self)]
        last_used = np.zeros(capacity)
        last_used[: len(self)] = self._last_used[: len(self)]
        owner_codes = np.zeros(capacity, dtype=np.int32)
        owner_codes[: len(self)] = self._owner_codes[: len(self)]
        rho = np.ones(capacity)
        rho[: len(self)] = self._rho[: len(self)]
        expires = np.full(capacity, np.inf)
        expires[: len(self)] = self._expires[: len(self)]
        self._vectors, self._last_used, self._owner_codes, self._rho = vectors, last_used, owner_codes, rho
        self._expires = expires

    # ---------------------------------------------------------------- removal
    def remove(self, entry_id: int) -> bool:
        """Delete one entry; the last row moves into its slot so C stays contiguous."""
        slot = self._slot_of.pop(entry_id, None)
        if slot is None:
            return False
        last = len(self) - 1
        entry = self._entries[slot]
        key = (entry.owner, exact_key(entry.query))
        if self._exact.get(key) == entry_id:
            del self._exact[key]
        if entry.owner is not None:
            self._n_private -= 1
        if entry.expires_at is not None:
            self._n_expiring -= 1
        if self.lsh is not None:
            self.lsh.remove(slot)
            if slot != last:
                self.lsh.rename(last, slot)          # the last row moves into the freed slot
        if slot != last:
            moved = self._entries[last]
            self._entries[slot] = moved
            self._tfidf[slot] = self._tfidf[last]
            self._vectors[slot] = self._vectors[last]
            self._last_used[slot] = self._last_used[last]
            self._owner_codes[slot] = self._owner_codes[last]
            self._rho[slot] = self._rho[last]
            self._expires[slot] = self._expires[last]
            self._slot_of[moved.id] = slot
        self._entries.pop()
        self._tfidf.pop()
        self._vectors[last] = 0.0
        self._owner_codes[last] = 0
        self._rho[last] = 1.0
        self._expires[last] = np.inf
        self._tfidf_matrix = None
        return True

    def observe(self, entry_id: int, score: float, correct: bool, replace_positive: bool = False) -> float | None:
        """Per-entry thresholds: record one (score, correct?) observation; returns the entry's tau."""
        from .thresholds import add_observation
        entry = self.get(entry_id)
        if entry is None:
            return None
        return add_observation(entry, score, correct, self.threshold, self.entry_precision_target, replace_positive)

    def remove_where(self, predicate) -> int:
        """Delete every entry for which ``predicate(entry)`` is true; returns how many."""
        ids = [e.id for e in self._entries if predicate(e)]
        for entry_id in ids:
            self.remove(entry_id)
        return len(ids)

    def purge_expired(self) -> int:
        if not self._n_expiring:
            return 0
        now = time.time()
        return self.remove_where(lambda e: self.expired(e, now))

    def remove_owned_by(self, owner: str) -> int:
        """Delete every private entry of ``owner``; returns how many were removed."""
        ids = [e.id for e in self._entries if e.owner == owner]
        for entry_id in ids:
            self.remove(entry_id)
        return len(ids)

    def _evict_lru(self) -> None:
        slot = int(np.argmin(self._last_used[: len(self)]))
        self.remove(self._entries[slot].id)
        self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()
        self._tfidf.clear()
        self._slot_of.clear()
        self._exact.clear()
        self._vectors[:] = 0.0
        self._last_used[:] = 0.0
        self._owner_codes[:] = 0
        self._rho[:] = 1.0
        self._expires[:] = np.inf
        self._n_expiring = 0
        self._tfidf_matrix = None
        self._owner_code.clear()
        self._n_private = 0
        if self.lsh is not None:
            self.lsh.clear()
        self.hits = self.misses = self.evictions = 0

    # -------------------------------------------------------------- utilities
    def reembed(self, embedder: "LSAEmbedder", min_known_ratio: float = 0.0,
                precomputed: dict | None = None) -> int:
        """Recompute every vector with a new embedder (e.g. after refitting the SVD).

        ``precomputed`` maps entry id -> (vector, sparse TF-IDF row, rho) already computed
        with ``embedder`` (a background rebuild does that outside the lock); only the
        entries missing from it are encoded here. Returns how many were encoded here."""
        self.dim = embedder.dim
        precomputed = precomputed or {}
        missing = [i for i, e in enumerate(self._entries) if e.id not in precomputed]
        if self._entries:
            vectors = np.zeros((len(self), self.dim))
            rho = np.ones(len(self))
            rows: list[SparseRow] = [_EMPTY_ROW] * len(self)
            for i, e in enumerate(self._entries):
                if e.id in precomputed:
                    vectors[i], rows[i], rho[i] = precomputed[e.id]
            if missing:
                V, X, r = embedder.encode_with_rho([self._entries[i].query for i in missing], min_known_ratio)
                for j, i in enumerate(missing):
                    vectors[i], rows[i], rho[i] = V[j], as_sparse_row(X[j]), r[j]
            rows = [row if np.any(vectors[i]) else _EMPTY_ROW for i, row in enumerate(rows)]
        else:
            vectors, rows, rho = np.zeros((0, self.dim)), [], np.zeros(0)
        capacity = max(64, self._vectors.shape[0])
        self._vectors = np.zeros((capacity, self.dim))
        self._vectors[: len(self)] = vectors
        self._rho = np.ones(capacity)
        self._rho[: len(self)] = rho
        self._tfidf = rows
        self._tfidf_matrix = None
        self.model_fingerprint = embedder.fingerprint
        self.query_policy = embedder.policy_signature
        if self.lsh is not None:
            s = self.lsh_settings
            self.lsh = la.RandomHyperplaneLSH(self.dim, s["n_bits"], s["n_tables"], s["seed"])
            self.lsh.add(range(len(self)), vectors)
        return len(missing)

    def stats(self) -> dict:
        total = self.hits + self.misses
        by_source: dict[str, int] = {}
        for e in self._entries:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        return {
            "entries": len(self),
            "shared_entries": len(self) - self._n_private,
            "private_entries": self._n_private,
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else None,
            "evictions": self.evictions,
            "threshold": self.threshold,
            "lsa_weight": self.lsa_weight,
            "min_latent_coverage": self.min_latent_coverage,
            "lsh": self.lsh is not None,
            "expiring_entries": self._n_expiring,
            "ttl_seconds": self.ttl_seconds,
            "by_source": by_source,
            "memory_bytes": int(len(self) * self.dim * 8 + sum(i.nbytes + v.nbytes for i, v in self._tfidf)),
        }

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors[: len(self)])
        np.save(directory / "rho.npy", self._rho[: len(self)])
        lengths = np.array([i.size for i, _ in self._tfidf], dtype=np.int64)
        np.savez_compressed(
            directory / "tfidf.npz",
            indptr=np.concatenate([[0], np.cumsum(lengths)]),
            indices=np.concatenate([i for i, _ in self._tfidf]) if self._tfidf else np.empty(0, np.int32),
            data=np.concatenate([v for _, v in self._tfidf]) if self._tfidf else np.empty(0),
        )
        meta = {
            "dim": self.dim,
            "threshold": self.threshold,
            "max_entries": self.max_entries,
            "model_fingerprint": self.model_fingerprint,
            "query_policy": self.query_policy,
            "next_id": self._next_id,
            "counters": {"hits": self.hits, "misses": self.misses, "evictions": self.evictions},
            "entries": [e.to_dict() for e in self._entries],
        }
        tmp = directory / "entries.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(directory / "entries.json")
        return directory

    @classmethod
    def load(cls, directory: str | Path, embedder: "LSAEmbedder | None" = None,
             min_known_ratio: float = 0.0, **settings) -> "SemanticCache":
        """Load a saved cache. ``settings`` (threshold, max_entries, use_lsh, ...) override
        the saved values. If ``embedder`` differs from the one the cache was built with,
        all entries are re-embedded from their stored text."""
        directory = Path(directory)
        meta = json.loads((directory / "entries.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy")
        rho_file = directory / "rho.npy"
        rho = np.load(rho_file) if rho_file.exists() else None
        if rho is not None and rho.shape[0] != len(meta["entries"]):
            rho = None
        tfidf_file = directory / "tfidf.npz"
        tf = None
        if tfidf_file.exists():
            # Read each array once: indexing an NpzFile decompresses the array again on
            # every access, which made loading quadratic in the number of entries.
            with np.load(tfidf_file) as archive:
                tf = {name: archive[name] for name in ("indptr", "indices", "data")}
        params = {"threshold": meta["threshold"], "max_entries": meta["max_entries"]}
        params.update(settings)
        cache = cls(meta["dim"], model_fingerprint=meta.get("model_fingerprint", ""),
                    query_policy=meta.get("query_policy", ""), **params)
        for slot, d in enumerate(meta["entries"]):
            entry = CacheEntry(**d)
            if slot == cache._vectors.shape[0]:
                cache._grow()
            cache._vectors[slot] = vectors[slot]
            cache._last_used[slot] = entry.last_used_at
            cache._rho[slot] = 1.0 if rho is None else float(rho[slot])
            cache._set_owner(slot, entry.owner)
            cache._set_expiry(slot, entry.expires_at)
            cache._entries.append(entry)
            if tf is not None and slot + 1 < len(tf["indptr"]):
                a, b = tf["indptr"][slot], tf["indptr"][slot + 1]
                cache._tfidf.append((tf["indices"][a:b].astype(np.int32), tf["data"][a:b]))
            else:
                cache._tfidf.append(_EMPTY_ROW)
            cache._slot_of[entry.id] = slot
            key = exact_key(entry.query)
            if key:
                cache._exact[(entry.owner, key)] = entry.id
        cache._next_id = meta.get("next_id", len(cache) + 1)
        counters = meta.get("counters", {})
        cache.hits, cache.misses = counters.get("hits", 0), counters.get("misses", 0)
        cache.evictions = counters.get("evictions", 0)
        if cache.lsh is not None and len(cache):
            cache.lsh.add(range(len(cache)), cache._vectors[: len(cache)])
        stale = embedder is not None and (embedder.fingerprint != cache.model_fingerprint
                                          or embedder.policy_signature != cache.query_policy
                                          or tf is None or rho is None)
        if stale:
            cache.reembed(embedder, min_known_ratio)
        while len(cache) > cache.max_entries:
            cache._evict_lru()
        return cache

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "entries.json").exists() and (directory / "vectors.npy").exists()
