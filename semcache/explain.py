"""Explain one lookup: every number behind "hit" or "miss", for the UI's math view.

Nothing is stored and the LLM is never called. ``explain_query`` returns
plain dicts, ready to be sent as JSON (POST /api/explain).

Why the explanation is exact
----------------------------
Let t_a be term a in latent space (row a of V_k Sigma_k^(p-1), see linalg.project),
q_a and c_b the unit TF-IDF weights of the question and of a cached question, and

    z_q = sum_a q_a t_a,      z_c = sum_b c_b t_b      (their latent vectors)

Then the LSA cosine splits into one part per pair of terms:

    cos(q, c) = sum_{a,b}  q_a c_b (t_a . t_b) / (|z_q| |z_c|)

Pairs with a = b are words the two questions share; pairs with a != b are links the
SVD learned between different words ("login" and "sign in"). The same cosine also
splits per latent direction: cos = sum_j e_j f_j, with e = z_q/|z_q| and f = z_c/|z_c|.

The effect of a word is how much the score drops when that word is left out of the
question (its TF-IDF weight set to zero, everything else unchanged).

Latent-coverage guard: rho = ||q V_k|| / ||q|| for each question. When the smaller rho
of a pair is below the cache's ``min_latent_coverage`` the pair is scored by word
overlap (the TF-IDF cosine) instead; the effects then use that score.

Privacy: entries the caller may not read (``can_read``) keep their numbers, but their
question text and words are replaced.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .text import STOPWORDS, _TOKEN, fold_plural, normalize

if TYPE_CHECKING:  # pragma: no cover
    from .cache import Match, SemanticCache
    from .embedder import LSAEmbedder

EPS = 1e-12
MAX_TERMS = 24          # terms per question shown in the pair grid
DIRECTION_WORDS = 4     # top words used to label a latent direction
TOP_DIMS = 6            # latent directions listed per comparison
NOTABLE = 0.02          # a word whose effect is smaller than this is not worth a reason
HIDDEN_QUESTION = "another user's question"
VK_BAND = 24            # sampled rows of V_k drawn around the question's own rows
C_ALL = 48              # caches up to this size are drawn whole ...
C_TOP, C_BAND = 20, 20  # ... larger ones as the top rows plus a sampled band


@dataclass
class _Encoded:
    """One question in both representations, restricted to its known terms."""
    terms: list[str]
    index: np.ndarray       # vocabulary indices of the terms
    weights: np.ndarray     # unit-length TF-IDF weights q_a
    T: np.ndarray           # term vectors t_a, one row per term (len(terms) x k)
    z: np.ndarray           # raw latent vector sum_a q_a t_a
    norm: float             # |z|
    R: np.ndarray           # the terms' rows of V_k (for rho)
    zr: np.ndarray          # q V_k
    in_svd: np.ndarray      # True for terms the SVD was fitted on
    svd_sq: float           # ||q||^2 over those terms
    rho: float              # ||q V_k|| / ||q||

    def rho_without(self, a: int) -> float:
        """rho with term a left out."""
        zr = self.zr - self.weights[a] * self.R[a]
        den = self.svd_sq - (self.weights[a] ** 2 if self.in_svd[a] else 0.0)
        return float(min(1.0, np.linalg.norm(zr) / math.sqrt(den))) if den > EPS else 0.0


def _encode(emb: "LSAEmbedder", text: str) -> _Encoded:
    row = emb.tfidf([text]).tocsr()
    row.sort_indices()
    index = row.indices.astype(np.intp)
    weights = row.data.astype(np.float64)
    order = np.argsort(-weights, kind="stable")          # heaviest terms first
    index, weights = index[order], weights[order]
    scale = emb.s ** (emb.config.sigma_power - 1.0)
    R = emb.V[index]
    T = R * scale
    z = weights @ T if index.size else np.zeros(emb.dim)
    zr = weights @ R if index.size else np.zeros(emb.dim)
    in_svd = emb.svd_mask[index]
    svd_sq = float(np.sum(weights[in_svd] ** 2))
    rho = float(min(1.0, np.linalg.norm(zr) / math.sqrt(svd_sq))) if svd_sq > EPS else 0.0
    return _Encoded([emb.vocab.terms[i] for i in index], index, weights, T, z, float(np.linalg.norm(z)),
                    R, zr, in_svd, svd_sq, rho)


def _passes_guard(emb: "LSAEmbedder", text: str, ratio: float) -> tuple[int, int, bool]:
    known, total = emb.coverage(text)
    ok = total > 0 and (ratio <= 0 or known / total >= ratio) and known > 0
    return known, total, ok


def _direction_words(emb: "LSAEmbedder", j: int, n: int = DIRECTION_WORDS) -> list[str]:
    """The words with the largest weight (either sign) in column j of V_k."""
    column = np.abs(emb.V[:, j])
    n = min(n, column.size)
    top = np.argpartition(-column, n - 1)[:n]
    top = top[np.argsort(-column[top])]
    return [emb.vocab.terms[t] for t in top]


def _r(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def _tokens(emb: "LSAEmbedder", text: str) -> list[dict]:
    """Each word of the (normalised) question, and what the tokenizer did with it."""
    out = []
    for word in _TOKEN.findall(normalize(text)):
        if word in STOPWORDS:
            out.append({"word": word, "term": None, "status": "stopword"})
            continue
        term = fold_plural(word) if emb.config.stem else word
        out.append({"word": word, "term": term, "status": "known" if term in emb.vocab else "unknown"})
    return out


def _query_terms(emb: "LSAEmbedder", text: str, enc: _Encoded) -> list[dict]:
    counts = Counter(emb.analyze(text))
    return [{"term": t, "tf": counts[t], "idf": _r(emb.idf[i]), "weight": _r(w),
             "bigram": " " in t}
            for t, i, w in zip(enc.terms, enc.index, enc.weights)]


def _drivers(enc: _Encoded, j: int, n: int = 3) -> list[str]:
    """The question's words that contribute most to its coordinate along direction j."""
    parts = enc.weights * enc.T[:, j]
    return [enc.terms[a] for a in np.argsort(-np.abs(parts))[:n] if abs(parts[a]) > 1e-4 * enc.norm]


def _latent_summary(emb: "LSAEmbedder", enc: _Encoded, top: int = 3) -> list[dict]:
    """The directions the question points along most, and which of its words put it there."""
    if enc.norm < EPS:
        return []
    e = enc.z / enc.norm
    return [{"dim": int(j) + 1, "value": _r(e[j]), "words": _direction_words(emb, int(j)),
             "from": _drivers(enc, int(j))}
            for j in np.argsort(-np.abs(e))[:top]]


def _score(lsa: float, tfidf: float, w: float, pair_rho: float = 1.0, cut: float = 0.0) -> float:
    """The cache's decision score (see linalg.guarded_scores)."""
    if pair_rho < cut:
        return tfidf
    return lsa if w >= 1.0 else w * lsa + (1.0 - w) * tfidf


def _explain_match(emb: "LSAEmbedder", q: _Encoded, match: "Match", rank: int, tau: float,
                   lsa_weight: float, ratio: float, exact_id: int | None, query_ok: bool,
                   cut: float, readable: bool) -> dict:
    entry = match.entry
    c = _encode(emb, entry.query)
    _, _, c_ok = _passes_guard(emb, entry.query, ratio)
    pair_rho = min(q.rho, c.rho)
    by_overlap = pair_rho < cut
    base = {"rank": rank, "id": entry.id, "query": entry.query if readable else HIDDEN_QUESTION,
            "hidden": not readable, "source": entry.source, "owner": entry.owner,
            "score": _r(match.score), "exact": entry.id == exact_id,
            "terms": [{"term": t, "weight": _r(w)} for t, w in zip(c.terms, c.weights)] if readable else [],
            "rho": _r(c.rho), "pair_rho": _r(pair_rho), "rho_blocked": by_overlap,
            "scored_by": "exact" if entry.id == exact_id else match.scored_by}
    # A question the known-word guard blocks can only hit by exact repeat.
    base["hit"] = base["exact"] or (query_ok and c_ok and match.score >= tau)

    if q.norm < EPS or c.norm < EPS or not c_ok:
        why = ("the cached question has too few words the model knows, so it can only match "
               "an exact repeat" if not c_ok or c.norm < EPS else "your question has no known words")
        return {**base, "lsa": 0.0, "tfidf": 0.0, "blocked": why, "pairs": None, "effects": [],
                "dims": [], "c_latent": [], "reasons": []}

    # --- term-pair decomposition of the LSA cosine -------------------------------------
    gram = q.T @ c.T.T                                      # t_a . t_b
    pair = np.outer(q.weights, c.weights) * gram / (q.norm * c.norm)
    lsa = float(pair.sum())
    term_norm_q = np.linalg.norm(q.T, axis=1)
    term_norm_c = np.linalg.norm(c.T, axis=1)
    term_cos = gram / np.maximum(np.outer(term_norm_q, term_norm_c), EPS)

    # --- exact word overlap (the TF-IDF cosine) ------------------------------------------
    c_weight = dict(zip(c.index.tolist(), c.weights.tolist()))
    shared_w = np.array([c_weight.get(int(i), 0.0) for i in q.index])
    tfidf = float(q.weights @ shared_w)
    score = _score(lsa, tfidf, lsa_weight, pair_rho, cut)

    # --- leave-one-word-out effects --------------------------------------------------------
    effects = []
    for a, term in enumerate(q.terms):
        z = q.z - q.weights[a] * q.T[a]
        nz = float(np.linalg.norm(z))
        lsa_wo = float(z @ c.z) / (nz * c.norm) if nz > EPS else 0.0
        rest = 1.0 - q.weights[a] ** 2
        tfidf_wo = (tfidf - q.weights[a] * shared_w[a]) / math.sqrt(rest) if rest > EPS else 0.0
        without = _score(lsa_wo, tfidf_wo, lsa_weight, min(q.rho_without(a), c.rho), cut)
        b = int(np.argmax(pair[a])) if c.terms else 0
        effects.append({"term": term, "effect": _r(score - without), "without": _r(without),
                        "shared": bool(shared_w[a] > 0) and readable,
                        "partner": c.terms[b] if c.terms and readable else None,
                        "partner_cos": _r(term_cos[a, b]) if c.terms else 0.0,
                        "row_total": _r(pair[a].sum())})
    effects.sort(key=lambda d: -abs(d["effect"]))

    # --- per-direction decomposition ---------------------------------------------------------
    e, f = q.z / q.norm, c.z / c.norm
    prod = e * f
    dims = [{"dim": int(j) + 1, "value": _r(prod[j]), "q": _r(e[j]), "c": _r(f[j]),
             "words": _direction_words(emb, int(j)),
             "from_q": _drivers(q, int(j)), "from_c": _drivers(c, int(j)) if readable else []}
            for j in np.argsort(-np.abs(prod))[:TOP_DIMS]]

    nq, nc = min(len(q.terms), MAX_TERMS), min(len(c.terms), MAX_TERMS)
    pairs = ({"rows": q.terms[:nq], "cols": c.terms[:nc],
              "contribution": np.round(pair[:nq, :nc], 4).tolist(),
              "similarity": np.round(term_cos[:nq, :nc], 3).tolist()} if readable else None)

    result = {**base, "lsa": _r(lsa), "tfidf": _r(tfidf), "computed": _r(score), "blocked": None,
              "pairs": pairs, "effects": effects, "dims": dims, "c_latent": np.round(f, 4).tolist()}
    result["reasons"] = _reasons(q, c, result, pair, tau, cut, readable)
    return result


def _reasons(q: _Encoded, c: _Encoded, m: dict, pair: np.ndarray, tau: float, cut: float,
             readable: bool) -> list[dict]:
    """Plain-language reasons, strongest first. kind: "plus", "minus" or "info"."""
    out: list[dict] = []
    if m["rho_blocked"]:
        who = "your question" if q.rho <= c.rho else "the cached question"
        out.append({"kind": "info", "value": None,
                    "text": f"Scored by word overlap (TF-IDF), not by the SVD: only {m['pair_rho']:.0%} of "
                            f"{who} survives in the SVD space (ρ below {cut:.2f})"})
    for eff in m["effects"]:
        v = eff["effect"]
        if v >= NOTABLE:
            if eff["shared"]:
                text = f'"{eff["term"]}" appears in both questions'
            elif not readable:
                text = f'"{eff["term"]}" brings the two questions closer'
            elif m["rho_blocked"]:
                continue                          # without shared words it adds nothing here
            else:
                text = (f'"{eff["term"]}" is linked to "{eff["partner"]}" through the SVD '
                        f'(word similarity {eff["partner_cos"]:.2f})')
            out.append({"kind": "plus", "value": v, "text": text})
        elif v <= -NOTABLE:
            out.append({"kind": "minus", "value": v,
                        "text": f'"{eff["term"]}" points away from this question and pulls the score down'})

    # Weighty words of the cached question that nothing in the query covers.
    covered = pair.sum(axis=0)
    missing = [t for t, w, cov in zip(c.terms, c.weights, covered)
               if w >= 0.3 and " " not in t and cov < 0.05 and t not in q.terms] if readable else []
    if missing:
        out.append({"kind": "info", "value": None,
                    "text": "The cached question is also about " + ", ".join(f'"{t}"' for t in missing[:4])
                            + ", which your question doesn't cover"})

    if not m["exact"] and not m["hit"]:
        best = max(m["effects"], key=lambda d: d["without"], default=None)
        if best and best["without"] >= tau and best["effect"] < 0:
            out.append({"kind": "info", "value": None,
                        "text": f'Without "{best["term"]}" the score would be {best["without"]:.3f}, '
                                f"above τ = {tau:.2f}"})
    return out


def _vk_band(emb: "LSAEmbedder", exclude: set[int]) -> list[dict]:
    """A fixed random sample of V_k rows (the same for every question), so the math view can
    show the question's rows sliding out of a band of the matrix. Most rows are near zero:
    most words barely survive the SVD."""
    rng = np.random.default_rng(0)
    picks = [int(i) for i in rng.choice(len(emb.vocab), size=min(len(emb.vocab), VK_BAND + 16), replace=False)
             if int(i) not in exclude][:VK_BAND]
    return [{"term": emb.vocab.terms[i], "index": i, "vk": np.round(emb.V[i], 3).tolist()} for i in sorted(picks)]


def _cache_matrix(emb: "LSAEmbedder", cache: "SemanticCache", vector: np.ndarray, row, rho: float,
                  user: str | None, exact_id: int | None, readable: Callable) -> dict:
    """C as the math view draws it: rows in their order in C, each with its score."""
    every = cache.score_all(vector, row, rho, user)
    entries, scores, lsa, blocked = every["entries"], every["scores"].copy(), every["lsa"], every["blocked"]
    if exact_id is not None:
        scores = np.array([1.0 if e.id == exact_id else s for e, s in zip(entries, scores)])
    n = len(entries)
    order = np.argsort(-scores, kind="stable")
    rank = {int(i): r + 1 for r, i in enumerate(order[:3])}
    if n <= C_ALL:
        shown = list(range(n))
    else:
        top = set(int(i) for i in order[:C_TOP])
        rest = [i for i in range(n) if i not in top]
        band = np.random.default_rng(0).choice(rest, size=min(C_BAND, len(rest)), replace=False)
        shown = sorted(top | {int(i) for i in band})
    mode = "svd" if cache.lsa_weight >= 1.0 else "hybrid"
    rows = []
    for i in shown:
        e = entries[i]
        ok = bool(readable(e))
        rows.append({"id": e.id, "query": e.query if ok else HIDDEN_QUESTION, "hidden": not ok,
                     "source": e.source, "score": _r(scores[i]), "lsa": _r(lsa[i]),
                     "scored_by": "exact" if e.id == exact_id else "tfidf" if blocked[i] else mode,
                     "rank": rank.get(i), "vector": np.round(cache.vector_of(e.id), 3).tolist()})
    return {"n": n, "k": emb.dim, "shown": len(rows), "sampled": len(rows) < n, "rows": rows}


def explain_query(emb: "LSAEmbedder", cache: "SemanticCache", query: str, *, threshold: float | None = None,
                  top_k: int = 5, user: str | None = None, min_known_ratio: float = 0.0,
                  can_read: Callable[..., bool] | None = None) -> dict:
    """Everything the UI needs to show why ``query`` is (or isn't) answered from the cache."""
    tau = cache.threshold if threshold is None else float(threshold)
    cut = cache.min_latent_coverage
    readable = can_read or (lambda entry: True)
    known, total, guard_ok = _passes_guard(emb, query, min_known_ratio)
    q = _encode(emb, query)
    vectors, row = emb.encode([query], min_known_ratio)

    # The real decision (the known-word guard may zero the vector) ...
    decision = cache.lookup(vectors[0], top_k, tau, record=False, query_text=query,
                            query_tfidf=row, user=user, query_rho=q.rho)
    # ... and the neighbours the vector would have without the guard, so a blocked
    # question can still show what its scores would be.
    raw = q.z / q.norm if q.norm > EPS else np.zeros(emb.dim)
    neighbours = cache.lookup(raw, top_k, tau, record=False, query_text=query,
                              query_tfidf=row, user=user, query_rho=q.rho)
    exact_id = decision.matches[0].entry.id if decision.method == "exact" else None

    if decision.method == "exact":
        verdict = "exact"
    elif len(cache) == 0:
        verdict = "empty"
    elif q.norm < EPS:
        verdict = "no_known_words"
    elif not guard_ok:
        verdict = "guard"
    else:
        verdict = "hit" if decision.hit else "miss"

    return {
        "query": query,
        "normalized": normalize(query),
        "threshold": tau,
        "verdict": verdict,
        "hit": decision.hit,
        "score": _r(decision.score),
        "method": decision.method,
        "compared": decision.compared,
        "model": {"rank": emb.dim, "vocabulary_size": len(emb.vocab),
                  "sigma_power": emb.config.sigma_power, "lsa_weight": cache.lsa_weight,
                  "min_latent_coverage": cut},
        "rho": _r(q.rho),
        "rho_blocked": bool(cut > 0 and q.rho < cut and q.norm > EPS),
        "tokens": _tokens(emb, query),
        "bigrams": [{"term": t, "known": t in emb.vocab} for t in emb.analyze(query) if " " in t],
        "coverage": {"known": known, "total": total, "ratio": _r(known / total) if total else 0.0,
                     "min_ratio": min_known_ratio, "passed": guard_ok},
        "terms": _query_terms(emb, query, q),
        "vocab_size": len(emb.vocab),
        # the question's own rows of V_k (k numbers each) and a sampled band of others
        "term_rows": [{"term": t, "index": int(i), "weight": _r(w), "in_svd": bool(emb.svd_mask[i]),
                       "vk": np.round(emb.V[i], 3).tolist()}
                      for t, i, w in zip(q.terms, q.index, q.weights)],
        "vk_band": _vk_band(emb, set(int(i) for i in q.index)),
        "cache_matrix": _cache_matrix(emb, cache, raw, row, q.rho, user, exact_id, readable),
        "latent": np.round(raw, 4).tolist(),
        "latent_top": _latent_summary(emb, q),
        "matches": [_explain_match(emb, q, m, i + 1, tau, cache.lsa_weight, min_known_ratio, exact_id,
                                   guard_ok, cut, bool(readable(m.entry)))
                    for i, m in enumerate(neighbours.matches)],
    }
