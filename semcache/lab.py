"""The Matrix Lab: checks and worked numbers behind the served model, for the "Matrix Lab"
tab of the UI. Nothing here changes the served model or the cache.

    correctness()   ||X - X_k||_F^2 vs the sum of the dropped sigma_i^2 (Eckart-Young),
                    ||V_k^T V_k - I||_F, and what the Jacobi solver did (sweeps, rotations,
                    time, the off-diagonal norm per sweep)
    worked(query)   idf of the rarest word, the TF-IDF row, e, the best cosine as a sum,
                    theta = arccos(score) vs arccos(tau), and the route
    ops(query)      floating-point operations for a cache hit vs an LLM answer (an estimate)
    refit(k)        a sandbox model with another k: the served model never changes
    latency()       per-request latency, hits vs LLM calls

The training matrix X is rebuilt from the corpus with the served model's vocabulary and
IDF, so it is exactly the matrix the model was fitted on.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from . import linalg as la
from .embedder import LSAEmbedder

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import SemanticCacheService

DENSE_LIMIT = 5_000_000          # entries of X we are willing to densify for explicit checks
REFIT_MAX_DOCS = 20_000          # an interactive sandbox refit is limited to corpora this size
DEFAULT_TOKENS = (40, 150)       # assumed prompt / output tokens when no LLM call was measured
FALLBACK_MODEL = "llama3.2:3b"   # the README's demo model, used (and labelled) when none is known


def _r(x, digits: int = 6) -> float:
    return round(float(x), digits)


def lab_corpus(service: "SemanticCacheService") -> list[str]:
    """The texts the served model was fitted on: the snapshot saved with the model when the
    corpus includes cached answers (faq-live), else the configured corpus."""
    snap = service.settings.model_dir / "corpus.json"
    if snap.exists():
        return json.loads(snap.read_text(encoding="utf-8"))["texts"]
    from .pipeline import build_corpus
    corpus, _ = build_corpus(service.settings)
    return corpus


def _training_matrix(emb: LSAEmbedder, corpus: list[str]):
    X = emb.tfidf([t for t in corpus if t and t.strip()])
    keep = emb.svd_mask
    return la.zero_columns(X, keep) if not keep.all() else X


def model_params_billion(name: str | None, override: float | None = None) -> float | None:
    """Parameter count from a model tag such as "llama3.2:3b" or "qwen2.5-7b-instruct"."""
    if override:
        return float(override)
    if not name:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower().replace("_", "-").split("/")[-1])
    return float(m.group(1)) if m else None


class Lab:
    def __init__(self, service: "SemanticCacheService"):
        self.service = service
        self._lock = threading.Lock()
        self._refit_lock = threading.Lock()
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------ correctness
    def correctness(self) -> dict:
        emb = self.service.embedder
        key = emb.fingerprint
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        corpus = lab_corpus(self.service)
        X = _training_matrix(emb, corpus)
        n, m = X.shape
        k_fit, _ = emb.fit_rank(n, m)
        cfg = emb.config
        trace: dict = {}
        t0 = time.perf_counter()
        small = min(n, m) <= 300 and n * m <= DENSE_LIMIT
        if small:
            # Same path as the fit (exact_svd), but keep every singular value.
            D = X.toarray()
            _, s_all, _ = la.exact_svd(D, trace=trace)
            trace["path"] = "exact"
        else:
            _, s_all, _ = la.truncated_svd(X, k_fit, oversample=cfg.oversample, power_iters=cfg.power_iters,
                                           seed=cfg.seed, trace=trace)
        svd_seconds = time.perf_counter() - t0
        k = emb.dim
        total = la.frobenius_norm_sq(X)
        V, s = emb.V, emb.s
        if small:
            U = (D @ V) / s
            resid = D - (U * s) @ V.T
            residual = float(np.einsum("ij,ij->", resid, resid))
            residual_how = "explicit: X_k = U_k Σ_k V_kᵀ formed and subtracted, entries squared and summed"
            dropped = float(np.sum(s_all[k:] ** 2))
            dropped_how = "Σ_{i>k} σ_i² from a full SVD of X (Jacobi on the Gram matrix)"
        else:
            XV = np.asarray(X @ V)
            residual = total - float(np.einsum("ij,ij->", XV, XV))
            residual_how = "‖X‖_F² − ‖X V_k‖_F² (V_k orthonormal; X is too large to densify)"
            dropped = total - float(np.sum(s ** 2))
            dropped_how = "‖X‖_F² − Σ_{i≤k} σ_i² (the full spectrum is too large to compute)"
        G = V.T @ V
        ortho = float(np.sqrt(np.sum((G - np.eye(G.shape[0])) ** 2)))
        diff = abs(residual - dropped)
        spectrum = s_all if small else s
        energy = np.cumsum(spectrum ** 2) / total if total else np.zeros_like(spectrum)
        out = {
            "profile": self.service.settings.profile, "fingerprint": key,
            "shape": [n, m], "nnz": int(X.nnz), "k": k, "frobenius_sq": _r(total, 9),
            "eckart_young": {"residual_sq": residual, "dropped_sigma_sq": dropped, "difference": diff,
                             "relative": diff / max(dropped, 1e-300), "equal": diff <= 1e-9 * max(1.0, total),
                             "residual_method": residual_how, "dropped_method": dropped_how},
            "orthogonality": {"value": ortho, "ok": ortho < 1e-8},
            "sigma_check": {"max_abs_diff": float(np.max(np.abs(spectrum[:k] - s))) if spectrum.size >= k else None},
            "jacobi": {"path": trace.get("path"), "size": trace.get("size"), "sweeps": trace.get("sweeps"),
                       "rotations": trace.get("rotations"), "seconds": trace.get("seconds"),
                       "converged": trace.get("converged"), "tol": trace.get("tol"),
                       "frobenius": trace.get("frobenius"),
                       "off_norms": [float(x) for x in trace.get("off_norms", [])],
                       "svd_seconds": svd_seconds},
            "spectrum": {"sigma": [float(x) for x in spectrum], "energy_cumulative": [_r(x) for x in energy],
                         "full": bool(small), "k": k, "energy_kept": _r(np.sum(s ** 2) / total if total else 0.0)},
        }
        with self._lock:
            self._cache = {k: v for k, v in self._cache.items() if k.endswith(key)}   # drop older models
            self._cache[key] = out
        return out

    # ------------------------------------------------------------- worked maths
    def worked(self, query: str, threshold: float | None = None, user: str | None = None,
               admin: bool = False) -> dict:
        svc = self.service
        emb = svc.embedder
        query = svc._validate(query)
        result, _ = svc.lookup(query, threshold, 3, record=False, user=user)
        tau = float(result.threshold)
        X = emb.tfidf([query])
        row = X.tocsr()
        row.sort_indices()
        counts: dict[str, int] = {}
        for t in emb.analyze(query):
            counts[t] = counts.get(t, 0) + 1
        N = emb.report.n_docs
        smooth = emb.config.smooth_idf
        terms = []
        raw = []
        for i, w in zip(row.indices, row.data):
            term = emb.vocab.terms[i]
            idf = float(emb.idf[i])
            df = (1 + N) * math.exp(-(idf - 1)) - 1 if smooth else N * math.exp(-idf)
            tf = counts.get(term, 1)
            tfw = 1 + math.log(tf) if emb.config.sublinear_tf else float(tf)
            raw.append(tfw * idf)
            terms.append({"term": term, "tf": tf, "df": int(round(df)), "idf": _r(idf, 4), "tf_weight": _r(tfw, 4),
                          "raw": _r(tfw * idf, 4), "weight": _r(w, 4), "bigram": " " in term})
        norm = float(np.sqrt(np.sum(np.square(raw)))) if raw else 0.0
        words = [t for t in terms if not t["bigram"]] or terms
        rarest = max(words, key=lambda t: t["idf"]) if words else None

        z = np.asarray(X @ emb.V).ravel()
        zn = float(np.sqrt(z @ z))
        e = z / zn if zn > 0 else np.zeros_like(z)
        adm = emb.admission(query, svc.settings.cache.min_known_ratio)
        known, total = adm.known, adm.total
        best = result.matches[0] if result.matches else None
        match = None
        if best is not None:
            f = svc.cache.vector_of(best.entry.id)
            prod = e * f
            order = np.argsort(-np.abs(prod))
            top = [{"j": int(j) + 1, "e": _r(e[j], 4), "f": _r(f[j], 4), "product": _r(prod[j], 4)} for j in order[:6]]
            readable = svc.can_read(best.entry, svc._user(user), admin)
            theta = math.degrees(math.acos(max(-1.0, min(1.0, best.score))))
            match = {"query": best.entry.query if readable else "another user's question", "hidden": not readable,
                     "exact": result.method == "exact", "score": _r(best.score, 4), "lsa": _r(float(prod.sum()), 4),
                     "tfidf": None if best.tfidf is None else _r(best.tfidf, 4), "scored_by": best.scored_by,
                     "lsa_weight": svc.cache.lsa_weight, "top": top, "rest": _r(float(prod[order[6:]].sum()), 4),
                     "terms_in_sum": int(prod.size), "theta_deg": _r(theta, 2)}
        return {
            "query": query, "N": N, "smooth_idf": smooth, "sublinear_tf": emb.config.sublinear_tf,
            "rarest": rarest, "terms": terms, "norm": _r(norm, 4),
            "k": emb.dim, "latent_norm": _r(zn, 4), "rho": _r(result.rho, 4),
            "e": [_r(x, 4) for x in e], "e_top": [{"j": int(j) + 1, "value": _r(e[j], 4)}
                                                  for j in np.argsort(-np.abs(e))[:6]],
            "tau": tau, "theta_tau_deg": _r(math.degrees(math.acos(tau)), 2),
            "match": match, "hit": result.hit, "refused": result.refused,
            "route": svc.route_for(result, known, total, admission=adm),
        }

    # ------------------------------------------------------------- operations
    def ops(self, query: str, user: str | None = None) -> dict:
        """FLOPs for answering ``query`` from the cache vs asking the LLM (an estimate)."""
        svc = self.service
        emb = svc.embedder
        query = svc._validate(query)
        result, _ = svc.lookup(query, None, 1, record=False, user=user)
        X = emb.tfidf([query])
        nnz, k, n = int(X.nnz), emb.dim, int(result.compared)
        parts = [
            {"step": "TF-IDF row (log, × idf, normalise)", "formula": "4·nnz", "flops": 4 * nnz},
            {"step": "fold-in q V_k", "formula": "2·nnz·k", "flops": 2 * nnz * k},
            {"step": "normalise e, ρ", "formula": "5·k", "flops": 5 * k},
            {"step": "scores s = C e", "formula": "2·n·k", "flops": 2 * n * k},
            {"step": "pick the best (comparisons)", "formula": "n", "flops": n},
        ]
        cache = svc.cache
        if cache.lsa_weight < 1.0 or cache.min_latent_coverage > 0:
            # word-overlap score: one multiply-add per cached non-zero in the question's columns
            T = cache._tfidf_rows()
            cols = X.indices[X.indices < T.shape[1]]
            post = int(np.sum(np.diff(T.tocsc().indptr)[cols])) if cols.size and T.shape[0] else 0
            parts.append({"step": "word-overlap scores (sparse)", "formula": "2·postings", "flops": 2 * post})
        hit_flops = sum(p["flops"] for p in parts)

        model = None
        try:
            model = svc.backend.describe().get("model")
        except Exception:                       # noqa: BLE001 - describe() may hit the network
            model = None
        model = model or (svc.settings.llm.model if svc.settings.llm.model != "auto" else None)
        model_basis = "the configured / running model"
        if not model and not getattr(svc.settings.llm, "model_params_billion", None):
            model, model_basis = FALLBACK_MODEL, "assumed: the README's demo model (no model is configured or running)"
        override = getattr(svc.settings.llm, "model_params_billion", None)
        params_b = model_params_billion(model, override)
        summary = svc.metrics.summary(svc.pricing)["tokens"]
        if summary["calls_with_counts"]:
            p_tok, o_tok = summary["avg_input_per_call"], summary["avg_output_per_call"]
            token_basis = f"average of {summary['calls_with_counts']} real LLM calls (reported token counts)"
        else:
            p_tok, o_tok = DEFAULT_TOKENS
            token_basis = "assumed (no LLM call with token counts yet)"
        llm = None
        if params_b:
            flops = 2.0 * params_b * 1e9 * (p_tok + o_tok)
            llm = {"model": model, "model_basis": model_basis, "params_billion": params_b,
                   "prompt_tokens": _r(p_tok, 1),
                   "output_tokens": _r(o_tok, 1), "token_basis": token_basis, "flops": flops,
                   "formula": "2 · P · (prompt + output tokens)",
                   "note": "a decoder forward pass costs about 2 FLOPs per parameter per token; "
                           "attention over the context is left out, so this is a lower bound"}
        return {"query": query, "nnz": nnz, "k": k, "n": n, "hit": {"parts": parts, "flops": hit_flops},
                "llm": llm, "ratio": (llm["flops"] / hit_flops) if llm and hit_flops else None,
                "estimate": True,
                "missing": None if llm else "the model's parameter count is unknown; set [llm] model_params_billion"}

    # ------------------------------------------------------------- sandbox refit
    def refit(self, k: int, query: str | None = None, user: str | None = None) -> dict:
        """Fit a sandbox model with rank k on the same corpus. The served model and the
        cache are never touched; the sandbox is thrown away after the response."""
        from .measure import headline
        svc = self.service
        corpus = lab_corpus(svc)
        if len(corpus) > REFIT_MAX_DOCS:
            raise ValueError(f"the corpus has {len(corpus):,} texts; an interactive refit is limited to "
                             f"{REFIT_MAX_DOCS:,}. Use `python -m semcache build` with another rank instead.")
        served = svc.embedder
        X = _training_matrix(served, corpus)
        max_k = min(X.shape) - 1
        if not 1 <= k <= max_k:
            raise ValueError(f"k must be between 1 and {max_k} for this {X.shape[0]} × {X.shape[1]} matrix")
        if not self._refit_lock.acquire(blocking=False):
            raise RuntimeError("another refit is running; try again in a moment")
        try:
            t0 = time.perf_counter()
            cfg = replace(served.config, rank=k, max_rank_fraction=1.0)
            from .pipeline import configure_embedder
            sandbox = configure_embedder(LSAEmbedder(cfg).fit(corpus), svc.settings)
            seconds = time.perf_counter() - t0
            check = self.correctness()
            Xs = _training_matrix(sandbox, corpus)
            total = la.frobenius_norm_sq(Xs)
            V, s = sandbox.V, sandbox.s
            if Xs.shape[0] * Xs.shape[1] <= DENSE_LIMIT:
                D = Xs.toarray()
                resid = D - (D @ V) @ V.T
                residual = float(np.einsum("ij,ij->", resid, resid))
            else:
                XV = np.asarray(Xs @ V)
                residual = total - float(np.einsum("ij,ij->", XV, XV))
            sig = np.array(check["spectrum"]["sigma"])
            dropped = float(np.sum(sig[sandbox.dim:] ** 2)) if check["spectrum"]["full"] else total - float(np.sum(s ** 2))
            h = headline(svc.settings, sandbox)
            served_h = self._served_headline()
            out = {"k": sandbox.dim, "requested_k": k, "seconds": seconds, "vocabulary_size": len(sandbox.vocab),
                   "energy_kept": sandbox.report.energy_kept, "sigma": [_r(x) for x in s],
                   "eckart_young": {"residual_sq": residual, "dropped_sigma_sq": dropped,
                                    "equal": abs(residual - dropped) <= 1e-9 * max(1.0, total)},
                   "headline": {key: h[key] for key in self.HEADLINE_KEYS},
                   "served_headline": served_h,
                   "served": {"k": served.dim, "fingerprint": served.fingerprint}, "question": None}
            if query:
                query = svc._validate(query)
                with svc._lock:
                    shadow = copy.deepcopy(svc.cache)
                shadow.reembed(sandbox, svc.settings.cache.min_known_ratio)
                vec, rows, rho = sandbox.encode_with_rho([query], svc.settings.cache.min_known_ratio)
                r = shadow.lookup(vec[0], 1, record=False, query_text=query, query_tfidf=rows[0],
                                  user=svc._user(user), query_rho=float(rho[0]))
                served_r, _ = svc.lookup(query, None, 1, record=False, user=user)
                out["question"] = {"query": query, "sandbox_score": _r(r.score, 4), "sandbox_hit": r.hit,
                                   "served_score": _r(served_r.score, 4), "served_hit": served_r.hit}
            return out
        finally:
            self._refit_lock.release()

    HEADLINE_KEYS = ("auc", "precision", "recall", "accepted_hits", "accepted_hits_correct", "random_false_hits",
                     "random_asked", "best_f1", "best_f1_tau", "tau")

    def _served_headline(self) -> dict:
        from .measure import headline
        key = "headline:" + self.service.embedder.fingerprint
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        h = headline(self.service.settings, self.service.embedder)
        out = {k: h[k] for k in self.HEADLINE_KEYS}
        with self._lock:
            self._cache[key] = out
        return out

    # ---------------------------------------------------------------- latency
    def latency(self, limit: int = 500, user: str | None = None) -> dict:
        events = self.service.metrics.events(user)[-limit:]
        points = [{"t": e["t"], "kind": e["kind"], "total_ms": e.get("total_ms"), "llm_ms": e.get("llm_ms"),
                   "llm": bool(e.get("llm")), "fallback": bool(e.get("fallback")), "backend": e.get("backend")}
                  for e in events]
        return {"points": points, "limit": limit}
