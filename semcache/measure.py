"""Measurements that change no behaviour: headline numbers, a traffic-mix simulation,
a cross-validated threshold, an LSH (b, L) sweep and a few fact checks on the model.

    python -m semcache measure --profile faq            # everything below
    python -m semcache measure --profile faq --only headline,traffic

Results go to artifacts/<profile>/eval/measurements.json and measurements.md.

Every report carries the same headline block: vocabulary size, k, ROC AUC, precision and
recall at the configured tau, and how many of 1,000 random Quora questions hit one of the
43 FAQ answers (almost every such hit is a wrong answer).

All the matrix work goes through semcache.linalg; this module only counts and times.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import linalg as la
from .cache import SemanticCache
from .config import Settings
from .data import FAQItem, QuestionPair, faq_corpus, faq_eval_pairs, load_faq
from .embedder import LSAEmbedder
from .evaluate import (TAUS, _json_safe, best_f1, pair_rho, pair_scores, roc_auc, sample_quora_questions,
                       tau_for_precision, tfidf_pair_scores, threshold_sweep)
from .text import exact_key

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def cache_for(settings: Settings, embedder: LSAEmbedder, threshold: float | None = None,
              **overrides) -> SemanticCache:
    """An empty cache configured exactly like the served one (every flag of [cache])."""
    from .pipeline import cache_params           # pipeline imports this package's peers
    params = cache_params(settings)
    params["max_entries"] = max(params["max_entries"], 10**7)
    if threshold is not None:
        params["threshold"] = threshold
    params.update(overrides)
    return SemanticCache(embedder.dim, model_fingerprint=embedder.fingerprint, **params)


def fill(cache: SemanticCache, embedder: LSAEmbedder, texts: list[str], ratio: float,
         source: str = "faq") -> list[int]:
    vectors, rows, rho = embedder.encode_with_rho(texts, ratio)
    return [cache.add(t, "", vectors[i], source=source, tfidf_row=rows[i], rho=float(rho[i])).id
            for i, t in enumerate(texts)]


def ask(cache: SemanticCache, embedder: LSAEmbedder, text: str, ratio: float):
    vectors, rows, rho = embedder.encode_with_rho([text], ratio)
    return cache.lookup(vectors[0], 1, record=False, query_text=text, query_tfidf=rows[0],
                        query_rho=float(rho[0]))


def decision_pair_scores(settings: Settings, embedder: LSAEmbedder, pairs: list[QuestionPair]):
    """The score the cache's hit decision uses for each pair: guarded hybrid score, and a
    pair the text guard (negation / antonym / number / entity) refuses scores -1."""
    lsa = pair_scores(embedder, pairs)
    lexical = tfidf_pair_scores(embedder, pairs)
    rho = pair_rho(embedder, pairs)
    scores, _ = la.guarded_scores(lsa, lexical, rho, settings.cache.lsa_weight,
                                  settings.cache.min_latent_coverage)
    refused = np.zeros(len(pairs), dtype=bool)
    if getattr(settings.cache, "negation_guard", False):
        from .guards import TextGuard
        guard = TextGuard.from_settings(settings.cache)
        refused = np.array([guard.check(p.q2, p.q1) is not None for p in pairs], dtype=bool)
        scores = np.where(refused, -1.0, scores)
    return scores, lsa, lexical, refused


def random_false_hits(settings: Settings, embedder: LSAEmbedder, n: int = 1000,
                      show: int = 5) -> dict | None:
    """Ask n random Quora validation questions against a cache holding only the FAQ
    answers (configured like the served cache). Almost every hit is a wrong answer.
    None when the Quora validation file is missing."""
    if not settings.path(settings.paths.quora_eval).exists():
        return None
    questions = sample_quora_questions(settings, n)
    items = load_faq(settings.path(settings.paths.faq_csv))
    ratio = settings.cache.min_known_ratio
    cache = cache_for(settings, embedder)
    fill(cache, embedder, [i.question for i in items], ratio)
    qv, qrows, qrho = embedder.encode_with_rho(questions, ratio)
    hits = []
    for i, text in enumerate(questions):
        r = cache.lookup(qv[i], 1, record=False, query_text=text, query_tfidf=qrows[i],
                         query_rho=float(qrho[i]))
        if r.hit:
            hits.append({"question": text, "matched": r.best.query, "score": round(r.score, 4)})
    hits.sort(key=lambda h: -h["score"])
    return {"asked": len(questions), "hits": len(hits), "examples": hits[:show]}


# ---------------------------------------------------------------------------
# 1. headline numbers
# ---------------------------------------------------------------------------

def headline(settings: Settings, embedder: LSAEmbedder | None = None,
             pairs: list[QuestionPair] | None = None, with_random: bool = True) -> dict:
    """Vocabulary, k, AUC (decision score, LSA only, TF-IDF only), best F1, precision and
    recall at the configured tau, accepted hits that were right, and random false hits."""
    if embedder is None:
        from .pipeline import load_embedder
        embedder = load_embedder(settings)
    if pairs is None:
        pairs = faq_eval_pairs(load_faq(settings.path(settings.paths.faq_csv)), seed=settings.eval.seed)
    labels = np.array([p.label for p in pairs])
    scores, lsa, lexical, refused = decision_pair_scores(settings, embedder, pairs)
    tau = settings.cache.threshold
    pred = scores >= tau
    tp, fp = int(np.sum(pred & (labels == 1))), int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    sweep = threshold_sweep(scores, labels)
    best = best_f1(sweep)
    out = {
        "profile": settings.profile,
        "vocabulary_size": len(embedder.vocab), "k": embedder.dim,
        "training_documents": embedder.report.n_docs,
        "pairs": len(pairs), "positives": int(labels.sum()),
        "auc": roc_auc(scores, labels), "auc_lsa_only": roc_auc(lsa, labels),
        "auc_tfidf_only": roc_auc(lexical, labels),
        "tau": tau, "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "accepted_hits": tp + fp, "accepted_hits_correct": tp,
        "best_f1": best["f1"], "best_f1_tau": best["tau"],
        "best_f1_precision": best["precision"], "best_f1_recall": best["recall"],
        "pairs_refused_by_text_guard": int(refused.sum()),
    }
    if with_random:
        rnd = random_false_hits(settings, embedder)
        out["random_false_hits"] = None if rnd is None else rnd["hits"]
        out["random_asked"] = None if rnd is None else rnd["asked"]
        out["random_examples"] = None if rnd is None else rnd["examples"]
    return out


def seed_sensitivity(settings: Settings, embedder: LSAEmbedder, seeds=range(12)) -> list[dict]:
    """AUC for different random negatives: the FAQ pairs are 86 positives and 86 sampled
    negatives, so the AUC moves with the sampling seed."""
    items = load_faq(settings.path(settings.paths.faq_csv))
    rows = []
    for seed in seeds:
        pairs = faq_eval_pairs(items, seed=seed)
        labels = np.array([p.label for p in pairs])
        scores, lsa, lexical, _ = decision_pair_scores(settings, embedder, pairs)
        rows.append({"seed": seed, "auc": roc_auc(scores, labels), "auc_lsa_only": roc_auc(lsa, labels),
                     "auc_tfidf_only": roc_auc(lexical, labels)})
    return rows


def headline_line(h: dict) -> str:
    rnd = ("n/a (no Quora validation file)" if h.get("random_false_hits") is None
           else f"{h['random_false_hits']} of {h['random_asked']:,}")
    return (f"vocabulary {h['vocabulary_size']:,} terms, k = {h['k']}, AUC {h['auc']:.3f} "
            f"(LSA only {h['auc_lsa_only']:.3f}, TF-IDF only {h['auc_tfidf_only']:.3f}), "
            f"at tau {h['tau']:.2f}: precision {h['precision']:.3f}, recall {h['recall']:.3f} "
            f"({h['accepted_hits_correct']} of {h['accepted_hits']} accepted hits correct), "
            f"random-question false hits {rnd}")


def headline_table(rows: list[tuple[str, dict]]) -> str:
    lines = ["| | vocabulary | k | AUC | AUC LSA only | AUC TF-IDF only | P @ tau | R @ tau | "
             "accepted hits correct | best F1 (tau) | random false hits / 1,000 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, h in rows:
        rnd = "n/a" if h.get("random_false_hits") is None else str(h["random_false_hits"])
        lines.append(f"| {name} | {h['vocabulary_size']:,} | {h['k']} | {h['auc']:.3f} | "
                     f"{h['auc_lsa_only']:.3f} | {h['auc_tfidf_only']:.3f} | {h['precision']:.3f} | "
                     f"{h['recall']:.3f} | {h['accepted_hits_correct']} of {h['accepted_hits']} | "
                     f"{h['best_f1']:.3f} ({h['best_f1_tau']:.2f}) | {rnd} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. traffic-mix simulation
# ---------------------------------------------------------------------------

def traffic_simulation(settings: Settings, embedder: LSAEmbedder, mix: tuple[float, float, float] | None = None,
                       n_queries: int | None = None, cached_share: float | None = None,
                       llm_usd_per_call: float | None = None, cpu_usd_per_hour: float | None = None,
                       seed: int | None = None) -> dict:
    """A realistic stream of questions against a cache of FAQ answers.

    The FAQ items are split: ``cached_share`` of them are in the cache. Queries are drawn
    with replacement from three pools:
      exact repeat   a cached FAQ question, verbatim or lower-cased without punctuation
      rewording      a held-out phrasing of a cached item (never trained on)
      new            a held-out phrasing of an item that is NOT cached (so any hit is wrong)
    h = hits / queries. Cost per query  E[cost] = c_lookup + (1 - h) c_LLM, where c_lookup
    is the measured lookup time priced at ``cpu_usd_per_hour`` (an estimate) and c_LLM is
    the assumed price of one LLM call.
    """
    cfg = settings.eval
    mix = tuple(mix or cfg.traffic_mix)
    if len(mix) != 3 or abs(sum(mix) - 1.0) > 1e-6 or min(mix) < 0:
        raise ValueError(f"traffic mix must be three shares that add up to 1, got {mix}")
    n_queries = n_queries or cfg.traffic_queries
    cached_share = cfg.traffic_cached_share if cached_share is None else cached_share
    c_llm = cfg.llm_usd_per_call if llm_usd_per_call is None else llm_usd_per_call
    cpu = cfg.cpu_usd_per_hour if cpu_usd_per_hour is None else cpu_usd_per_hour
    rng = random.Random(cfg.seed if seed is None else seed)

    items = [i for i in load_faq(settings.path(settings.paths.faq_csv)) if i.heldout]
    order = items[:]
    rng.shuffle(order)
    n_cached = max(1, min(len(order) - 1, round(cached_share * len(order))))
    cached, uncached = order[:n_cached], order[n_cached:]
    ratio = settings.cache.min_known_ratio
    cache = cache_for(settings, embedder)
    ids = fill(cache, embedder, [i.question for i in cached], ratio)
    id_of_item = {item.id: entry_id for item, entry_id in zip(cached, ids)}

    exact_pool = [(i.question, i.id) for i in cached] + [(exact_key(i.question), i.id) for i in cached]
    reword_pool = [(h, i.id) for i in cached for h in i.heldout]
    new_pool = [(h, None) for i in uncached for h in i.heldout]
    kinds = rng.choices(["exact", "reword", "new"], weights=mix, k=n_queries)
    stream = [(k, *rng.choice({"exact": exact_pool, "reword": reword_pool, "new": new_pool}[k])) for k in kinds]

    # Unique questions are scored once; the stream reuses the outcome.
    outcome: dict[str, tuple[bool, int | None, float, str]] = {}
    t0 = time.perf_counter()
    lookups = 0
    for text in dict.fromkeys(q for _, q, _ in stream):
        r = ask(cache, embedder, text, ratio)
        lookups += 1
        outcome[text] = (r.hit, r.best.id if r.best else None, r.score, r.method)
    lookup_s = (time.perf_counter() - t0) / max(1, lookups)

    counts = {k: {"queries": 0, "hits": 0, "right": 0, "wrong": 0} for k in ("exact", "reword", "new")}
    for kind, text, item_id in stream:
        hit, got, _, _ = outcome[text]
        c = counts[kind]
        c["queries"] += 1
        if hit:
            c["hits"] += 1
            right = item_id is not None and got == id_of_item.get(item_id)
            c["right" if right else "wrong"] += 1
    hits = sum(c["hits"] for c in counts.values())
    wrong = sum(c["wrong"] for c in counts.values())
    h = hits / n_queries
    c_lookup = lookup_s * cpu / 3600.0
    per_query = c_lookup + (1 - h) * c_llm
    return {
        "mix": {"exact": mix[0], "reword": mix[1], "new": mix[2]}, "queries": n_queries,
        "cached_items": len(cached), "uncached_items": len(uncached),
        "pools": {"exact": len(exact_pool), "reword": len(reword_pool), "new": len(new_pool)},
        "by_kind": counts, "hit_rate": h, "hits": hits, "wrong_hits": wrong,
        "precision_of_hits": (hits - wrong) / hits if hits else float("nan"),
        "lookup_ms": lookup_s * 1000,
        "assumed": {"llm_usd_per_call": c_llm, "cpu_usd_per_hour": cpu},
        "c_lookup_usd": c_lookup,
        "cost_per_million_usd": per_query * 1e6,
        "cost_per_million_without_cache_usd": c_llm * 1e6,
        "savings_per_million_usd": (c_llm - per_query) * 1e6,
    }


# ---------------------------------------------------------------------------
# 3. choosing tau on one part, testing on another
# ---------------------------------------------------------------------------

def tau_cross_validation(settings: Settings, embedder: LSAEmbedder, folds: int = 5,
                         targets=(0.95, 0.99), seed: int | None = None) -> dict:
    """Group k-fold over FAQ items (all pairs of one FAQ question stay in one fold, so
    no item is both tuned and tested on). Rules for picking tau on the tune folds:
    best F1, and the lowest tau whose tune precision reaches each target. The fixed
    tau = 0.70 is reported on the same test folds for comparison."""
    items = load_faq(settings.path(settings.paths.faq_csv))
    pairs = faq_eval_pairs(items, seed=settings.eval.seed)
    scores, _, _, _ = decision_pair_scores(settings, embedder, pairs)
    labels = np.array([p.label for p in pairs]).astype(bool)
    groups = sorted({p.q1 for p in pairs})
    random.Random(settings.eval.seed if seed is None else seed).shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}
    fold = np.array([fold_of[p.q1] for p in pairs])

    rules = {"best F1": lambda sw: best_f1(sw)["tau"]}
    for t in targets:
        rules[f"precision >= {t:.2f}"] = (lambda sw, t=t: (tau_for_precision(sw, t) or {"tau": 0.99})["tau"])
    rules["fixed 0.70"] = lambda sw: settings.cache.threshold

    per_fold = []
    pooled = {name: {"tp": 0, "fp": 0, "fn": 0, "taus": []} for name in rules}
    for f in range(folds):
        tune, test = fold != f, fold == f
        sweep = threshold_sweep(scores[tune], labels[tune])
        row = {"fold": f + 1, "tune_pairs": int(tune.sum()), "test_pairs": int(test.sum())}
        for name, rule in rules.items():
            tau = float(rule(sweep))
            pred = scores[test] >= tau
            tp = int(np.sum(pred & labels[test]))
            fp = int(np.sum(pred & ~labels[test]))
            fn = int(np.sum(~pred & labels[test]))
            p = pooled[name]
            p["tp"] += tp
            p["fp"] += fp
            p["fn"] += fn
            p["taus"].append(tau)
            row[name] = {"tau": tau, "precision": tp / (tp + fp) if tp + fp else float("nan"),
                         "recall": tp / (tp + fn) if tp + fn else float("nan")}
        per_fold.append(row)
    summary = {}
    for name, p in pooled.items():
        tp, fp, fn = p["tp"], p["fp"], p["fn"]
        summary[name] = {"taus": p["taus"], "tp": tp, "fp": fp, "fn": fn,
                         "precision": tp / (tp + fp) if tp + fp else float("nan"),
                         "recall": tp / (tp + fn) if tp + fn else float("nan")}
    return {"folds": folds, "pairs": len(pairs), "groups": len(groups), "per_fold": per_fold,
            "pooled": summary}


# ---------------------------------------------------------------------------
# 4. LSH sweep over bits per table (b) and tables (L)
# ---------------------------------------------------------------------------

def lsh_sweep(settings: Settings, sizes=(10000, 50000), bits=(6, 8, 10, 12, 14, 16),
              tables=(4, 8, 12, 16, 24, 32), n_queries: int = 400, tau: float | None = None,
              recall_target: float = 0.95, seed: int = 0, verbose: bool = False) -> dict:
    """Recall of the true best match (brute force) and lookup time for every (b, L).

    Vectors: the Quora profile's model (k = 200) embedding Quora validation questions,
    because a 50,000-entry cache needs 50,000 real questions; the FAQ has 172 texts.
    Cached: ``size`` questions. Asked: partners of cached questions in labelled pairs,
    never cached themselves. Recall counts only queries whose brute-force best score
    is >= tau (the lookups that would be hits), and asks whether LSH returns that same
    best entry. Time: mean wall-clock ms per SemanticCache.lookup, brute vs LSH.
    """
    from .data import load_question_pairs
    tau = settings.cache.threshold if tau is None else tau
    q_settings = Settings.load(profile="quora") if "quora" in settings.profiles else None
    if q_settings is None or not LSAEmbedder.exists(q_settings.model_dir) \
            or not settings.path(settings.paths.quora_eval).exists():
        return {"skipped": "needs the quora profile's built model and data/quora/validation.csv"}
    emb = LSAEmbedder.load(q_settings.model_dir)
    pairs = load_question_pairs(settings.path(settings.paths.quora_eval))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    partner: dict[str, str] = {}
    for p in pairs:
        partner.setdefault(p.q1, p.q2)
    questions = list(dict.fromkeys(q for p in pairs for q in (p.q1, p.q2)))
    asked_pool = [q for q in partner if partner[q] != q]
    t0 = time.perf_counter()
    vec_of_all = emb.embed(questions)
    index = {q: i for i, q in enumerate(questions)}
    if verbose:
        print(f"  embedded {len(questions):,} Quora questions with k = {emb.dim} "
              f"({time.perf_counter() - t0:.1f}s)")
    rows = []
    for size in sizes:
        # queries: questions whose partner is cached; cached: partners + filler, never a query
        queries = asked_pool[: n_queries * 3]
        cached_set = dict.fromkeys(partner[q] for q in queries)
        query_set = set(queries)
        for q in questions:
            if len(cached_set) >= size:
                break
            if q not in query_set:
                cached_set.setdefault(q, None)
        cached = list(cached_set)[:size]
        V = vec_of_all[[index[q] for q in cached]]
        keep = np.any(V, axis=1)
        cached = [q for q, k in zip(cached, keep) if k]
        V = V[keep]
        Q = vec_of_all[[index[q] for q in queries]]
        Q = Q[np.any(Q, axis=1)]
        cache = SemanticCache(emb.dim, threshold=1.0, max_entries=len(cached) + 1)
        for i, q in enumerate(cached):
            cache.add(q, "", V[i])
        # brute force once: the truth, and its time
        t0 = time.perf_counter()
        truth = [cache.lookup(q, 1, record=False) for q in Q]
        brute_ms = (time.perf_counter() - t0) * 1000 / len(Q)
        relevant = [i for i, r in enumerate(truth) if r.matches and r.matches[0].score >= tau]
        relevant = relevant[:n_queries]
        Qr = Q[relevant]
        best_ids = [truth[i].matches[0].entry.id for i in relevant]
        best_scores = np.array([truth[i].matches[0].score for i in relevant])
        t0 = time.perf_counter()
        for q in Qr:
            cache.lookup(q, 1, record=False)
        brute_rel_ms = (time.perf_counter() - t0) * 1000 / max(1, len(Qr))
        vecs = np.array(cache.vectors)
        for b in bits:
            for L in tables:
                lsh = la.RandomHyperplaneLSH(emb.dim, b, L, seed)
                lsh.add(range(len(cache)), vecs)         # the cache keys LSH by row of C
                cache.lsh, cache.lsh_min_entries = lsh, 0
                t0 = time.perf_counter()
                got = [cache.lookup(q, 1, record=False) for q in Qr]
                ms = (time.perf_counter() - t0) * 1000 / max(1, len(Qr))
                same = [bool(r.matches) and r.matches[0].entry.id == want for r, want in zip(got, best_ids)]
                rows.append({"size": len(cached), "b": b, "L": L, "queries": len(Qr),
                             "recall": float(np.mean(same)) if same else float("nan"),
                             "ms": ms, "brute_ms": brute_rel_ms, "speedup": brute_rel_ms / ms if ms else float("nan"),
                             "candidates": float(np.mean([r.compared for r in got])),
                             "p_found_at_tau": float(la.lsh_collision_probability(tau, b, L)),
                             "p_found_at_median_best": float(la.lsh_collision_probability(
                                 float(np.median(best_scores)), b, L))})
                cache.lsh = None
        if verbose:
            print(f"  size {len(cached):,}: brute {brute_ms:.2f} ms/lookup, {len(Qr)} queries with a match "
                  f">= {tau:.2f}")
    # recommendation: recall >= target at every size, then the fastest at the largest size,
    # then fewer tables (less memory)
    largest = max(r["size"] for r in rows)
    ok = []
    for b in bits:
        for L in tables:
            mine = [r for r in rows if r["b"] == b and r["L"] == L]
            if mine and all(r["recall"] >= recall_target for r in mine):
                big = next(r for r in mine if r["size"] == largest)
                ok.append((big["ms"], L, b, big))
    rec = None
    if ok:
        ms, L, b, big = min(ok)
        rec = {"b": b, "L": L, "ms_at_largest": ms, "recall_at_largest": big["recall"],
               "speedup_at_largest": big["speedup"]}
    return {"tau": tau, "model": "quora profile (k = %d, %d terms)" % (emb.dim, len(emb.vocab)),
            "recall_target": recall_target, "rows": rows, "recommended": rec,
            "previous_default": {"b": 12, "L": 8}}


# ---------------------------------------------------------------------------
# 5. fact checks on the model
# ---------------------------------------------------------------------------

def _term_cos(emb: LSAEmbedder, a: str, b: str) -> dict:
    out = {"a": a, "b": b, "a_in_vocab": a in emb.vocab, "b_in_vocab": b in emb.vocab, "cosine": None}
    if out["a_in_vocab"] and out["b_in_vocab"]:
        ra, rb = emb.V[emb.vocab.index[a]], emb.V[emb.vocab.index[b]]
        na, nb = float(np.sqrt(ra @ ra)), float(np.sqrt(rb @ rb))
        out.update(cosine=float(ra @ rb) / (na * nb) if na and nb else 0.0, norm_a=na, norm_b=nb)
    return out


def training_matrix(settings: Settings, embedder: LSAEmbedder, corpus: list[str] | None = None):
    """The TF-IDF matrix the served model was fitted on, rebuilt from the corpus with the
    model's own vocabulary and IDF (latent stopwords zeroed, as in the fit)."""
    from .pipeline import build_corpus
    if corpus is None:
        corpus, _ = build_corpus(settings)
    X = embedder.tfidf([t for t in corpus if t and t.strip()])
    keep = embedder.svd_mask
    return la.zero_columns(X, keep) if not keep.all() else X


def eckart_young(settings: Settings, embedder: LSAEmbedder, X=None, dense_limit: int = 5_000_000) -> dict:
    """||X - X_k||_F^2 against the sum of the dropped sigma_i^2, and the energy kept.

    Small matrices: X_k = U_k Sigma_k V_k^T is formed explicitly (U_k = X V_k / sigma) and
    the residual is summed entry by entry; the dropped sigmas come from a separate full
    SVD (linalg.exact_svd), so the two sides are computed independently. Large matrices:
    the residual uses ||X||^2 - ||X V_k||^2 (V_k orthonormal), never a dense copy.
    """
    X = training_matrix(settings, embedder) if X is None else X
    V, s = embedder.V, embedder.s
    k = len(s)
    total = la.frobenius_norm_sq(X)
    kept = float(np.sum(s ** 2))
    n, m = X.shape
    out = {"shape": [n, m], "k": k, "frobenius_sq": total, "energy_kept": kept / total if total else 0.0}
    if n * m <= dense_limit:
        D = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        U = (D @ V) / s
        resid = D - (U * s) @ V.T
        out["residual_sq"] = float(np.einsum("ij,ij->", resid, resid))
        out["residual_method"] = "explicit: X_k = U_k Sigma_k V_k^T formed and subtracted"
        _, s_all, _ = la.exact_svd(D)
        out["sigma_all"] = s_all.tolist()
        out["dropped_sigma_sq"] = float(np.sum(s_all[k:] ** 2))
        out["dropped_method"] = "independent full SVD (linalg.exact_svd, Jacobi on the Gram matrix)"
        out["sigma_match_max_abs"] = float(np.max(np.abs(s_all[:k] - s)))
    else:
        XV = np.asarray(X @ V)
        out["residual_sq"] = total - float(np.einsum("ij,ij->", XV, XV))
        out["residual_method"] = "||X||_F^2 - ||X V_k||_F^2 (no dense copy)"
        out["dropped_sigma_sq"] = total - kept
        out["dropped_method"] = "||X||_F^2 - sum of kept sigma_i^2 (the full spectrum is too big)"
    out["difference"] = abs(out["residual_sq"] - out["dropped_sigma_sq"])
    out["relative_difference"] = out["difference"] / max(out["dropped_sigma_sq"], 1e-300)
    return out


def fact_checks(settings: Settings, embedder: LSAEmbedder) -> dict:
    ratio = settings.cache.min_known_ratio
    on, off = "How do I turn on notifications?", "How do I turn off notifications?"
    Z, X, rho = embedder.encode_with_rho([on, off], ratio)
    lexical = float(la.rowwise_cosine(X[0], X[1])[0])
    lsa = float(Z[0] @ Z[1])
    scores, _ = la.guarded_scores(np.array([lsa]), np.array([lexical]), np.array([min(rho)]),
                                  settings.cache.lsa_weight, settings.cache.min_latent_coverage)
    goa = "cheap flight tickets to goa"
    _, _, rho_goa = embedder.encode_with_rho([goa], ratio)
    ey = eckart_young(settings, embedder)
    ey.pop("sigma_all", None)
    return {
        "on_off": {"a": on, "b": off, "lsa_cosine": lsa, "tfidf_cosine": lexical,
                   "decision_score": float(scores[0]), "tau": settings.cache.threshold,
                   "would_hit": bool(scores[0] >= settings.cache.threshold),
                   "rho": [float(rho[0]), float(rho[1])],
                   "note": "both are FAQ questions, so in the served cache each is an exact hit of its own row"},
        "term_cosines": [_term_cos(embedder, "cheap", "budget"), _term_cos(embedder, "cheap", "low cost"),
                         _term_cos(embedder, "cheap", "low"), _term_cos(embedder, "cheap", "cost"),
                         _term_cos(embedder, "on", "off")],
        "rho_goa": {"query": goa, "rho": float(rho_goa[0])},
        "eckart_young": ey,
    }


# ---------------------------------------------------------------------------
# 6. latent stopwords on a growing profile
# ---------------------------------------------------------------------------

LATENT_WORDS = ["what", "how", "which", "best", "way", "good"]


def latent_stopword_study(settings: Settings, words: list[str] | None = None, directions: int = 5) -> dict:
    """Fit the profile's corpus (as it would be built now) with the latent stopwords off and
    on, and compare the headline numbers and the top words of the first SVD directions.
    The rule: keep them on only if precision and random false hits are no worse and AUC
    and recall are both at least as good, one of them better."""
    from .pipeline import build_corpus
    words = words or LATENT_WORDS
    entries = None
    if settings.corpus.include_cached and SemanticCache.exists(settings.cache_dir):
        entries = SemanticCache.load(settings.cache_dir).entries(newest_first=False)
    corpus, info = build_corpus(settings, entries=entries)
    rows = {}
    for name, on in (("off", False), ("on", True)):
        cfg = replace(settings.embedder, use_latent_stopwords=on, latent_stopwords=list(words))
        from .pipeline import configure_embedder
        emb = configure_embedder(LSAEmbedder(cfg).fit(corpus), settings)
        h = headline(settings, emb)
        h.pop("random_examples", None)
        rows[name] = {"headline": h, "directions": emb.direction_terms(directions, 6)}
    off, on = rows["off"]["headline"], rows["on"]["headline"]
    rnd_ok = (on.get("random_false_hits") or 0) <= (off.get("random_false_hits") or 0)
    better = (on["precision"] >= off["precision"] and rnd_ok and on["auc"] >= off["auc"]
              and on["recall"] >= off["recall"] and (on["auc"] > off["auc"] or on["recall"] > off["recall"]))
    return {"words": words, "corpus_texts": len(corpus), "cached_texts": info["cached_texts"],
            "off": rows["off"], "on": rows["on"], "keep_on": bool(better)}


# ---------------------------------------------------------------------------
# 7. incremental SVD (Brand) vs a full refit
# ---------------------------------------------------------------------------

def incremental_svd_study(settings: Settings, new_entries: int = 20, repeats: int = 3) -> dict:
    """Would Brand's row update be a faster, equally exact rebuild for faq-live?

    New rows: ``new_entries`` cached question + answer texts (FAQ held-out phrasings with
    their FAQ answers), the size of one automatic rebuild. Three comparisons:
      exact      Brand from the FULL SVD of X0, rows in the old vocabulary, vs the full SVD
                 of [X0; A] in that vocabulary: tests the algorithm itself
      truncated  Brand from the served truncated SVD (k), vs the truncated SVD of [X0; A]
      rebuild    Brand (old vocabulary and IDF) vs what a rebuild really fits: the new corpus
                 with its new vocabulary and IDF
    Adopted only if the rebuild comparison matches within 1e-6 and Brand is faster.
    """
    from .pipeline import build_corpus
    entries = SemanticCache.load(settings.cache_dir).entries(newest_first=False) \
        if SemanticCache.exists(settings.cache_dir) else None
    corpus0, _ = build_corpus(settings, entries=entries)
    emb0 = LSAEmbedder(settings.embedder).fit(corpus0)
    items = load_faq(settings.path(settings.paths.faq_csv))
    new_docs = [f"{h} {i.answer}" for i in items for h in i.heldout][:new_entries]
    X0 = emb0.tfidf(corpus0)
    A = emb0.tfidf(new_docs)                                   # old vocabulary and IDF
    k0 = emb0.dim
    D0 = X0.toarray()

    def timed(fn):
        best, out = float("inf"), None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return out, best

    # exact: from the full SVD of X0
    U, s, V = la.exact_svd(D0)
    (Ub, sb, Vb), t_brand_full = timed(lambda: la.brand_update_rows(U, s, V, A))
    _, sf, Vf = la.exact_svd(np.vstack([D0, A.toarray()]))
    n = min(len(sb), len(sf), k0)
    exact = {"sigma_max_abs_diff": float(np.max(np.abs(sb[:n] - sf[:n]))),
             "subspace_distance_top_k": la.subspace_distance(Vb[:, :n], Vf[:, :n]), "k": n,
             "seconds_brand": t_brand_full}
    # truncated: from the served rank-k SVD (U_k = X0 V_k / s)
    Uk = (D0 @ emb0.V) / emb0.s
    k1, _ = emb0.fit_rank(X0.shape[0] + A.shape[0], X0.shape[1])
    (Ut, st, Vt), t_brand = timed(lambda: la.brand_update_rows(Uk, emb0.s, emb0.V, A, k=k1))
    _, sft, Vft = la.exact_svd(np.vstack([D0, A.toarray()]), k1)
    truncated = {"sigma_max_abs_diff": float(np.max(np.abs(st - sft[:len(st)]))),
                 "subspace_distance": la.subspace_distance(Vt, Vft), "k": int(len(st)), "k_in": int(k0),
                 "seconds_brand": t_brand}
    # rebuild: the real path re-fits TF-IDF (new terms, new IDF) and the SVD
    corpus1 = corpus0 + new_docs
    emb1, t_full = timed(lambda: LSAEmbedder(settings.embedder).fit(corpus1))
    m = min(len(st), emb1.dim)
    common = [t for t in emb0.vocab.terms if t in emb1.vocab.index]
    i0 = np.array([emb0.vocab.index[t] for t in common])
    i1 = np.array([emb1.vocab.index[t] for t in common])
    rebuild = {"sigma_max_abs_diff": float(np.max(np.abs(st[:m] - emb1.s[:m]))),
               "new_terms": len(emb1.vocab) - len(common), "k_full": emb1.dim, "k_brand": int(len(st)),
               "subspace_distance_on_shared_terms": la.subspace_distance(
                   la.gram_schmidt_qr(Vt[i0, :m])[0], la.gram_schmidt_qr(emb1.V[i1, :m])[0]),
               "seconds_full_refit": t_full, "seconds_brand": t_brand}
    matches = rebuild["sigma_max_abs_diff"] <= 1e-6 and rebuild["subspace_distance_on_shared_terms"] <= 1e-6
    faster = t_brand < t_full
    return {"new_rows": len(new_docs), "X0": list(X0.shape), "exact": exact, "truncated": truncated,
            "rebuild": rebuild, "matches_full_refit": bool(matches), "faster": bool(faster),
            "adopted": bool(matches and faster)}


# ---------------------------------------------------------------------------
# 8. the known-word guard: plain rule vs the topic-word rule
# ---------------------------------------------------------------------------

def load_wordy(settings: Settings) -> list[tuple[str, str]]:
    path = settings.path("data/eval_wordy_paraphrases.csv")
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    import csv as _csv
    return [(r["question"], r["expected"]) for r in _csv.DictReader(lines)]


def known_word_guard_study(settings: Settings, embedder: LSAEmbedder) -> dict:
    """What the topic-word rule (cache.relaxed_known_words) buys and costs, against the plain
    known-word rule, on a cache of the FAQ answers: wordy rewordings answered right, FAQ
    held-out phrasings answered right, and random Quora questions that hit (each wrong)."""
    import copy
    from .pipeline import configure_embedder
    items = load_faq(settings.path(settings.paths.faq_csv))
    ratio = settings.cache.min_known_ratio
    streams = {"wordy": load_wordy(settings),
               "heldout": [(h, i.question) for i in items for h in i.heldout]}
    if settings.path(settings.paths.quora_eval).exists():
        streams["random"] = [(q, None) for q in sample_quora_questions(settings, 1000)]
    on = configure_embedder(embedder, settings) if embedder.policy is None else embedder
    off = copy.copy(on)
    off.policy, off._everyday_keep = None, None
    rows = []
    for name, emb in (("plain rule", off), ("topic-word rule", on)):
        if name == "topic-word rule" and on.policy is None:
            continue
        cache = cache_for(settings, emb)
        fill(cache, emb, [i.question for i in items], ratio)
        row = {"rule": name}
        for kind, stream in streams.items():
            if not stream:
                continue
            V, X, rho = emb.encode_with_rho([q for q, _ in stream], ratio)
            hits = right = relaxed = 0
            for i, (q, want) in enumerate(stream):
                relaxed += emb.admission(q, ratio).status == "relaxed"
                r = cache.lookup(V[i], 1, record=False, query_text=q, query_tfidf=X[i], query_rho=float(rho[i]))
                hits += r.hit
                right += bool(r.hit and want is not None and r.best.query == want)
            row[kind] = {"asked": len(stream), "hits": hits, "right": right, "wrong": hits - right,
                         "relaxed": relaxed}
        rows.append(row)
    p = on.policy
    return {"min_known_ratio": ratio, "policy": None if p is None else
            {"min_topic_words": p.min_topic_words, "min_topic_ratio": p.min_topic_ratio}, "rows": rows}


def guard_markdown(r: dict) -> str:
    p = r["policy"]
    lines = ["## Known-word guard: plain rule vs topic-word rule", "",
             f"Plain rule: at least {r['min_known_ratio']:.0%} of the content words known. Topic-word rule "
             + (f"(cache.relaxed_known_words): otherwise, unknown everyday words (where, go, one ...) are not counted, "
                f"and at least {p['min_topic_words']} known topic words making up {p['min_topic_ratio']:.0%} of the topic "
                "words admit the question, matched on those topic words only." if p else "is off for this profile."),
             "Wordy: data/eval_wordy_paraphrases.csv (hand-written, small). Random: 1,000 Quora questions vs the "
             "43 FAQ answers (every hit is wrong).", "",
             "| rule | wordy right / asked (wrong) | held-out right / asked (wrong) | random false hits / 1,000 | relaxed admissions (wordy, held-out, random) |",
             "|---|---:|---:|---:|---:|"]
    for row in r["rows"]:
        w, h, rnd = row.get("wordy"), row["heldout"], row.get("random")
        lines.append(f"| {row['rule']} | " + (f"{w['right']} / {w['asked']} ({w['wrong']})" if w else "n/a") +
                     f" | {h['right']} / {h['asked']} ({h['wrong']}) | " + (str(rnd["hits"]) if rnd else "n/a") +
                     f" | {w['relaxed'] if w else 0}, {h['relaxed']}, {rnd['relaxed'] if rnd else 0} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

SECTIONS = ("headline", "traffic", "tau_cv", "lsh", "facts", "latent", "incremental", "thresholds", "neural",
            "guard")


def run_measurements(settings: Settings, only: list[str] | None = None, out_dir: str | Path | None = None,
                     verbose: bool = True) -> dict:
    only = list(only or SECTIONS)
    unknown = set(only) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown sections {sorted(unknown)}; choose from {', '.join(SECTIONS)}")
    from .pipeline import load_embedder
    embedder = load_embedder(settings)
    out = Path(out_dir) if out_dir else settings.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "measurements.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["profile"] = settings.profile
    t0 = time.perf_counter()
    if "headline" in only:
        report["headline"] = headline(settings, embedder)
        report["seed_sensitivity"] = seed_sensitivity(settings, embedder)
        if verbose:
            print("  " + headline_line(report["headline"]))
    if "traffic" in only:
        report["traffic"] = traffic_simulation(settings, embedder)
        if verbose:
            t = report["traffic"]
            print(f"  traffic: h = {t['hit_rate']:.3f}, E[cost] ${t['cost_per_million_usd']:,.2f} per million "
                  f"(estimate)")
    if "tau_cv" in only:
        report["tau_cv"] = tau_cross_validation(settings, embedder)
    if "lsh" in only:
        report["lsh"] = lsh_sweep(settings, verbose=verbose)
    if "facts" in only:
        report["facts"] = fact_checks(settings, embedder)
    if "latent" in only:
        report["latent"] = latent_stopword_study(settings)
    if "incremental" in only:
        report["incremental"] = incremental_svd_study(settings)
    if "guard" in only:
        report["guard"] = known_word_guard_study(settings, embedder)
    if "thresholds" in only:
        from .thresholds import per_entry_threshold_study
        report["thresholds"] = per_entry_threshold_study(settings, embedder)
    if "neural" in only:
        from .neural import available, neural_comparison
        report["neural"] = (neural_comparison(settings, embedder, verbose) if available() else
                            {"skipped": "optional packages missing: pip install -r requirements-neural.txt"})
    report["seconds"] = time.perf_counter() - t0
    path.write_text(json.dumps(_json_safe(report), indent=1, allow_nan=False), encoding="utf-8")
    (out / "measurements.md").write_text(measurements_markdown(report), encoding="utf-8")
    if verbose:
        print(f"  report: {out / 'measurements.md'}")
    return report


def measurements_markdown(r: dict) -> str:
    md = [f"# Measurements: profile {r.get('profile')}", ""]
    h = r.get("headline")
    if h:
        md += ["## Headline", "", headline_table([(r["profile"], h)]), "",
               f"Pairs: {h['pairs']} FAQ held-out pairs ({h['positives']} positives). "
               f"Best F1 {h['best_f1']:.3f} at tau {h['best_f1_tau']:.2f} "
               f"(precision {h['best_f1_precision']:.3f}, recall {h['best_f1_recall']:.3f}).", ""]
        if h.get("random_examples"):
            md += ["Random-question hits (each is a wrong answer):", ""]
            md += [f"- {e['score']:.3f} \"{e['question']}\" → \"{e['matched']}\"" for e in h["random_examples"]]
            md.append("")
    ss = r.get("seed_sensitivity")
    if ss:
        a = np.array([x["auc_lsa_only"] for x in ss])
        b = np.array([x["auc_tfidf_only"] for x in ss])
        md += ["## AUC across negative-sampling seeds", "",
               "The 86 negatives are drawn at random, so the AUC depends on the seed. "
               f"LSA {a.mean():.3f} ± {a.std():.3f} (range {a.min():.3f}–{a.max():.3f}), TF-IDF {b.mean():.3f} ± "
               f"{b.std():.3f}; LSA is higher for {int(np.sum(a > b))} of {len(ss)} seeds.", "",
               "| seed | AUC (decision score) | LSA only | TF-IDF only |", "|---:|---:|---:|---:|"]
        md += [f"| {x['seed']} | {x['auc']:.4f} | {x['auc_lsa_only']:.4f} | {x['auc_tfidf_only']:.4f} |" for x in ss]
        md.append("")
    t = r.get("traffic")
    if t:
        md += ["## Traffic-mix simulation", "",
               f"Mix: {t['mix']['exact']:.0%} exact repeats, {t['mix']['reword']:.0%} rewordings, "
               f"{t['mix']['new']:.0%} new questions; {t['queries']:,} queries; {t['cached_items']} FAQ items "
               f"cached, {t['uncached_items']} held back (their held-out phrasings are the new questions).", "",
               "| kind | queries | hits | right | wrong |", "|---|---:|---:|---:|---:|"]
        for kind, c in t["by_kind"].items():
            md.append(f"| {kind} | {c['queries']} | {c['hits']} | {c['right']} | {c['wrong']} |")
        md += ["", f"- Hit rate h = **{t['hit_rate']:.3f}** ({t['wrong_hits']} wrong hits; precision of hits "
                   f"{t['precision_of_hits']:.3f})",
               f"- Measured lookup time {t['lookup_ms']:.3f} ms → c_lookup ≈ ${t['c_lookup_usd']:.2e} "
               f"(estimate at an assumed ${t['assumed']['cpu_usd_per_hour']}/CPU-hour)",
               f"- E[cost] = c_lookup + (1 − h) c_LLM with an assumed c_LLM = ${t['assumed']['llm_usd_per_call']} "
               f"per call: **${t['cost_per_million_usd']:,.2f} per million queries** (estimate), against "
               f"${t['cost_per_million_without_cache_usd']:,.2f} without the cache "
               f"(saves ${t['savings_per_million_usd']:,.2f}).", ""]
    cv = r.get("tau_cv")
    if cv:
        md += ["## Choosing tau on tune folds, testing on held-out folds", "",
               f"{cv['folds']}-fold cross-validation grouped by FAQ item ({cv['groups']} items, {cv['pairs']} pairs).", "",
               "| rule | taus chosen per fold | pooled test precision | pooled test recall | TP / FP / FN |",
               "|---|---|---:|---:|---|"]
        for name, s in cv["pooled"].items():
            md.append(f"| {name} | {', '.join(f'{x:.2f}' for x in s['taus'])} | {s['precision']:.3f} | "
                      f"{s['recall']:.3f} | {s['tp']} / {s['fp']} / {s['fn']} |")
        md.append("")
    lsh = r.get("lsh")
    if lsh and "rows" in lsh:
        md += ["## LSH sweep", "", f"Vectors: {lsh['model']}. Recall = share of lookups whose brute-force best "
               f"score is >= {lsh['tau']:.2f} that LSH answers with the same entry.", ""]
        for size in sorted({x["size"] for x in lsh["rows"]}):
            rows = [x for x in lsh["rows"] if x["size"] == size]
            md += [f"### {size:,} entries (brute force {rows[0]['brute_ms']:.2f} ms per lookup)", "",
                   "| b \\ L | " + " | ".join(str(L) for L in sorted({x['L'] for x in rows})) + " |",
                   "|---:|" + "---:|" * len({x['L'] for x in rows})]
            for b in sorted({x["b"] for x in rows}):
                cells = [f"{x['recall']:.2f} / {x['ms']:.2f} ms" for x in sorted(rows, key=lambda x: x["L"]) if x["b"] == b]
                md.append(f"| {b} | " + " | ".join(cells) + " |")
            md.append("")
        rec = lsh.get("recommended")
        md += [(f"Recommended (recall ≥ {lsh['recall_target']:.2f} at every size, then fastest at the largest): "
                f"**b = {rec['b']}, L = {rec['L']}** ({rec['ms_at_largest']:.2f} ms, recall {rec['recall_at_largest']:.3f}, "
                f"{rec['speedup_at_largest']:.1f}× faster than brute force)") if rec else
               "No (b, L) reached the recall target.", ""]
    elif lsh:
        md += ["## LSH sweep", "", f"Skipped: {lsh['skipped']}", ""]
    lat = r.get("latent")
    if lat:
        md += ["## Latent stopwords", "",
               f"Words left out of the SVD fit when on: {', '.join(lat['words'])}. Corpus: {lat['corpus_texts']} texts "
               f"({lat['cached_texts']} cached question + answer texts).", "",
               headline_table([("latent stopwords off", lat["off"]["headline"]),
                               ("latent stopwords on", lat["on"]["headline"])]), "",
               f"Decision: keep them **{'on' if lat['keep_on'] else 'off'}** (on only if precision and random false "
               "hits are no worse and AUC and recall are at least as good, one of them better).", "",
               "Top words of the first SVD directions:", "", "| direction | off | on |", "|---:|---|---|"]
        for a_, b_ in zip(lat["off"]["directions"], lat["on"]["directions"]):
            md.append(f"| {a_['direction']} | {', '.join(a_['terms'])} | {', '.join(b_['terms'])} |")
        md.append("")
    inc = r.get("incremental")
    if inc:
        e, t, rb = inc["exact"], inc["truncated"], inc["rebuild"]
        md += ["## Incremental SVD (Brand) vs a full refit", "",
               f"X0 is {inc['X0'][0]} × {inc['X0'][1]}; {inc['new_rows']} new cached question + answer rows "
               "(one automatic rebuild).", "",
               "| comparison | max abs σ difference | subspace distance (sin of largest angle) | time |",
               "|---|---:|---:|---:|",
               f"| exact: Brand from the full SVD vs full SVD (same vocabulary) | {e['sigma_max_abs_diff']:.1e} | "
               f"{e['subspace_distance_top_k']:.1e} (top {e['k']}) | {e['seconds_brand'] * 1000:.1f} ms |",
               f"| truncated: Brand from the served rank-{t['k_in']} SVD vs the rank-{t['k']} SVD (same vocabulary) | "
               f"{t['sigma_max_abs_diff']:.1e} | {t['subspace_distance']:.1e} | {t['seconds_brand'] * 1000:.1f} ms |",
               f"| rebuild: Brand vs the real refit ({rb['new_terms']} new terms, new IDF) | "
               f"{rb['sigma_max_abs_diff']:.1e} | {rb['subspace_distance_on_shared_terms']:.1e} (shared terms) | "
               f"Brand {rb['seconds_brand'] * 1000:.1f} ms vs refit {rb['seconds_full_refit'] * 1000:.1f} ms |", "",
               f"Adopted as the faq-live rebuild path: **{'yes' if inc['adopted'] else 'no'}** "
               f"(matches the full refit within 1e-6: {'yes' if inc['matches_full_refit'] else 'no'}; "
               f"faster: {'yes' if inc['faster'] else 'no'}).", ""]
    if r.get("guard"):
        md.append(guard_markdown(r["guard"]))
    th = r.get("thresholds")
    if th:
        from .thresholds import thresholds_markdown
        md.append(thresholds_markdown(th))
    nn = r.get("neural")
    if nn:
        if "skipped" in nn:
            md += ["## Neural baseline (MiniLM) vs LSA", "", f"Skipped: {nn['skipped']}", ""]
        else:
            from .neural import neural_markdown
            md.append(neural_markdown(nn))
    f = r.get("facts")
    if f:
        oo, ey = f["on_off"], f["eckart_young"]
        md += ["## Fact checks", "",
               f"- \"{oo['a']}\" vs \"{oo['b']}\": LSA cosine **{oo['lsa_cosine']:.3f}**, TF-IDF cosine "
               f"{oo['tfidf_cosine']:.3f}, decision score {oo['decision_score']:.3f} "
               f"({'above' if oo['would_hit'] else 'below'} tau {oo['tau']:.2f}). {oo['note']}."]
        for c in f["term_cosines"]:
            if c["cosine"] is None:
                missing = [w for w, ok in ((c['a'], c['a_in_vocab']), (c['b'], c['b_in_vocab'])) if not ok]
                md.append(f"- V_k rows \"{c['a']}\"–\"{c['b']}\": not computable, "
                          + ", ".join(f'"{w}"' for w in missing) + " not in the vocabulary")
            else:
                md.append(f"- V_k row cosine \"{c['a']}\"–\"{c['b']}\": **{c['cosine']:.3f}** "
                          f"(row lengths {c['norm_a']:.3f}, {c['norm_b']:.3f})")
        md += [f"- rho(\"{f['rho_goa']['query']}\") = **{f['rho_goa']['rho']:.3f}**",
               f"- Eckart–Young at k = {ey['k']} ({ey['shape'][0]} × {ey['shape'][1]}): ||X − X_k||_F² = "
               f"**{ey['residual_sq']:.6f}** ({ey['residual_method']}); sum of dropped σ_i² = "
               f"**{ey['dropped_sigma_sq']:.6f}** ({ey['dropped_method']}); difference {ey['difference']:.2e}. "
               f"Energy kept {ey['energy_kept']:.4f} ({ey['energy_kept']:.1%}).", ""]
    return "\n".join(md) + "\n"
