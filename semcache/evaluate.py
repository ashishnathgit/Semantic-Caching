"""Evaluation: how good is the cache, and what does it cost?

    threshold sweep   precision / recall / F1 / false-hit rate for every tau
    rank sweep        the same as k (number of SVD directions) grows, against a
                      TF-IDF-only baseline with no SVD
    blend sweep       LSA vs TF-IDF vs mixtures of the two (the cache's lsa_weight)
    cache simulation  fill a cache with question 1 of each pair, then ask question 2
    latency           lookup time against cache size, brute force vs LSH
    guard sweep       latent stopwords on/off x k x rho cutoff: Quora numbers, what the
                      guard costs the FAQ demo questions, and known false hits

Every score here is the one the cache uses, latent-coverage guard included: a pair
whose smaller rho is below min_latent_coverage is scored by word overlap (TF-IDF).

Run it with  python -m semcache evaluate [--guard-sweep]  after building the model.
Results go to artifacts/<profile>/eval/: report.md (tables), report.json, PNG charts,
and guard_sweep.md / guard_sweep.json.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import random
import time
from pathlib import Path

import numpy as np

from . import linalg as la
from .cache import SemanticCache
from .config import Settings
from .data import FAQItem, QuestionPair, faq_eval_pairs, load_faq, load_question_pairs
from .embedder import LSAEmbedder
from .text import exact_key

TAUS = np.round(np.arange(0.30, 0.991, 0.01), 2)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def pair_scores(embedder: LSAEmbedder, pairs: list[QuestionPair], rank: int | None = None,
                min_known_ratio: float = 0.0) -> np.ndarray:
    """Cosine similarity of each pair in the LSA space."""
    a = embedder.embed([p.q1 for p in pairs], rank, min_known_ratio)
    b = embedder.embed([p.q2 for p in pairs], rank, min_known_ratio)
    return np.einsum("ij,ij->i", a, b)


def tfidf_pair_scores(embedder: LSAEmbedder, pairs: list[QuestionPair]) -> np.ndarray:
    """Baseline without the SVD: cosine of the raw TF-IDF rows."""
    return la.rowwise_cosine(embedder.tfidf([p.q1 for p in pairs]), embedder.tfidf([p.q2 for p in pairs]))


def pair_rho(embedder: LSAEmbedder, pairs: list[QuestionPair]) -> np.ndarray:
    """min(rho(q1), rho(q2)) per pair: what the latent-coverage guard compares."""
    r1 = embedder.latent_coverage(embedder.tfidf([p.q1 for p in pairs]))
    r2 = embedder.latent_coverage(embedder.tfidf([p.q2 for p in pairs]))
    return np.minimum(r1, r2)


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, taus: np.ndarray = TAUS) -> list[dict]:
    """Treat "cos >= tau" as predicting "duplicate" and score it against the labels."""
    labels = np.asarray(labels).astype(bool)
    rows = []
    for tau in taus:
        pred = scores >= tau
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        tn = int(np.sum(~pred & ~labels))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"tau": float(tau), "precision": precision, "recall": recall, "f1": f1,
                     "false_hit_rate": fp / (fp + tn) if fp + tn else 0.0,
                     "accuracy": (tp + tn) / len(labels)})
    return rows


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: the chance a random duplicate pair scores higher than a
    random non-duplicate pair (Mann-Whitney U with average ranks for ties)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_rank = ends - (counts - 1) / 2.0
    ranks = np.empty(len(scores))
    ranks[order] = average_rank[inverse]
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def best_f1(sweep: list[dict]) -> dict:
    return max(sweep, key=lambda r: (r["f1"], r["tau"]))


def tau_for_precision(sweep: list[dict], target: float = 0.95) -> dict | None:
    """Lowest tau whose precision reaches the target: wrong answers are costly in a cache."""
    ok = [r for r in sweep if r["precision"] >= target and r["recall"] > 0]
    return min(ok, key=lambda r: r["tau"]) if ok else None


def rank_sweep(embedder: LSAEmbedder, pairs: list[QuestionPair], ranks: list[int]) -> list[dict]:
    """Because the SVD is nested, the top-k directions of one fit give every smaller k:
    the TF-IDF rows are computed once and projected onto V[:, :k] for each k."""
    labels = np.array([p.label for p in pairs])
    X1 = embedder.tfidf([p.q1 for p in pairs])
    X2 = embedder.tfidf([p.q2 for p in pairs])
    power = embedder.config.sigma_power
    rows = []
    for k in sorted({min(r, embedder.dim) for r in ranks}):
        V, s = embedder.V[:, :k], embedder.s[:k]
        a = la.l2_normalize_rows(la.project(X1, V, s, power))
        b = la.l2_normalize_rows(la.project(X2, V, s, power))
        scores = np.einsum("ij,ij->i", a, b)
        best = best_f1(threshold_sweep(scores, labels))
        rows.append({"k": int(k), "auc": roc_auc(scores, labels), "best_f1": best["f1"],
                     "best_tau": best["tau"],
                     "energy_kept": float(embedder.energy_profile()[:k].sum())})
    return rows


def blend_sweep(lsa: np.ndarray, lexical: np.ndarray, labels: np.ndarray,
                weights=(0.0, 0.25, 0.5, 0.75, 1.0), rho: np.ndarray | None = None,
                min_coverage: float = 0.0) -> list[dict]:
    """Score w * cos_LSA + (1 - w) * cos_TFIDF for several w (w = 1 is pure LSA), with
    the latent-coverage guard applied when ``rho`` (per pair) is given."""
    rows = []
    rho = np.ones(len(lsa)) if rho is None else rho
    for w in weights:
        scores, _ = la.guarded_scores(lsa, lexical, rho, w, min_coverage)
        best = best_f1(threshold_sweep(scores, labels))
        rows.append({"lsa_weight": float(w), "auc": roc_auc(scores, labels),
                     "best_f1": best["f1"], "best_tau": best["tau"]})
    return rows


def cache_simulation(embedder: LSAEmbedder, pairs: list[QuestionPair], min_known_ratio: float,
                     taus: np.ndarray = TAUS, lsa_weight: float = 1.0,
                     min_latent_coverage: float = 0.0) -> list[dict]:
    """Realistic use: every question 1 is cached; every question 2 is asked.

    The labels only tell us about each question's own partner, so outcomes are split:
      duplicate      -> hit on its partner (right), hit on another entry, or miss
      non-duplicate  -> hit on its partner (certainly a wrong answer), hit on another
                        entry (unknown: it may be a real duplicate of that entry), or miss
    """
    firsts = list(dict.fromkeys(p.q1 for p in pairs))
    cache = SemanticCache(embedder.dim, threshold=1.0, max_entries=len(firsts) + 1,
                          lsa_weight=lsa_weight, min_latent_coverage=min_latent_coverage)
    vectors, rows, rhos = embedder.encode_with_rho(firsts, min_known_ratio)
    for i, (q, v) in enumerate(zip(firsts, vectors)):
        cache.add(q, "", v, tfidf_row=rows[i], rho=float(rhos[i]))
    # Key by exact_key: questions differing only in case/punctuation share one entry.
    id_of = {exact_key(e.query): e.id for e in cache.entries()}
    queries, query_rows, query_rho = embedder.encode_with_rho([p.q2 for p in pairs], min_known_ratio)

    best_score = np.zeros(len(pairs))
    best_is_partner = np.zeros(len(pairs), dtype=bool)
    for i, (p, v) in enumerate(zip(pairs, queries)):
        r = cache.lookup(v, top_k=1, record=False, query_text=p.q2, query_tfidf=query_rows[i],
                         query_rho=float(query_rho[i]))
        if r.matches:
            best_score[i] = r.matches[0].score
            best_is_partner[i] = r.matches[0].entry.id == id_of.get(exact_key(p.q1))
    labels = np.array([p.label for p in pairs]).astype(bool)

    def share(mask: np.ndarray, group: np.ndarray) -> float:
        return float(np.mean(mask[group])) if group.any() else 0.0

    rows = []
    for tau in taus:
        hit = best_score >= tau
        rows.append({
            "tau": float(tau),
            "duplicate_hit_partner": share(hit & best_is_partner, labels),
            "duplicate_hit_other": share(hit & ~best_is_partner, labels),
            "duplicate_miss": share(~hit, labels),
            "non_duplicate_hit_partner": share(hit & best_is_partner, ~labels),
            "non_duplicate_hit_other": share(hit & ~best_is_partner, ~labels),
            "non_duplicate_miss": share(~hit, ~labels),
        })
    return rows


def latency_benchmark(vectors: np.ndarray, sizes: list[int], lsh_bits: int, lsh_tables: int,
                      n_queries: int = 200, seed: int = 0) -> list[dict]:
    """Mean lookup time for growing caches, brute force vs LSH, and how often LSH
    returns the same best entry as brute force."""
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size > len(vectors) - n_queries:
            continue
        perm = rng.permutation(len(vectors))
        stored, probes = vectors[perm[:size]], vectors[perm[size:size + n_queries]]
        brute = SemanticCache(vectors.shape[1], threshold=1.0, max_entries=size + 1)
        lsh = SemanticCache(vectors.shape[1], threshold=1.0, max_entries=size + 1, use_lsh=True,
                            lsh_bits=lsh_bits, lsh_tables=lsh_tables, lsh_min_entries=0)
        for i, v in enumerate(stored):
            brute.add(f"q{i}", "", v)
            lsh.add(f"q{i}", "", v)
        timings = {}
        results = {}
        for name, cache in (("brute", brute), ("lsh", lsh)):
            t0 = time.perf_counter()
            results[name] = [cache.lookup(q, top_k=1, record=False) for q in probes]
            timings[name] = (time.perf_counter() - t0) * 1000 / len(probes)
        # An LSH lookup can find no candidates at all; that counts as a disagreement.
        same = [bool(b.matches) and a.matches[0].entry.query == b.matches[0].entry.query
                for a, b in zip(results["brute"], results["lsh"]) if a.matches]
        compared = np.mean([r.compared for r in results["lsh"]])
        rows.append({"size": int(size), "brute_ms": timings["brute"], "lsh_ms": timings["lsh"],
                     "lsh_agreement": float(np.mean(same)) if same else float("nan"),
                     "lsh_compared": float(compared)})
    return rows


# ---------------------------------------------------------------------------
# charts (matplotlib, imported lazily so the core package doesn't need it)
# ---------------------------------------------------------------------------

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3


_RC = {
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    "legend.labelcolor": INK2,
}


class _Figures:
    """Object-oriented matplotlib without pyplot: charts are drawn and saved without
    touching the global backend, so they work in scripts, servers and notebooks alike."""

    @staticmethod
    def subplots(nrows: int = 1, ncols: int = 1, figsize=(7.5, 4.6), dpi: int = 150):
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize, dpi=dpi)
        return fig, fig.subplots(nrows, ncols)

    @staticmethod
    def close(fig) -> None:
        pass


def _plt() -> _Figures:
    import matplotlib  # noqa: F401  (fails early with a clear error if matplotlib is missing)
    return _Figures()


def _styled(fn):
    """Apply the chart style only while one chart is drawn and saved."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import matplotlib
        with matplotlib.rc_context(_RC):
            return fn(*args, **kwargs)
    return wrapper


def _frame(ax, title: str, subtitle: str | None, xlabel: str, ylabel: str):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def _end_labels(ax, xs, series: list[tuple[str, np.ndarray]], min_gap: float):
    """Direct labels at the right end of each line, skipped when they would collide."""
    placed: list[float] = []
    for name, ys in series:
        y = float(ys[-1])
        if all(abs(y - p) >= min_gap for p in placed):
            ax.annotate(name, (xs[-1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
            placed.append(y)


def _legend_below(ax, ncol: int):
    """Legend in its own row under the x-axis label, so it never covers a line."""
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=ncol, handlelength=1.6,
              columnspacing=1.4, borderaxespad=0.0)


def _line(ax, xs, ys, color, label, markers=False):
    ax.plot(xs, ys, color=color, linewidth=1.6, solid_capstyle="round", solid_joinstyle="round",
            label=label, marker="o" if markers else None, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)


@_styled
def plot_spectrum(embedder: LSAEmbedder, path: Path) -> Path:
    plt = _plt()
    s = embedder.s
    energy = np.cumsum(embedder.energy_profile()) * 100
    ks = np.arange(1, len(s) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    _line(a1, ks, s, SERIES[0], "singular value")
    _frame(a1, "Singular value spectrum", "How much each latent direction carries", "index $i$", r"$\sigma_i$")
    _line(a2, ks, energy, SERIES[0], "energy")
    _frame(a2, "Energy kept by the top k directions",
           r"$(\sigma_1^2 + \cdots + \sigma_k^2)\ /\ \|X\|_F^2$", "$k$", "% of matrix energy")
    a2.set_ylim(0, 100)
    a2.annotate(f"{energy[-1]:.0f}% at k = {len(s)}", (ks[-1], energy[-1]), xytext=(-4, 8),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


@_styled
def plot_threshold_sweep(sweep: list[dict], chosen: dict, path: Path) -> Path:
    plt = _plt()
    taus = np.array([r["tau"] for r in sweep])
    lines = [("Precision", np.array([r["precision"] for r in sweep])),
             ("Recall", np.array([r["recall"] for r in sweep])),
             ("F1", np.array([r["f1"] for r in sweep]))]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for (name, ys), color in zip(lines, SERIES):
        _line(ax, taus, ys, color, name)
    ax.axvline(chosen["tau"], color=MUTED, linewidth=0.8)
    ax.annotate(rf"best F1 at $\tau$ = {chosen['tau']:.2f}", (chosen["tau"], 1.0), xytext=(4, -2),
                textcoords="offset points", va="top", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(taus[0], taus[-1] + 0.06)
    _frame(ax, "Hit decision quality across thresholds",
           r"A pair counts as a hit when its similarity is at least $\tau$", r"threshold $\tau$", "score")
    _end_labels(ax, taus, lines, min_gap=0.06)
    _legend_below(ax, 3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


@_styled
def plot_rank_sweep(rows: list[dict], baseline: dict, path: Path) -> Path:
    plt = _plt()
    ks = np.array([r["k"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    for ax, key, base, title in ((a1, "auc", baseline["auc"], "ROC AUC by rank k"),
                                 (a2, "best_f1", baseline["best_f1"], "Best F1 by rank k")):
        ys = np.array([r[key] for r in rows])
        _line(ax, ks, ys, SERIES[0], "LSA", markers=True)
        ax.axhline(base, color=MUTED, linewidth=0.8)
        ax.annotate("TF-IDF only (no SVD)", (ks[0], base), xytext=(0, 4), textcoords="offset points",
                    color=INK2, fontsize=9)
        lo = min(ys.min(), base)
        ax.set_ylim(max(0.0, lo - 0.05), min(1.0, max(ys.max(), base) + 0.05))
        _frame(ax, title, "Blue line: LSA embeddings; grey line: baseline", "k (SVD directions kept)",
               key.replace("best_", "").upper() if key == "auc" else "F1")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


@_styled
def plot_cache_simulation(rows: list[dict], path: Path) -> Path:
    plt = _plt()
    taus = np.array([r["tau"] for r in rows])
    lines = [("Duplicates answered with their partner", np.array([r["duplicate_hit_partner"] for r in rows])),
             ("Non-duplicates answered with their partner (wrong)",
              np.array([r["non_duplicate_hit_partner"] for r in rows]))]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for (name, ys), color in zip(lines, SERIES):
        _line(ax, taus, ys, color, name)
    ax.set_ylim(0, 1.05)
    _frame(ax, "Cache simulation: what gets served from the cache",
           "Question 1 of every pair is cached; question 2 is asked", r"threshold $\tau$", "share of queries")
    _legend_below(ax, 2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


@_styled
def plot_latency(rows: list[dict], path: Path) -> Path:
    plt = _plt()
    sizes = np.array([r["size"] for r in rows])
    lines = [("Brute force", np.array([r["brute_ms"] for r in rows])),
             ("LSH", np.array([r["lsh_ms"] for r in rows]))]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for (name, ys), color in zip(lines, SERIES):
        _line(ax, sizes, ys, color, name, markers=True)
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    _frame(ax, "Lookup time as the cache grows", "Mean milliseconds per lookup", "cached entries (log scale)",
           "ms per lookup")
    _end_labels(ax, sizes, lines, min_gap=max(0.05 * max(l[1].max() for l in lines), 1e-3))
    _legend_below(ax, 2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# the whole evaluation
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Replace NaN/inf with None: json.dumps would write them as NaN, which is not valid JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in columns) + " |"
    sep = "|" + "|".join("---:" for _ in columns) + "|"
    body = ["| " + " | ".join(format(r[c[0]], c[2]) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def run_evaluation(settings: Settings | None = None, pairs_path: str | Path | None = None,
                   max_pairs: int | None = None, out_dir: str | Path | None = None,
                   charts: bool = True, verbose: bool = True) -> dict:
    settings = settings or Settings.load()
    if not LSAEmbedder.exists(settings.model_dir):
        raise FileNotFoundError("no built model; run `python -m semcache build` first")
    from .pipeline import load_embedder          # applies the query-time [cache] policy
    embedder = load_embedder(settings)
    out = Path(out_dir) if out_dir else settings.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = settings.eval
    max_pairs = max_pairs or cfg.max_pairs

    default_quora = settings.path(settings.paths.quora_eval)
    if cfg.pairs not in ("auto", "faq", "quora"):
        raise ValueError(f"[eval] pairs must be auto, faq or quora, not {cfg.pairs!r}")
    if pairs_path is None and cfg.pairs == "quora" and not default_quora.exists():
        raise FileNotFoundError(f"[eval] pairs = \"quora\" needs {default_quora}")
    if pairs_path is None and cfg.pairs != "faq" and default_quora.exists():
        pairs_path = default_quora
    if pairs_path is not None:
        pairs = load_question_pairs(pairs_path, max_pairs, cfg.seed)
        source = f"Quora Question Pairs ({Path(pairs_path).name}, {len(pairs):,} pairs)"
    else:
        pairs = faq_eval_pairs(load_faq(settings.path(settings.paths.faq_csv)), seed=cfg.seed)
        source = (f"FAQ held-out phrasings ({len(pairs)} pairs; small, for a quick check only - "
                  "download Quora for real numbers)")
    labels = np.array([p.label for p in pairs])
    if verbose:
        print(f"Evaluating on {source}")

    t0 = time.perf_counter()
    weight = settings.cache.lsa_weight
    cut = settings.cache.min_latent_coverage
    lsa_scores = pair_scores(embedder, pairs)
    tfidf_scores = tfidf_pair_scores(embedder, pairs)
    rho = pair_rho(embedder, pairs)
    scores, blocked = la.guarded_scores(lsa_scores, tfidf_scores, rho, weight, cut)   # what the cache uses
    sweep = threshold_sweep(scores, labels)
    chosen = best_f1(sweep)
    precise = tau_for_precision(sweep, 0.95)
    tfidf_sweep = threshold_sweep(tfidf_scores, labels)
    baseline = {"auc": roc_auc(tfidf_scores, labels), "best_f1": best_f1(tfidf_sweep)["f1"],
                "best_tau": best_f1(tfidf_sweep)["tau"]}
    ranks = rank_sweep(embedder, pairs, cfg.ranks + [embedder.dim])
    blends = blend_sweep(lsa_scores, tfidf_scores, labels, rho=rho, min_coverage=cut)
    # Most-LSA blend whose AUC is within 0.005 of the best (smaller gaps are noise).
    top_auc = max(r["auc"] for r in blends)
    recommended = max((r for r in blends if r["auc"] >= top_auc - 0.005), key=lambda r: r["lsa_weight"])
    if verbose:
        print(f"  pair scores, threshold and rank sweeps: {time.perf_counter() - t0:.1f}s")

    sim_pairs = random.Random(cfg.seed).sample(pairs, min(cfg.cache_sim_pairs, len(pairs)))
    simulation = cache_simulation(embedder, sim_pairs, settings.cache.min_known_ratio,
                                  lsa_weight=weight, min_latent_coverage=cut)

    pool_texts = list(dict.fromkeys([p.q1 for p in pairs] + [p.q2 for p in pairs]))
    pool = embedder.embed(pool_texts)
    pool = pool[np.any(pool, axis=1)]
    latency = latency_benchmark(pool, cfg.latency_sizes, settings.cache.lsh_bits,
                                settings.cache.lsh_tables, seed=cfg.seed)
    if verbose:
        print(f"  cache simulation and latency benchmark done ({time.perf_counter() - t0:.1f}s total)")

    auc = roc_auc(scores, labels)
    lsa_auc = roc_auc(lsa_scores, labels)
    configured = next((r for r in sweep if abs(r["tau"] - settings.cache.threshold) < 1e-9), None)
    report = {
        "data": source,
        "model": embedder.describe(),
        "similarity": {"lsa_weight": weight, "min_latent_coverage": cut, "auc": auc, "best_f1": chosen,
                       "tau_for_95_precision": precise, "at_configured_tau": configured,
                       "pairs_scored_by_word_overlap": float(np.mean(blocked))},
        "lsa_only_auc": lsa_auc,
        "tfidf_baseline": baseline,
        "recommended_lsa_weight": recommended,
        "threshold_sweep": sweep,
        "rank_sweep": ranks,
        "blend_sweep": blends,
        "cache_simulation": simulation,
        "latency": latency,
    }
    (out / "report.json").write_text(json.dumps(_json_safe(report), indent=1, allow_nan=False),
                                     encoding="utf-8")

    chart_files = []
    if charts:
        try:
            chart_files = [
                plot_spectrum(embedder, out / "singular_values.png"),
                plot_threshold_sweep(sweep, chosen, out / "threshold_sweep.png"),
                plot_rank_sweep(ranks, baseline, out / "rank_sweep.png"),
                plot_cache_simulation(simulation, out / "cache_simulation.png"),
            ]
            if latency:
                chart_files.append(plot_latency(latency, out / "latency.png"))
        except ModuleNotFoundError:
            if verbose:
                print("  matplotlib is not installed; skipping charts (pip install matplotlib)")

    every_tenth = [r for r in sweep if round(r["tau"] * 100) % 5 == 0]
    sim_rows = [r for r in simulation if round(r["tau"] * 100) % 5 == 0]
    md = [
        "# Evaluation report", "",
        f"Data: {source}", "",
        f"Model: k = {embedder.dim}, vocabulary {len(embedder.vocab):,} terms, "
        f"trained on {embedder.report.n_docs:,} texts.", "",
        "## Headline", "",
        f"- Cache similarity: lsa_weight = {weight:.2f} "
        f"({'pure LSA' if weight == 1 else 'pure TF-IDF' if weight == 0 else 'LSA + TF-IDF blend'}), "
        f"rho guard min_latent_coverage = {cut:.2f} ({np.mean(blocked):.1%} of pairs scored by word overlap)",
        f"- ROC AUC **{auc:.3f}** (LSA only {lsa_auc:.3f}, TF-IDF only {baseline['auc']:.3f})",
        f"- Best blend on this data: lsa_weight = {recommended['lsa_weight']:.2f} "
        f"(AUC {recommended['auc']:.3f}, best F1 {recommended['best_f1']:.3f} at tau {recommended['best_tau']:.2f})",
        f"- Best F1 **{chosen['f1']:.3f}** at tau = {chosen['tau']:.2f} "
        f"(precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f})",
        (f"- For 95% precision use tau >= {precise['tau']:.2f} (recall {precise['recall']:.3f})"
         if precise else "- No threshold reaches 95% precision on this data"),
        (f"- At the configured tau = {settings.cache.threshold:.2f}: precision {configured['precision']:.3f}, "
         f"recall {configured['recall']:.3f}" if configured else ""),
        "", "## Threshold sweep", "",
        _table(every_tenth, [("tau", "tau", ".2f"), ("precision", "precision", ".3f"),
                             ("recall", "recall", ".3f"), ("f1", "F1", ".3f"),
                             ("false_hit_rate", "false-hit rate", ".3f")]),
        "", "## Rank sweep (k)", "",
        _table(ranks, [("k", "k", "d"), ("auc", "AUC", ".3f"), ("best_f1", "best F1", ".3f"),
                       ("best_tau", "at tau", ".2f"), ("energy_kept", "energy kept", ".1%")]),
        "", f"TF-IDF only (no SVD): AUC {baseline['auc']:.3f}, best F1 {baseline['best_f1']:.3f} "
            f"at tau {baseline['best_tau']:.2f}",
        "", "## Blend sweep (LSA weight w)", "",
        f"Score = w * cos_LSA + (1 - w) * cos_TFIDF, with the rho guard at {cut:.2f}. "
        "Set `lsa_weight` in config.toml to the best row.", "",
        _table(blends, [("lsa_weight", "w", ".2f"), ("auc", "AUC", ".3f"), ("best_f1", "best F1", ".3f"),
                        ("best_tau", "at tau", ".2f")]),
        "", "## Cache simulation", "",
        "Question 1 of each pair is cached, question 2 is asked. A non-duplicate answered with its own "
        "partner is certainly wrong; a hit on another entry may or may not be (the labels don't say).", "",
        _table(sim_rows, [("tau", "tau", ".2f"), ("duplicate_hit_partner", "duplicates: right entry", ".3f"),
                          ("duplicate_hit_other", "duplicates: other entry", ".3f"),
                          ("duplicate_miss", "duplicates: miss", ".3f"),
                          ("non_duplicate_hit_partner", "non-dup.: partner (wrong)", ".3f"),
                          ("non_duplicate_hit_other", "non-dup.: other entry", ".3f")]),
        "", "## Lookup latency", "",
        (_table(latency, [("size", "entries", "d"), ("brute_ms", "brute ms", ".3f"),
                          ("lsh_ms", "LSH ms", ".3f"), ("lsh_compared", "LSH vectors scored", ".0f"),
                          ("lsh_agreement", "LSH same best match", ".3f")])
         if latency else "Not enough vectors for the configured cache sizes."),
        "", "## Charts", "",
        *[f"![{p.stem}]({p.name})" for p in chart_files],
    ]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report["files"] = [str(out / "report.md"), str(out / "report.json"), *map(str, chart_files)]
    if verbose:
        print(f"  AUC {auc:.3f} at lsa_weight {weight:.2f} (LSA only {lsa_auc:.3f}, TF-IDF only "
              f"{baseline['auc']:.3f}); best F1 {chosen['f1']:.3f} at tau {chosen['tau']:.2f}")
        print(f"  best blend here: lsa_weight = {recommended['lsa_weight']:.2f} (AUC {recommended['auc']:.3f})")
        print(f"  report: {out / 'report.md'}")
    return report


# ---------------------------------------------------------------------------
# guard sweep: latent stopwords x k x rho cutoff
# ---------------------------------------------------------------------------

# Questions that hit a wrong entry with the plain Quora model, and the FAQ questions
# a hit on would be acceptable. The cache for this check holds the 43 FAQ answers
# plus EXTRA_CACHED (an Ollama answer from our saved cache).
FALSE_HIT_PROBES: list[tuple[str, list[str]]] = [
    ("what is the capital of france", []),
    ("how to enable dark mode", []),
    ("what is svd", ["What is singular value decomposition?"]),
    ("what is phishing email ?", ["How can I recognise a phishing email?"]),
]
EXTRA_CACHED = ["i forgot my password, how can i reset it"]


def faq_cache(embedder: LSAEmbedder, items: list[FAQItem], tau: float, lsa_weight: float,
              min_latent_coverage: float, min_known_ratio: float) -> SemanticCache:
    """A cache like the served one: the FAQ answers plus EXTRA_CACHED."""
    cache = SemanticCache(embedder.dim, threshold=tau, lsa_weight=lsa_weight,
                          min_latent_coverage=min_latent_coverage)
    texts = [i.question for i in items] + EXTRA_CACHED
    vectors, rows, rho = embedder.encode_with_rho(texts, min_known_ratio)
    for i, text in enumerate(texts):
        cache.add(text, "", vectors[i], source="faq" if i < len(items) else "ollama",
                  tfidf_row=rows[i], rho=float(rho[i]))
    return cache


def ask_cache(cache: SemanticCache, embedder: LSAEmbedder, text: str, min_known_ratio: float):
    vectors, rows, rho = embedder.encode_with_rho([text], min_known_ratio)
    return cache.lookup(vectors[0], 1, record=False, query_text=text, query_tfidf=rows[0],
                        query_rho=float(rho[0]))


def faq_guard_check(embedder: LSAEmbedder, items: list[FAQItem], tau: float, lsa_weight: float,
                    cut: float, min_known_ratio: float) -> dict:
    """What the guard costs the FAQ demo questions, and which known false hits remain."""
    questions = [i.question for i in items]
    heldout = [(h, i.question) for i in items for h in i.heldout]
    rho_q = embedder.latent_coverage(embedder.tfidf(questions))
    rho_h = embedder.latent_coverage(embedder.tfidf([h for h, _ in heldout]))
    cache = faq_cache(embedder, items, tau, lsa_weight, cut, min_known_ratio)
    right = 0
    for text, target in heldout:
        r = ask_cache(cache, embedder, text, min_known_ratio)
        right += bool(r.hit and r.best is not None and r.best.query == target)
    probes = []
    for text, acceptable in FALSE_HIT_PROBES:
        r = ask_cache(cache, embedder, text, min_known_ratio)
        best = r.matches[0] if r.matches else None
        probes.append({"query": text, "hit": r.hit, "score": float(r.score),
                       "matched": best.entry.query if best else None,
                       "scored_by": best.scored_by if best else None,
                       "wrong": bool(r.hit and best is not None and best.entry.query not in acceptable)})
    return {"faq_questions": len(questions), "faq_blocked": int(np.sum(rho_q < cut)),
            "heldout": len(heldout), "heldout_blocked": int(np.sum(rho_h < cut)),
            "heldout_recall": right / len(heldout) if heldout else float("nan"),
            "wrong_hits": sum(p["wrong"] for p in probes), "probes": probes}


def random_question_hits(embedder: LSAEmbedder, items: list[FAQItem], questions: list[str], tau: float,
                         lsa_weight: float, cut: float, min_known_ratio: float, show: int = 10) -> dict:
    """Ask unrelated questions against a cache holding only the 43 FAQ answers.

    Random Quora questions are almost never about the FAQ's topics, so nearly every hit
    is a wrong answer. Returns the hit count and the highest-scoring hits.
    """
    cache = SemanticCache(embedder.dim, threshold=tau, lsa_weight=lsa_weight, min_latent_coverage=cut)
    vectors, rows, rho = embedder.encode_with_rho([i.question for i in items], min_known_ratio)
    for i, item in enumerate(items):
        cache.add(item.question, "", vectors[i], source="faq", tfidf_row=rows[i], rho=float(rho[i]))
    qv, qrows, qrho = embedder.encode_with_rho(questions, min_known_ratio)
    best = []
    for i, text in enumerate(questions):
        r = cache.lookup(qv[i], 1, record=False, query_text=text, query_tfidf=qrows[i],
                         query_rho=float(qrho[i]))
        if r.matches:
            m = r.matches[0]
            best.append({"question": text, "matched": m.entry.query, "score": round(m.score, 4),
                         "rho": round(float(qrho[i]), 3), "scored_by": m.scored_by, "hit": r.hit})
    best.sort(key=lambda h: -h["score"])
    hits = [h for h in best if h["hit"]]
    # "top": the highest-scoring questions, hits or not, to show how close the misses come.
    return {"asked": len(questions), "hits": len(hits), "top": best[:show]}


def sample_quora_questions(settings: Settings, n: int = 1000, seed: int = 11) -> list[str]:
    """n distinct questions from the Quora validation file (never used to fit the SVD)."""
    pairs = load_question_pairs(settings.path(settings.paths.quora_eval))
    pool = sorted({q for p in pairs for q in (p.q1, p.q2)})
    return random.Random(seed).sample(pool, min(n, len(pool)))


def guard_sweep(settings: Settings, ranks=(200, 300, 400), cutoffs=(0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
                latent=(False, True), max_pairs: int | None = None, out_dir: str | Path | None = None,
                verbose: bool = True) -> dict:
    """Refit the model for every (latent stopwords, k) and score every rho cutoff.

    Quora numbers use the profile's lsa_weight and tau; cutoff 0 means the guard is off.
    FAQ numbers come from a cache holding the FAQ answers (see faq_guard_check).
    """
    from .pipeline import build_corpus          # imported here: pipeline imports this module's peers

    tau, weight = settings.cache.threshold, settings.cache.lsa_weight
    ratio = settings.cache.min_known_ratio
    corpus, info = build_corpus(settings)
    quora_eval = settings.path(settings.paths.quora_eval)
    if not quora_eval.exists():
        raise FileNotFoundError(f"the guard sweep needs Quora pairs at {quora_eval}")
    pairs = load_question_pairs(quora_eval, max_pairs or settings.eval.max_pairs, settings.eval.seed)
    labels = np.array([p.label for p in pairs])
    items = load_faq(settings.path(settings.paths.faq_csv))
    randoms = sample_quora_questions(settings)
    if verbose:
        print(f"Guard sweep on {len(corpus):,} texts, {len(pairs):,} Quora pairs, tau {tau:.2f}, "
              f"lsa_weight {weight:.2f}")

    rows = []
    for use_latent in latent:
        for k in ranks:
            t0 = time.perf_counter()
            cfg = dataclasses.replace(settings.embedder, rank=k, use_latent_stopwords=use_latent)
            emb = LSAEmbedder(cfg).fit(corpus)
            Z1, X1, r1 = emb.encode_with_rho([p.q1 for p in pairs])
            Z2, X2, r2 = emb.encode_with_rho([p.q2 for p in pairs])
            lsa = np.einsum("ij,ij->i", Z1, Z2)
            lexical = la.rowwise_cosine(X1, X2)
            rho = np.minimum(r1, r2)
            auc_lsa = roc_auc(lsa, labels)
            for cut in cutoffs:
                scores, blocked = la.guarded_scores(lsa, lexical, rho, weight, cut)
                sweep = threshold_sweep(scores, labels)
                at = next(r for r in sweep if abs(r["tau"] - tau) < 1e-9)
                best = best_f1(sweep)
                faq = faq_guard_check(emb, items, tau, weight, cut, ratio)
                faq["random_hits"] = random_question_hits(emb, items, randoms, tau, weight, cut, ratio,
                                                          show=0)["hits"]
                rows.append({"latent_stopwords": use_latent, "k": emb.dim, "cutoff": cut,
                             "auc_lsa": auc_lsa, "auc": roc_auc(scores, labels),
                             "precision": at["precision"], "recall": at["recall"], "f1": at["f1"],
                             "best_f1": best["f1"], "best_tau": best["tau"],
                             "quora_blocked": float(np.mean(blocked)), **faq})
            if verbose:
                print(f"  latent stopwords {'on ' if use_latent else 'off'}, k = {emb.dim}: "
                      f"{time.perf_counter() - t0:.1f}s")

    recommended = _recommend(rows)
    report = {"tau": tau, "lsa_weight": weight, "pairs": len(pairs), "corpus": info,
              "rows": rows, "recommended": recommended, "probes": FALSE_HIT_PROBES,
              "extra_cached": EXTRA_CACHED,
              "configured": {"latent_stopwords": settings.embedder.use_latent_stopwords,
                             "k": settings.embedder.rank, "cutoff": settings.cache.min_latent_coverage}}
    out = Path(out_dir) if out_dir else settings.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "guard_sweep.json").write_text(json.dumps(_json_safe(report), indent=1, allow_nan=False),
                                          encoding="utf-8")
    (out / "guard_sweep.md").write_text(_guard_markdown(report), encoding="utf-8")
    if verbose:
        print(f"  report: {out / 'guard_sweep.md'}")
    return report


def _recommend(rows: list[dict]) -> dict | None:
    """No known false hit first; then the most FAQ held-out recall; then Quora F1 at tau."""
    safe = [r for r in rows if r["wrong_hits"] == 0]
    if not safe:
        return None
    return max(safe, key=lambda r: (round(r["heldout_recall"], 3), round(r["f1"], 3), -r["cutoff"]))


def _guard_markdown(report: dict) -> str:
    rec, conf = report["recommended"], report.get("configured")
    keys = ("latent_stopwords", "k", "cutoff")

    def same(r, other):
        return other is not None and all(abs(r[k] - other[k]) < 1e-9 if k == "cutoff" else r[k] == other[k]
                                         for k in keys)

    def mark(r):
        return " ".join(m for m, on in (("**configured**", same(r, conf)), ("→", same(r, rec))) if on)
    lines = [
        "# Guard sweep", "",
        f"Quora: {report['pairs']:,} validation pairs, tau = {report['tau']:.2f}, "
        f"lsa_weight = {report['lsa_weight']:.2f}. Cutoff 0 = guard off.",
        "FAQ: a cache with the 43 FAQ answers plus " + ", ".join(f'"{t}"' for t in report["extra_cached"])
        + "; held-out recall = held-out paraphrases that hit their own FAQ answer.",
        "Wrong hits: " + "; ".join(f'"{q}"' for q, _ in report["probes"])
        + " (a hit on the matching FAQ question counts as right).",
        "Random Quora hits: 1,000 random Quora validation questions asked against the 43 FAQ answers "
        "(almost every hit is wrong).", "",
        "| | latent stopwords | k | rho cutoff | Quora AUC (LSA only) | Quora AUC | P @ tau | R @ tau | "
        "best F1 (tau) | Quora pairs blocked | FAQ questions blocked | FAQ held-out blocked | "
        "FAQ held-out recall | wrong hits | random Quora hits / 1000 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {mark(r)} | {'on' if r['latent_stopwords'] else 'off'} | {r['k']} | {r['cutoff']:.2f} | "
            f"{r['auc_lsa']:.3f} | {r['auc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['best_f1']:.3f} ({r['best_tau']:.2f}) | {r['quora_blocked']:.1%} | "
            f"{r['faq_blocked']}/{r['faq_questions']} | {r['heldout_blocked']}/{r['heldout']} | "
            f"{r['heldout_recall']:.3f} | {r['wrong_hits']}/{len(report['probes'])} | {r['random_hits']} |")
    if rec is not None:
        lines += ["", f"→ the automatic rule's pick (no wrong hits, then most FAQ held-out recall, then "
                      f"Quora F1 at tau): latent stopwords {'on' if rec['latent_stopwords'] else 'off'}, "
                      f"k = {rec['k']}, cutoff {rec['cutoff']:.2f}. **configured** marks what config.toml uses."]
    return "\n".join(lines) + "\n"
