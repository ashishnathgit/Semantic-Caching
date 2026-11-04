"""LSA embedder: text -> TF-IDF row -> latent "meaning" vector.

All the matrix work is delegated to semcache.linalg; this class only keeps the
fitted pieces (vocabulary, IDF weights, V_k, singular values) together.

    fit(corpus):   X = TF-IDF(corpus)                 (n_docs x n_terms, sparse)
                   X ~= U_k Sigma_k V_k^T             (linalg.truncated_svd)
    embed(text):   q = TF-IDF(text)                   (1 x n_terms)
                   e = q V_k Sigma_k^{-1} Sigma_k^p   (linalg.project, fold-in)
                   e = e / ||e||                      (linalg.l2_normalize_rows)
    coverage:      rho = ||q V_k|| / ||q||            (linalg.latent_coverage)

Latent stopwords: with ``use_latent_stopwords`` the listed words ("what", "how", ...)
are left out of the SVD fit (their columns of X are zeroed, so their rows of V_k are
zero) but stay in the TF-IDF rows, which the exact and word-overlap scores use.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from . import linalg as la
from .text import EVERYDAY_WORDS, STOPWORDS, Vocabulary, analyze, fold_plural, tokenize

DEFAULT_LATENT_STOPWORDS = ["what", "how", "why", "which", "best", "way", "good", "can", "does"]


@dataclass
class EmbedderConfig:
    rank: int = 200                  # k, number of latent dimensions kept
    max_rank_fraction: float = 0.35  # k is also capped at this share of the documents
    ngram_max: int = 2               # 1 = words only, 2 = words + bigrams
    min_df: int = 1                  # drop single words seen in fewer documents
    min_df_ngram: int = 2            # ... and bigrams seen in fewer documents
    max_df_ratio: float = 1.0        # drop terms seen in more than this share of documents
    max_features: int = 40000        # vocabulary cap
    sublinear_tf: bool = True        # TF -> 1 + ln(TF)
    smooth_idf: bool = True          # ln((1+N)/(1+df)) + 1
    stem: bool = True                # plural folding
    sigma_power: float = 1.0         # embedding = fold-in * Sigma^p (see linalg.project)
    oversample: int = 20             # randomized SVD settings
    power_iters: int = 7
    seed: int = 42
    use_latent_stopwords: bool = False   # leave the words below out of the SVD fit
    latent_stopwords: list[str] = field(default_factory=lambda: list(DEFAULT_LATENT_STOPWORDS))


@dataclass(frozen=True)
class KnownWordPolicy:
    """The known-word guard's topic-word rule (a query-time policy, not part of the fitted
    model, so it is not in the fingerprint).

    The plain rule: at least ``min_known_ratio`` of a question's content words must be in
    the vocabulary, else it can only hit by exact repeat. A long, wordy question fails it
    through everyday words the model never saw ("where do I go to ... the old one").
    This policy gives such a question a second chance: counting only topic words (content
    words that are not EVERYDAY_WORDS), it needs at least ``min_topic_words`` known ones and
    ``min_topic_ratio`` of its topic words known. It is then matched on its known topic
    words alone, so known everyday words ("why", "find") can't carry it to a hit either.
    """
    min_topic_words: int = 2
    min_topic_ratio: float = 0.4
    everyday: frozenset = field(default=EVERYDAY_WORDS | frozenset(fold_plural(w) for w in EVERYDAY_WORDS))

    @property
    def signature(self) -> str:
        h = hashlib.sha1(" ".join(sorted(self.everyday)).encode()).hexdigest()[:8]
        return f"topic>={self.min_topic_words},{self.min_topic_ratio:g},{h}"


@dataclass
class Admission:
    """How the known-word guard treats one question."""
    status: str                      # "ok" (plain rule), "relaxed" (topic-word rule) or "blocked"
    known: int                       # known content words
    total: int                       # content words
    topic_known: list[str] = field(default_factory=list)
    topic_unknown: list[str] = field(default_factory=list)     # unknown words that carry topic
    everyday_unknown: list[str] = field(default_factory=list)  # unknown everyday words (not counted)

    @property
    def passed(self) -> bool:
        return self.status != "blocked"


@dataclass
class FitReport:
    n_docs: int = 0
    n_terms: int = 0
    rank: int = 0
    nnz: int = 0
    seconds_tfidf: float = 0.0
    seconds_svd: float = 0.0
    energy_kept: float = 0.0         # sum(s_k^2) / ||X||_F^2
    notes: list[str] = field(default_factory=list)


class LSAEmbedder:
    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()
        self.vocab: Vocabulary | None = None
        self.idf: np.ndarray | None = None
        self.V: np.ndarray | None = None      # n_terms x k
        self.s: np.ndarray | None = None      # k singular values, descending
        self.total_energy: float = 0.0        # ||X||_F^2 of the training matrix
        self.report = FitReport()
        self._svd_mask: np.ndarray | None = None
        self.policy: KnownWordPolicy | None = None   # query-time; set from [cache] (pipeline.configure_embedder)
        self._everyday_keep: np.ndarray | None = None

    # ------------------------------------------------------------------ fitting
    def analyze(self, text: str) -> list[str]:
        return analyze(text, self.config.ngram_max, STOPWORDS, self.config.stem)

    def fit(self, corpus: Sequence[str], verbose: bool = False) -> "LSAEmbedder":
        cfg = self.config
        corpus = [t for t in corpus if t and t.strip()]
        if len(corpus) < 3:
            raise ValueError("need at least 3 non-empty documents to fit the embedder")

        t0 = time.perf_counter()
        terms = [self.analyze(t) for t in corpus]
        self.vocab = Vocabulary.fit(terms, cfg.min_df, cfg.max_df_ratio, cfg.max_features,
                                    cfg.min_df_ngram)
        if len(self.vocab) < 2:
            raise ValueError("vocabulary is empty; lower min_df or add more text")
        ids = [self.vocab.encode(t) for t in terms]
        tf = la.term_frequency_matrix(ids, len(self.vocab))
        df = la.document_frequency(tf)
        self.idf = la.inverse_document_frequency(df, len(corpus), cfg.smooth_idf)
        X = la.tfidf_matrix(tf, self.idf, sublinear_tf=cfg.sublinear_tf)
        self._svd_mask = None
        in_svd = self.svd_mask
        if not in_svd.all():
            X = la.zero_columns(X, in_svd)        # latent stopwords: out of the SVD only
        t1 = time.perf_counter()

        report = FitReport(n_docs=X.shape[0], n_terms=X.shape[1], nnz=X.nnz)
        if not in_svd.all():
            report.notes.append(f"left {int((~in_svd).sum())} latent stopwords out of the SVD fit: "
                                + ", ".join(self.vocab.terms[i] for i in np.flatnonzero(~in_svd)))
        # LSA only merges related words when k is well below the number of documents.
        rank, cap = self.fit_rank(X.shape[0], X.shape[1])
        if rank < cfg.rank:
            reason = (f"max_rank_fraction = {cfg.max_rank_fraction} caps it at {cap}" if rank == cap
                      else "k must be smaller than both sides of the matrix")
            report.notes.append(
                f"rank lowered from {cfg.rank} to {rank}: the TF-IDF matrix is {X.shape[0]:,} x "
                f"{X.shape[1]:,} and {reason}")
        if verbose:
            print(f"  TF-IDF matrix: {X.shape[0]:,} docs x {X.shape[1]:,} terms, "
                  f"{X.nnz:,} non-zeros ({t1 - t0:.1f}s)")
            print(f"  truncated SVD, k = {rank} ...", flush=True)

        _, s, V = la.truncated_svd(X, rank, oversample=cfg.oversample,
                                   power_iters=cfg.power_iters, seed=cfg.seed)
        t2 = time.perf_counter()

        keep = s > s[0] * 1e-10                 # drop numerically zero directions
        if not keep.all():
            report.notes.append(f"dropped {int((~keep).sum())} zero singular values")
            s, V = s[keep], V[:, keep]
            rank = int(keep.sum())
        V = V * in_svd[:, None]                  # exactly zero rows for left-out terms
        self.V, self.s = V, s
        self.total_energy = la.frobenius_norm_sq(X)
        report.rank = rank
        report.seconds_tfidf, report.seconds_svd = t1 - t0, t2 - t1
        report.energy_kept = float(np.sum(s ** 2) / self.total_energy) if self.total_energy else 0.0
        self.report = report
        if verbose:
            print(f"  SVD done in {t2 - t1:.1f}s; top-{rank} directions keep "
                  f"{100 * report.energy_kept:.1f}% of the matrix energy")
        return self

    def fit_rank(self, n_docs: int, n_terms: int) -> tuple[int, int]:
        """(k the fit asks the SVD for, the max_rank_fraction cap) for an n_docs x n_terms matrix."""
        cfg = self.config
        cap = max(2, int(cfg.max_rank_fraction * n_docs))
        return max(1, min(cfg.rank, cap, n_docs - 1, n_terms - 1)), cap

    # --------------------------------------------------------------- embedding
    @property
    def is_fitted(self) -> bool:
        return self.V is not None

    @property
    def dim(self) -> int:
        self._check()
        return int(self.V.shape[1])

    def _check(self):
        if not self.is_fitted:
            raise RuntimeError("embedder is not fitted yet; call fit() or load()")

    def tfidf(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Unit-length TF-IDF rows for new texts, using the fitted vocabulary and IDF."""
        self._check()
        ids = [self.vocab.encode(self.analyze(t)) for t in texts]
        tf = la.term_frequency_matrix(ids, len(self.vocab))
        return la.tfidf_matrix(tf, self.idf, sublinear_tf=self.config.sublinear_tf)

    def embed(self, texts: Sequence[str], rank: int | None = None,
              min_known_ratio: float = 0.0) -> np.ndarray:
        """Unit-length latent vectors, one row per text.

        ``rank`` uses only the top directions (the SVD is nested, so no refit is needed
        for a smaller k). Texts whose share of known words is below ``min_known_ratio``
        get a zero vector, which never produces a semantic match.
        """
        return self.encode(texts, min_known_ratio, rank)[0]

    def encode(self, texts: Sequence[str], min_known_ratio: float = 0.0,
               rank: int | None = None) -> tuple[np.ndarray, sp.csr_matrix]:
        """Both representations at once: (latent vectors, unit-length TF-IDF rows)."""
        self._check()
        k = self.dim if rank is None else max(1, min(rank, self.dim))
        X = self.tfidf(texts)
        Z = la.l2_normalize_rows(la.project(X, self.V[:, :k], self.s[:k], self.config.sigma_power))
        if min_known_ratio > 0:
            relaxed = []
            for i, text in enumerate(texts):
                status = self.admission(text, min_known_ratio).status
                if status == "blocked":
                    Z[i] = 0.0
                elif status == "relaxed":
                    relaxed.append(i)
            if relaxed:
                # Match these on their known topic words alone: the everyday terms' columns are
                # zeroed in the TF-IDF row (then re-normalised) before it is folded in.
                X = sp.csr_matrix(X, copy=True)
                keep = self.everyday_keep
                for i in relaxed:
                    a, b = X.indptr[i], X.indptr[i + 1]
                    X.data[a:b] *= keep[X.indices[a:b]]
                    norm = float(np.sqrt(X.data[a:b] @ X.data[a:b]))
                    if norm > 0:
                        X.data[a:b] /= norm
                X.eliminate_zeros()
                idx = np.array(relaxed)
                Z[idx] = la.l2_normalize_rows(la.project(X[idx], self.V[:, :k], self.s[:k], self.config.sigma_power))
        return Z, X

    def encode_with_rho(self, texts: Sequence[str], min_known_ratio: float = 0.0
                        ) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
        """(latent vectors, TF-IDF rows, latent coverage rho) for each text."""
        Z, X = self.encode(texts, min_known_ratio)
        return Z, X, self.latent_coverage(X)

    def latent_coverage(self, X_rows) -> np.ndarray:
        """rho = ||q V_k|| / ||q|| per TF-IDF row: the share of the question that survives
        in the k-dimensional space (latent stopwords don't count; see linalg.latent_coverage)."""
        self._check()
        keep = self.svd_mask
        return la.latent_coverage(X_rows, self.V, None if keep.all() else keep)

    @property
    def svd_mask(self) -> np.ndarray:
        """True for every vocabulary term the SVD is fitted on (all but latent stopwords)."""
        if self._svd_mask is None or self._svd_mask.size != len(self.vocab):
            keep = np.ones(len(self.vocab), dtype=bool)
            if self.config.use_latent_stopwords:
                for word in self.config.latent_stopwords:
                    term = fold_plural(word.lower()) if self.config.stem else word.lower()
                    if term in self.vocab:
                        keep[self.vocab.index[term]] = False
            self._svd_mask = keep
        return self._svd_mask

    def embed_one(self, text: str, min_known_ratio: float = 0.0) -> np.ndarray:
        return self.embed([text], min_known_ratio=min_known_ratio)[0]

    def known_terms(self, text: str) -> list[str]:
        """Which of the text's terms the model knows (helps explain a miss)."""
        self._check()
        return self.vocab.known(self.analyze(text))

    @property
    def everyday_keep(self) -> np.ndarray:
        """1.0 for every vocabulary term with no everyday word in it, 0.0 otherwise."""
        everyday = self.policy.everyday if self.policy else frozenset()
        if self._everyday_keep is None or self._everyday_keep.size != len(self.vocab):
            self._everyday_keep = np.array([0.0 if any(w in everyday for w in t.split()) else 1.0
                                            for t in self.vocab.terms])
        return self._everyday_keep

    @property
    def policy_signature(self) -> str:
        """Identifies the query-time policy; a cache built under another one is re-embedded."""
        return self.policy.signature if self.policy else ""

    def admission(self, text: str, min_known_ratio: float) -> Admission:
        """The known-word guard for one question: "ok", "relaxed" (the topic-word rule of
        ``self.policy``) or "blocked" (only an exact repeat can hit)."""
        self._check()
        words = tokenize(text, STOPWORDS, self.config.stem)
        known = [w for w in words if w in self.vocab]
        out = Admission("ok", len(known), len(words))
        if min_known_ratio <= 0:
            return out
        if words and known and len(known) / len(words) >= min_known_ratio:
            return out
        out.status = "blocked"
        if self.policy is not None:
            every = self.policy.everyday
            out.topic_known = [w for w in known if w not in every]
            out.topic_unknown = [w for w in words if w not in self.vocab and w not in every]
            out.everyday_unknown = [w for w in words if w not in self.vocab and w in every]
            tk, tu = len(out.topic_known), len(out.topic_unknown)
            if tk >= self.policy.min_topic_words and tk / (tk + tu) >= self.policy.min_topic_ratio:
                out.status = "relaxed"
        else:
            out.topic_unknown = [w for w in words if w not in self.vocab]
        return out

    def coverage(self, text: str) -> tuple[int, int]:
        """(known content words, total content words). If most words are unknown, the
        vector only reflects the few known ones, so a semantic match can't be trusted."""
        self._check()
        words = tokenize(text, STOPWORDS, self.config.stem)
        return sum(w in self.vocab for w in words), len(words)

    def direction_terms(self, n_directions: int = 8, n_terms: int = 6) -> list[dict]:
        """The terms that load most heavily on each of the first latent directions.

        Column j of V_k is a unit vector over the vocabulary, so its largest entries
        (in absolute value; the sign of a singular vector is arbitrary) show which
        words that direction groups together: a rough "topic".
        """
        self._check()
        out = []
        for j in range(min(n_directions, self.dim)):
            column = self.V[:, j]
            top = np.argsort(-np.abs(column))[:n_terms]
            out.append({"direction": j + 1, "sigma": round(float(self.s[j]), 4),
                        "terms": [self.vocab.terms[t] for t in top]})
        return out

    def energy_profile(self) -> np.ndarray:
        """Share of the TF-IDF matrix energy captured by each latent direction."""
        self._check()
        return self.s ** 2 / self.total_energy

    @property
    def fingerprint(self) -> str:
        """Identifies this exact model; caches built with another model get re-embedded."""
        self._check()
        h = hashlib.sha1()
        h.update("\n".join(self.vocab.terms).encode())
        h.update(np.round(self.s, 8).tobytes())
        h.update(json.dumps(asdict(self.config), sort_keys=True).encode())
        return h.hexdigest()[:12]

    def describe(self) -> dict:
        self._check()
        return {
            "fingerprint": self.fingerprint,
            "rank": self.dim,
            "vocabulary_size": len(self.vocab),
            "training_documents": self.report.n_docs,
            "energy_kept": round(self.report.energy_kept, 4),
            "config": asdict(self.config),
        }

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        self._check()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "embedder.npz", V=self.V, s=self.s, idf=self.idf)
        meta = {
            "config": asdict(self.config),
            "terms": self.vocab.terms,
            "total_energy": self.total_energy,
            "report": asdict(self.report),
            "fingerprint": self.fingerprint,
        }
        (directory / "embedder.json").write_text(json.dumps(meta), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "LSAEmbedder":
        directory = Path(directory)
        meta = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
        known = set(EmbedderConfig.__dataclass_fields__)
        emb = cls(EmbedderConfig(**{k: v for k, v in meta["config"].items() if k in known}))
        arrays = np.load(directory / "embedder.npz")
        emb.V, emb.s, emb.idf = arrays["V"], arrays["s"], arrays["idf"]
        emb.vocab = Vocabulary(meta["terms"])
        emb.total_energy = float(meta["total_energy"])
        emb.report = FitReport(**meta.get("report", {}))
        return emb

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "embedder.npz").exists() and (directory / "embedder.json").exists()
