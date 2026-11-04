"""A neural baseline: MiniLM sentence embeddings behind the same embedder interface as
LSAEmbedder, used ONLY to compare against LSA in the evaluation (the served cache never
uses it; its linear algebra stays in semcache/linalg.py).

    pip install -r requirements-neural.txt
    python -m semcache evaluate --profile faq --neural

The model (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions, ~90 MB) is downloaded
on first use. Without the optional packages everything here raises a clear ImportError.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from . import linalg as la

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class MiniLMEmbedder:
    """embed / encode / encode_with_rho / dim / fingerprint, like LSAEmbedder. There is no
    TF-IDF row and no SVD, so rho is 1 and the TF-IDF part is an empty sparse matrix."""

    def __init__(self, model: str = MODEL, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:                       # pragma: no cover - depends on the install
            raise ImportError("the neural baseline needs: pip install -r requirements-neural.txt") from exc
        self.name = model
        self._model = SentenceTransformer(model)
        self.batch_size = batch_size
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self._dim = int(get_dim())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return "minilm:" + self.name

    def embed(self, texts: Sequence[str], rank: int | None = None, min_known_ratio: float = 0.0) -> np.ndarray:
        Z = self._model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                               show_progress_bar=False)
        return la.l2_normalize_rows(np.asarray(Z, dtype=np.float64))

    def encode(self, texts: Sequence[str], min_known_ratio: float = 0.0, rank: int | None = None):
        return self.embed(texts), sp.csr_matrix((len(texts), 1))

    def encode_with_rho(self, texts: Sequence[str], min_known_ratio: float = 0.0):
        Z, X = self.encode(texts)
        return Z, X, np.ones(len(texts))


def neural_comparison(settings, lsa=None, verbose: bool = True) -> dict:
    """LSA (the served model and scoring) against MiniLM on the same FAQ held-out pairs and
    the same 1,000 random Quora questions. MiniLM uses plain cosine (no guards: rho and the
    known-word guard are LSA ideas). Its cosines have another scale, so it is reported at the
    profile's tau and at the tau that gives it the same precision LSA has at tau."""
    from .cache import SemanticCache
    from .data import faq_eval_pairs, load_faq
    from .embedder import LSAEmbedder
    from .evaluate import best_f1, roc_auc, sample_quora_questions, threshold_sweep
    from .measure import headline

    if lsa is None:
        from .pipeline import load_embedder
        lsa = load_embedder(settings)
    h_lsa = headline(settings, lsa)
    t0 = time.perf_counter()
    mini = MiniLMEmbedder()
    load_s = time.perf_counter() - t0
    items = load_faq(settings.path(settings.paths.faq_csv))
    pairs = faq_eval_pairs(items, seed=settings.eval.seed)
    labels = np.array([p.label for p in pairs])
    t0 = time.perf_counter()
    a, b = mini.embed([p.q1 for p in pairs]), mini.embed([p.q2 for p in pairs])
    embed_ms = (time.perf_counter() - t0) * 1000 / (2 * len(pairs))
    scores = np.einsum("ij,ij->i", a, b)
    sweep = threshold_sweep(scores, labels)
    best = best_f1(sweep)
    tau = settings.cache.threshold

    def at(t):
        pred = scores >= t
        tp, fp = int(np.sum(pred & (labels == 1))), int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        return {"tau": float(t), "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"), "accepted": tp + fp, "correct": tp}

    # the lowest tau at which MiniLM is at least as precise as LSA at its tau
    same_p = next((r for r in sorted(sweep, key=lambda r: r["tau"]) if r["precision"] >= h_lsa["precision"]
                   and r["recall"] > 0), None)
    random_hits = {}
    if settings.path(settings.paths.quora_eval).exists():
        questions = sample_quora_questions(settings, 1000)
        Q = mini.embed(questions)
        C = mini.embed([i.question for i in items])
        best_sim = (Q @ C.T).max(axis=1)
        for name, t in (("profile_tau", tau), ("best_f1_tau", best["tau"]),
                        ("same_precision_tau", same_p["tau"] if same_p else None)):
            if t is not None:
                random_hits[name] = {"tau": float(t), "hits": int(np.sum(best_sim >= t))}
    out = {
        "model": mini.name, "dim": mini.dim, "load_seconds": load_s, "embed_ms_per_text": embed_ms,
        "lsa": {k: h_lsa[k] for k in ("vocabulary_size", "k", "auc", "precision", "recall", "accepted_hits",
                                      "accepted_hits_correct", "best_f1", "best_f1_tau", "random_false_hits", "tau")},
        "minilm": {"auc": roc_auc(scores, labels), "best_f1": best["f1"], "best_f1_tau": best["tau"],
                   "at_profile_tau": at(tau), "at_same_precision": at(same_p["tau"]) if same_p else None,
                   "random_false_hits": random_hits},
    }
    if verbose:
        m = out["minilm"]
        print(f"  MiniLM: AUC {m['auc']:.3f}, at tau {tau:.2f} P {m['at_profile_tau']['precision']:.3f} "
              f"R {m['at_profile_tau']['recall']:.3f}; LSA AUC {h_lsa['auc']:.3f}")
    return out


def neural_markdown(r: dict) -> str:
    L, M = r["lsa"], r["minilm"]
    rh = M["random_false_hits"]
    a, s = M["at_profile_tau"], M["at_same_precision"]
    lines = [
        "## Neural baseline (MiniLM) vs LSA", "",
        f"{r['model']} ({r['dim']} dimensions, {r['embed_ms_per_text']:.1f} ms per text on this machine) vs the "
        f"served LSA model (k = {L['k']}, {L['vocabulary_size']:,} terms). Same 172 FAQ held-out pairs; "
        "random false hits: 1,000 Quora questions against the 43 FAQ answers.", "",
        "| model | vocabulary | k / dim | AUC | tau | P @ tau | R @ tau | accepted hits correct | best F1 (tau) | random false hits / 1,000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| LSA (served) | {L['vocabulary_size']:,} | {L['k']} | {L['auc']:.3f} | {L['tau']:.2f} | {L['precision']:.3f} | "
        f"{L['recall']:.3f} | {L['accepted_hits_correct']} of {L['accepted_hits']} | {L['best_f1']:.3f} ({L['best_f1_tau']:.2f}) | "
        f"{L['random_false_hits'] if L['random_false_hits'] is not None else 'n/a'} |",
        f"| MiniLM at the same tau | subword | {r['dim']} | {M['auc']:.3f} | {a['tau']:.2f} | {a['precision']:.3f} | "
        f"{a['recall']:.3f} | {a['correct']} of {a['accepted']} | {M['best_f1']:.3f} ({M['best_f1_tau']:.2f}) | "
        f"{rh.get('profile_tau', {}).get('hits', 'n/a')} |",
    ]
    if s:
        lines.append(f"| MiniLM at LSA's precision | subword | {r['dim']} | {M['auc']:.3f} | {s['tau']:.2f} | "
                     f"{s['precision']:.3f} | {s['recall']:.3f} | {s['correct']} of {s['accepted']} | "
                     f"{M['best_f1']:.3f} ({M['best_f1_tau']:.2f}) | {rh.get('same_precision_tau', {}).get('hits', 'n/a')} |")
    return "\n".join(lines) + "\n"
