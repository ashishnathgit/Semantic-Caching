"""Phase G: per-entry thresholds, Brand's incremental SVD, the MiniLM baseline (all off by default)."""

import numpy as np
import pytest

from semcache import linalg as la
from semcache.cache import SemanticCache
from semcache.config import PROJECT_ROOT, Settings
from semcache.pipeline import SemanticCacheService, build_artifacts
from semcache.thresholds import add_observation, learn_threshold


# ------------------------------------------------------------------ 27. per-entry thresholds
def test_learn_threshold_only_tightens():
    assert learn_threshold([], 0.7, 0.95) is None
    assert learn_threshold([[0.9, 1], [0.8, 1]], 0.7, 0.95) is None            # all correct: keep tau
    assert learn_threshold([[0.78, 0]], 0.7, 0.95) == pytest.approx(0.79)       # just above the wrong hit
    assert learn_threshold([[0.85, 1], [0.78, 0], [0.75, 1]], 0.7, 0.95) == pytest.approx(0.79)
    assert learn_threshold([[1.0, 0]], 0.7, 0.95) == 1.0


def test_settings_default_off(settings):
    assert not settings.cache.per_entry_thresholds
    assert not Settings.load(PROJECT_ROOT / "config.toml", profile="faq-live").cache.per_entry_thresholds


def test_cache_uses_the_entry_threshold():
    rng = np.random.default_rng(2)
    v = la.l2_normalize_rows(rng.standard_normal(16))
    q = la.l2_normalize_rows(v + 0.45 * la.l2_normalize_rows(rng.standard_normal(16)))
    score = float(q @ v)
    assert 0.7 < score < 0.98
    cache = SemanticCache(16, threshold=0.7, per_entry_thresholds=True)
    e = cache.add("cached", "answer", v)
    assert cache.lookup(q, record=False).hit
    cache.observe(e.id, score, False)
    r = cache.lookup(q, record=False)
    assert not r.hit and r.threshold_used == pytest.approx(round(score + 0.005, 2), abs=0.011)
    off = SemanticCache(16, threshold=0.7)
    e2 = off.add("cached", "answer", v)
    e2.threshold = 0.99                                    # ignored while the flag is off
    assert off.lookup(q, record=False).hit


def test_wrong_answer_raises_the_entry_threshold_when_on(settings):
    settings.cache.per_entry_thresholds = True
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    first = svc.ask("what is the best laptop for students")
    hit = svc.ask("best student laptop")
    assert hit.hit and hit.matched_id == first.stored_id
    entry = svc.cache.get(hit.matched_id)
    assert entry.feedback == [[pytest.approx(hit.score, abs=1e-4), 1]]
    out = svc.report_wrong("best student laptop", hit.matched_id, user="bo")
    assert not out["evicted"] and out["entry_threshold"] > hit.score
    assert entry.feedback == [[pytest.approx(hit.score, abs=1e-4), 0]]   # the positive was replaced
    again = svc.lookup("best student laptop")[0]
    assert not again.hit and again.threshold_used == out["entry_threshold"]
    reloaded = SemanticCacheService.from_settings(settings, verbose=False)
    assert reloaded.cache.get(hit.matched_id).threshold == out["entry_threshold"]


def test_add_observation_caps_history():
    class E:
        feedback = None
        threshold = None
    e = E()
    for i in range(80):
        add_observation(e, 0.9, True, 0.7, 0.95)
    assert len(e.feedback) == 50 and e.threshold is None


# ------------------------------------------------------------------ 28. Brand's update
def test_brand_update_matches_a_full_svd():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 50))
    A = rng.standard_normal((6, 50))
    U, s, V = la.exact_svd(X)
    U2, s2, V2 = la.brand_update_rows(U, s, V, A)
    full = np.linalg.svd(np.vstack([X, A]), compute_uv=False)        # NumPy only as the answer key
    assert np.allclose(s2, full[: len(s2)], atol=1e-9)
    assert np.allclose((U2 * s2) @ V2.T, np.vstack([X, A]), atol=1e-9)
    assert np.allclose(V2.T @ V2, np.eye(V2.shape[1]), atol=1e-9)
    _, _, Vf = la.exact_svd(np.vstack([X, A]))
    assert la.subspace_distance(V2[:, :10], Vf[:, :10]) < 1e-8
    _, sk, Vk = la.brand_update_rows(U, s, V, A, k=5)
    assert sk.shape == (5,) and Vk.shape == (50, 5)


def test_incremental_study_is_not_adopted(tmp_path, monkeypatch):
    from semcache.measure import incremental_svd_study
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path))
    s = Settings.load(PROJECT_ROOT / "config.toml", profile="faq-live")
    build_artifacts(s, verbose=False)
    r = incremental_svd_study(s, repeats=1)
    assert r["exact"]["sigma_max_abs_diff"] < 1e-8
    assert r["rebuild"]["new_terms"] > 0 and not r["matches_full_refit"] and not r["adopted"]


# ------------------------------------------------------------------ 29. MiniLM baseline
def test_minilm_behind_the_embedder_interface():
    pytest.importorskip("sentence_transformers")
    from semcache.neural import MiniLMEmbedder
    try:
        m = MiniLMEmbedder()
    except Exception as exc:                                  # offline and not downloaded yet
        pytest.skip(f"MiniLM weights unavailable: {exc}")
    Z, X, rho = m.encode_with_rho(["cheap flight tickets to goa", "How can I find cheap flights to Goa?"])
    assert Z.shape == (2, m.dim) and np.allclose(np.linalg.norm(Z, axis=1), 1.0)
    assert float(Z[0] @ Z[1]) > 0.7 and np.all(rho == 1.0)
    cache = SemanticCache(m.dim, threshold=0.7)
    cache.add("How can I find cheap flights to Goa?", "a", Z[1])
    assert cache.lookup(Z[0], record=False).hit
