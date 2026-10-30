"""The latent-coverage (rho) guard and latent stopwords."""

import numpy as np
import pytest

from semcache import linalg as la
from semcache.cache import SemanticCache
from semcache.config import PROJECT_ROOT, Settings
from semcache.embedder import EmbedderConfig, LSAEmbedder

TOPICS = {
    "laptop": "laptop battery charger screen keyboard",
    "flight": "flight ticket airport booking airline",
    "cooking": "recipe oven bake flour sugar",
    "garden": "plant soil seed water flower",
}


@pytest.fixture(scope="module")
def what_heavy():
    """Every document starts with "what"; "zebra" and "quokka" appear once each, so the
    SVD keeps the "what" direction and drops them."""
    rng = np.random.default_rng(0)
    corpus = []
    for words in TOPICS.values():
        w = words.split()
        corpus += ["what " + " ".join(rng.choice(w, size=3, replace=False)) for _ in range(12)]
    corpus += ["what is zebra", "what is quokka"]
    return LSAEmbedder(EmbedderConfig(rank=4, ngram_max=1, max_rank_fraction=0.5)).fit(corpus)


def _ask(cache, emb, text):
    z, x, rho = emb.encode_with_rho([text])
    return cache.lookup(z[0], query_text=text, query_tfidf=x[0], query_rho=float(rho[0]))


def _cache(emb, cutoff):
    cache = SemanticCache(emb.dim, threshold=0.8, min_latent_coverage=cutoff)
    for text in ["what is zebra", "what laptop battery charger"]:
        z, x, rho = emb.encode_with_rho([text])
        cache.add(text, "answer", z[0], tfidf_row=x[0], rho=float(rho[0]))
    return cache


def test_dominant_word_causes_a_false_hit_without_the_guard(what_heavy):
    r = _ask(_cache(what_heavy, 0.0), what_heavy, "what is quokka")
    assert r.hit and r.best.query == "what is zebra" and r.score > 0.99   # both collapse onto "what"


def test_rho_guard_blocks_the_false_hit_but_keeps_real_paraphrases(what_heavy):
    cache = _cache(what_heavy, 0.30)
    r = _ask(cache, what_heavy, "what is quokka")
    assert r.rho < 0.30 and not r.hit
    assert r.matches[0].scored_by == "tfidf" and r.matches[0].score < 0.2   # only "what" overlaps
    good = _ask(cache, what_heavy, "what about the laptop battery")
    assert good.rho >= 0.30 and good.hit and good.matches[0].scored_by == "svd"


def test_latent_coverage_is_the_share_of_the_row_kept_by_v_k():
    rng = np.random.default_rng(3)
    V, _ = np.linalg.qr(rng.standard_normal((30, 5)))          # test-only: orthonormal columns
    X = la.l2_normalize_rows(rng.standard_normal((4, 30)))
    rho = la.latent_coverage(X, V)
    np.testing.assert_allclose(rho, np.linalg.norm(X @ V, axis=1))   # unit rows: rho = ||x V||
    assert np.all((rho >= 0) & (rho <= 1))
    inside = V @ np.ones(5)                                   # a row in the span of V
    assert la.latent_coverage(inside[None, :], V)[0] == pytest.approx(1.0)


def test_guarded_scores_use_word_overlap_for_blocked_pairs():
    scores, blocked = la.guarded_scores(np.array([0.9, 0.9]), np.array([0.1, 0.1]),
                                        np.array([0.1, 0.8]), lsa_weight=0.5, min_coverage=0.3)
    assert blocked.tolist() == [True, False]
    np.testing.assert_allclose(scores, [0.1, 0.5])


def test_latent_stopwords_leave_the_svd_but_stay_in_tfidf(what_heavy):
    cfg = EmbedderConfig(rank=4, ngram_max=1, max_rank_fraction=0.5, use_latent_stopwords=True,
                         latent_stopwords=["what"])
    emb = LSAEmbedder(cfg).fit(["what " + w for w in TOPICS.values()] * 6 + ["what is zebra"])
    what = emb.vocab.index["what"]
    assert not emb.svd_mask[what] and np.all(emb.V[what] == 0)          # out of the SVD
    assert emb.tfidf(["what laptop"])[0, what] > 0                       # still in TF-IDF
    assert emb.latent_coverage(emb.tfidf(["what"]))[0] == 0.0            # nothing left to measure
    assert any("latent stopwords" in note for note in emb.report.notes)


# --- regression on the real Quora model --------------------------------------------

QUORA = PROJECT_ROOT / "data" / "quora" / "train.csv"


@pytest.fixture(scope="module")
def quora_service(tmp_path_factory):
    if not QUORA.exists():
        pytest.skip("data/quora is missing (python scripts/download_qqp.py)")
    import os

    from semcache.pipeline import SemanticCacheService, build_artifacts

    old = os.environ.get("SEMCACHE_ARTIFACTS")
    os.environ["SEMCACHE_ARTIFACTS"] = str(tmp_path_factory.mktemp("quora"))
    try:
        settings = Settings.load(PROJECT_ROOT / "config.toml", profile="quora")
        settings.llm.backend = "stub"
        embedder, cache = build_artifacts(settings, verbose=False, fresh=True)
        service = SemanticCacheService(embedder, cache, None, settings)
        # The Ollama answer that "how to enable dark mode" used to hit.
        z, x, rho = embedder.encode_with_rho(["i forgot my password, how can i reset it"])
        cache.add("i forgot my password, how can i reset it", "...", z[0], source="ollama",
                  tfidf_row=x[0], rho=float(rho[0]))
        yield service
    finally:
        if old is None:
            os.environ.pop("SEMCACHE_ARTIFACTS", None)
        else:
            os.environ["SEMCACHE_ARTIFACTS"] = old


@pytest.mark.parametrize("query, acceptable", [
    ("what is the capital of france", []),
    ("how to enable dark mode", []),
    ("what is svd", ["What is singular value decomposition?"]),
    ("what is phishing email ?", ["How can I recognise a phishing email?"]),
])
def test_quora_model_has_no_known_false_hits(quora_service, query, acceptable):
    r, _ = quora_service.lookup(query)
    assert not r.hit or r.best.query in acceptable, (r.best.query, r.score)
