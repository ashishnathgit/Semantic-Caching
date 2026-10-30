import numpy as np
import pytest

from semcache.pipeline import SemanticCacheService


@pytest.fixture()
def service(settings):
    return SemanticCacheService.from_settings(settings, verbose=False)


def test_pair_parts_add_up_to_the_cache_score(service):
    x = service.explain("i forgot my password, how do i reset it")
    assert x["verdict"] in ("hit", "miss") and x["matches"]
    for m in x["matches"]:
        if m["blocked"]:
            continue
        total = float(np.sum(m["pairs"]["contribution"]))
        assert total == pytest.approx(m["lsa"], abs=2e-3)       # cells are rounded to 4 places
        assert m["computed"] == pytest.approx(m["score"], abs=1e-3)
        # the per-direction parts add up to the same cosine
        e, f = np.array(x["latent"]), np.array(m["c_latent"])
        assert float(e @ f) == pytest.approx(m["lsa"], abs=2e-3)


def test_matches_follow_the_real_lookup(service):
    q = "my laptop is really slow"
    x = service.explain(q, threshold=0.5)
    real, _ = service.lookup(q, threshold=0.5)
    assert [m["id"] for m in x["matches"]] == [m.entry.id for m in real.matches]
    assert x["hit"] == real.hit
    assert x["matches"][0]["hit"] == real.hit


def test_word_effect_is_the_score_without_that_word(service):
    x = service.explain("cheap flight tickets to goa")
    m = x["matches"][0]
    for eff in m["effects"]:
        assert eff["effect"] == pytest.approx(m["score"] - eff["without"], abs=2e-4)


def test_tokens_show_stopwords_unknown_and_folded_words(service):
    x = service.explain("the flights to zzqxville")
    status = {t["word"]: t["status"] for t in x["tokens"]}
    assert status["the"] == "stopword" and status["zzqxville"] == "unknown"
    flights = next(t for t in x["tokens"] if t["word"] == "flights")
    assert flights["term"] == "flight"


def test_unknown_words_and_exact_repeats(service):
    assert service.explain("zzqx blorp")["verdict"] == "no_known_words"
    faq_question = service.visible_entries()[0].query
    x = service.explain(faq_question)
    assert x["verdict"] == "exact" and x["matches"][0]["exact"]


def test_explain_does_not_store_or_count(service):
    before = service.stats()
    service.explain("how many moons does jupiter have")
    after = service.stats()
    assert after["entries"] == before["entries"]
    assert (after["hits"], after["misses"]) == (before["hits"], before["misses"])


def test_stage_data_is_small_and_real(service):
    x = service.explain("cheap flight tickets to goa")
    k = service.embedder.dim
    assert x["term_rows"] and all(len(r["vk"]) == k for r in x["term_rows"])
    idx = [r["index"] for r in x["term_rows"]]
    np.testing.assert_allclose([r["vk"] for r in x["term_rows"]], np.round(service.embedder.V[idx], 3))
    assert 0 < len(x["vk_band"]) <= 24 and all(len(r["vk"]) == k for r in x["vk_band"])
    C = x["cache_matrix"]
    assert C["n"] == len(service.cache) == C["shown"] and not C["sampled"]        # 43 rows: drawn whole
    best = max(C["rows"], key=lambda r: r["score"])
    assert best["rank"] == 1 and best["id"] == x["matches"][0]["id"]
    assert best["score"] == pytest.approx(x["matches"][0]["score"], abs=1e-3)


def test_large_cache_is_sampled(service):
    for i in range(30):
        service.add_entry(f"how do i fix error number {i} in python", "answer")
    C = service.explain("how do i fix a python error")["cache_matrix"]
    assert C["n"] == 73 and C["sampled"] and C["shown"] == 40
    assert sum(r["rank"] is not None for r in C["rows"]) == 3                   # the top rows are kept


def test_ask_returns_the_new_row_of_c(service):
    before = len(service.cache)
    r = service.ask("which planets have rings")
    assert r.stored_id and r.cache_rows == before + 1
    np.testing.assert_allclose(r.stored_vector, service.cache.vector_of(r.stored_id), atol=1e-4)
    hit = service.ask("which planets have rings?")
    assert hit.hit and hit.stored_vector is None and hit.cache_rows == before + 1
