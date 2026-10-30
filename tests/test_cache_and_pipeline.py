import numpy as np
import pytest

from semcache import linalg as la
from semcache.backends import FallbackBackend, OllamaBackend, StubBackend
from semcache.cache import SemanticCache
from semcache.data import faq_corpus, faq_eval_pairs, load_faq
from semcache.embedder import EmbedderConfig, LSAEmbedder
from semcache.pipeline import SemanticCacheService, build_artifacts
from semcache.text import analyze, fold_plural, tokenize


# --- text ----------------------------------------------------------------------

def test_tokenizer_keeps_negations_question_words_and_on_off():
    assert tokenize("I don't want to turn OFF notifications") == ["not", "want", "turn", "off", "notification"]
    assert "how" in tokenize("How do I reset it?")
    assert fold_plural("batteries") == "battery" and fold_plural("analysis") == "analysis"
    assert "cheap flight" in analyze("cheap flights to Goa", ngram_max=2)


# --- embedder -----------------------------------------------------------------

@pytest.fixture(scope="module")
def faq_embedder():
    from semcache.config import PROJECT_ROOT
    items = load_faq(PROJECT_ROOT / "data" / "faq.csv")
    return items, LSAEmbedder().fit(faq_corpus(items))


def test_embedder_outputs_unit_vectors_and_caps_rank(faq_embedder):
    items, emb = faq_embedder
    E = emb.embed([i.question for i in items])
    assert E.shape == (len(items), emb.dim)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0)
    assert emb.dim <= int(0.35 * emb.report.n_docs)          # max_rank_fraction cap
    assert np.all(np.diff(emb.s) <= 1e-12)                    # singular values descending


def test_unknown_words_give_a_zero_vector(faq_embedder):
    _, emb = faq_embedder
    assert not np.any(emb.embed_one("zzqx blorp"))
    assert emb.known_terms("zzqx blorp") == []


def test_held_out_paraphrases_find_their_faq(faq_embedder):
    """Phrasings never seen in training should mostly match the right FAQ entry."""
    items, emb = faq_embedder
    E = emb.embed([i.question for i in items])
    held = [(h, idx) for idx, it in enumerate(items) for h in it.heldout]
    S = emb.embed([h for h, _ in held]) @ E.T
    accuracy = np.mean([S[i].argmax() == idx for i, (_, idx) in enumerate(held)])
    assert accuracy >= 0.75


def test_lsa_beats_raw_tfidf_on_a_synonym(faq_embedder):
    _, emb = faq_embedder
    a, b = "low cost airfare to goa", "How can I find cheap flights to Goa?"
    tfidf_cos = la.rowwise_cosine(emb.tfidf([a]), emb.tfidf([b]))[0]
    lsa_cos = float(emb.embed_one(a) @ emb.embed_one(b))
    assert lsa_cos > tfidf_cos


def test_embedder_save_load_round_trip(tmp_path, faq_embedder):
    _, emb = faq_embedder
    emb.save(tmp_path)
    loaded = LSAEmbedder.load(tmp_path)
    np.testing.assert_allclose(loaded.embed(["what is svd"]), emb.embed(["what is svd"]))
    assert loaded.fingerprint == emb.fingerprint


def test_smaller_rank_uses_nested_directions(faq_embedder):
    _, emb = faq_embedder
    assert emb.embed(["what is svd"], rank=10).shape == (1, 10)


# --- cache --------------------------------------------------------------------

def unit(v):
    return la.l2_normalize_rows(np.asarray(v, dtype=float))


def test_cache_hit_and_miss_follow_threshold():
    cache = SemanticCache(dim=3, threshold=0.9)
    cache.add("q1", "a1", unit([1, 0, 0]))
    near = unit([1, 0.2, 0])            # cos ~ 0.98
    far = unit([0, 1, 0])
    assert cache.lookup(near).hit
    assert not cache.lookup(far).hit
    assert not cache.lookup(near, threshold=0.99).hit
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 2


def test_cache_evicts_least_recently_used():
    cache = SemanticCache(dim=2, threshold=0.9, max_entries=2)
    a = cache.add("a", "A", unit([1, 0]))
    b = cache.add("b", "B", unit([0, 1]))
    cache.lookup(unit([1, 0]))          # touch "a", so "b" is least recently used
    cache.add("c", "C", unit([1, 1]))
    assert cache.get(a.id) is not None and cache.get(b.id) is None
    assert len(cache) == 2 and cache.evictions == 1


def test_cache_remove_keeps_matrix_contiguous():
    cache = SemanticCache(dim=2, threshold=0.5)
    ids = [cache.add(str(i), str(i), unit([1, i])).id for i in range(5)]
    assert cache.remove(ids[1])
    assert len(cache) == 4 and cache.vectors.shape == (4, 2)
    assert {e.query for e in cache.entries()} == {"0", "2", "3", "4"}
    assert not cache.remove(999)


def test_lsh_lookup_agrees_with_brute_force_for_near_duplicates():
    rng = np.random.default_rng(0)
    data = unit(rng.standard_normal((3000, 24)))
    brute = SemanticCache(dim=24, threshold=0.9, max_entries=10_000)
    lsh = SemanticCache(dim=24, threshold=0.9, max_entries=10_000, use_lsh=True,
                        lsh_bits=10, lsh_tables=12, lsh_min_entries=100)
    for i, v in enumerate(data):
        brute.add(str(i), str(i), v)
        lsh.add(str(i), str(i), v)
    agree, compared = 0, []
    for i in range(50):
        q = unit(data[i] + 0.05 * rng.standard_normal(24))
        r1, r2 = brute.lookup(q), lsh.lookup(q)
        agree += int(r2.hit and r1.best.query == r2.best.query)
        compared.append(r2.compared)
    assert agree >= 45
    assert np.mean(compared) < 0.5 * len(data)       # LSH scores far fewer vectors


def test_cache_save_load_round_trip(tmp_path):
    cache = SemanticCache(dim=2, threshold=0.8)
    cache.add("hello", "world", unit([1, 0]), source="manual")
    cache.lookup(unit([1, 0]))
    cache.save(tmp_path)
    loaded = SemanticCache.load(tmp_path)
    assert len(loaded) == 1 and loaded.entries()[0].answer == "world"
    assert loaded.lookup(unit([1, 0])).hit
    assert loaded.hits == 2


# --- backends -----------------------------------------------------------------

def test_fallback_uses_stub_when_ollama_is_unreachable():
    backend = FallbackBackend(OllamaBackend("http://127.0.0.1:9", timeout_s=1), StubBackend())
    gen = backend.generate("hello")
    assert gen.backend == "stub" and "Placeholder" in gen.text
    assert backend.last_error and "cannot reach Ollama" in backend.last_error
    assert backend.describe()["available"] is False


# --- pipeline -----------------------------------------------------------------

def test_service_miss_then_hit_and_faq_warm_start(settings):
    build_artifacts(settings, verbose=False)
    service = SemanticCacheService.from_settings(settings, verbose=False)
    n_faq = service.stats()["entries"]
    assert n_faq == len(load_faq(settings.path(settings.paths.faq_csv)))

    faq_hit = service.ask("forgot my password, how can I reset it")
    assert faq_hit.hit and faq_hit.matched_source == "faq"

    first = service.ask("what is the capital of the netherlands")
    assert not first.hit and first.source == "stub" and first.stored_id
    again = service.ask("What is the capital of the Netherlands?")
    assert again.hit and again.matched_id == first.stored_id
    assert again.timings_ms["llm"] == 0.0


def test_service_persists_and_resets(settings):
    service = SemanticCacheService.from_settings(settings, verbose=False)   # builds on first use
    entry = service.add_entry("what is our wifi password", "It's on the router label.")
    reloaded = SemanticCacheService.from_settings(settings, verbose=False)
    assert reloaded.cache.get(entry.id) is not None
    n = reloaded.reset()
    assert reloaded.stats()["entries"] == n and reloaded.cache.get(entry.id) is None


def test_service_rejects_empty_and_long_queries(settings):
    service = SemanticCacheService.from_settings(settings, verbose=False)
    with pytest.raises(ValueError):
        service.ask("   ")
    with pytest.raises(ValueError):
        service.ask("x" * 2000)


def test_faq_eval_pairs_are_balanced(faq_path):
    pairs = faq_eval_pairs(load_faq(faq_path), negatives_per_item=2)
    labels = [p.label for p in pairs]
    assert 0 < sum(labels) < len(labels)


def test_unknown_word_queries_only_hit_on_exact_repeats(settings):
    """Regression: with only 'what' known, two different questions had identical vectors."""
    service = SemanticCacheService.from_settings(settings, verbose=False)
    first = service.ask("what is the capital of the netherlands")
    assert not first.hit
    other = service.ask("what is the capital of france")
    assert not other.hit and "words are known" in (other.note or "")
    repeat = service.ask("What is the capital of the Netherlands?")
    assert repeat.hit and repeat.lookup_method == "exact" and repeat.score == 1.0


def test_exact_layer_replaces_duplicate_questions():
    cache = SemanticCache(dim=2, threshold=0.9)
    old = cache.add("Reset password?", "old", unit([1, 0]))
    new = cache.add("reset   PASSWORD", "new", unit([1, 0]))
    assert len(cache) == 1 and cache.get(old.id) is None
    r = cache.lookup(np.zeros(2), query_text="reset password")
    assert r.hit and r.method == "exact" and r.best.id == new.id


def test_hybrid_scoring_reports_both_parts(faq_embedder):
    items, emb = faq_embedder
    vectors, rows = emb.encode([i.question for i in items])
    cache = SemanticCache(emb.dim, threshold=0.5, lsa_weight=0.5)
    cache.add_many([(i.question, i.answer) for i in items], vectors, tfidf_rows=rows)
    qv, qrow = emb.encode(["how do i reset my password"])
    r = cache.lookup(qv[0], query_tfidf=qrow[0])
    best = r.matches[0]
    assert best.lsa is not None and best.tfidf is not None
    assert best.score == pytest.approx(0.5 * best.lsa + 0.5 * best.tfidf)
    assert "password" in best.entry.query.lower()


def test_cache_save_load_keeps_tfidf_rows(tmp_path, faq_embedder):
    items, emb = faq_embedder
    vectors, rows = emb.encode([i.question for i in items[:5]])
    cache = SemanticCache(emb.dim, threshold=0.5, lsa_weight=0.5)
    cache.add_many([(i.question, i.answer) for i in items[:5]], vectors, tfidf_rows=rows)
    cache.save(tmp_path)
    loaded = SemanticCache.load(tmp_path, lsa_weight=0.5)
    qv, qrow = emb.encode([items[2].question + " please"])
    a = cache.lookup(qv[0], query_tfidf=qrow[0], record=False)
    b = loaded.lookup(qv[0], query_tfidf=qrow[0], record=False)
    assert a.best.id == b.best.id and a.score == pytest.approx(b.score)


def test_rebuild_keeps_cached_answers_unless_fresh(settings):
    service = SemanticCacheService.from_settings(settings, verbose=False)
    entry = service.add_entry("what time does the canteen open", "8 am on weekdays.")
    settings.embedder.rank = 30                         # a different model forces re-embedding
    build_artifacts(settings, verbose=False)
    kept = SemanticCacheService.from_settings(settings, verbose=False)
    assert any(e.query == entry.query for e in kept.cache.entries())
    assert kept.ask("What time does the canteen open?").hit
    build_artifacts(settings, verbose=False, fresh=True)
    fresh = SemanticCacheService.from_settings(settings, verbose=False)
    assert not any(e.query == entry.query for e in fresh.cache.entries())


# --- regressions --------------------------------------------------------------

@pytest.mark.parametrize("plural, singular", [
    ("flights", "flight"), ("batteries", "battery"), ("cookies", "cookie"), ("movies", "movie"),
    ("boxes", "box"), ("watches", "watch"), ("wishes", "wish"), ("classes", "class"),
    ("caches", "cache"), ("ties", "tie"),
])
def test_plural_and_singular_fold_to_the_same_term(plural, singular):
    assert fold_plural(plural) == fold_plural(singular)


def test_left_curly_quote_still_expands_contractions():
    assert tokenize("don‘t") == tokenize("don't") == tokenize("don’t") == ["not"]


@pytest.mark.parametrize("tau", [0.0, -1.0, 1.5])
def test_lookup_rejects_thresholds_outside_zero_one(tau):
    cache = SemanticCache(dim=2)
    cache.add("q", "a", unit([1, 0]))
    with pytest.raises(ValueError, match="threshold"):
        cache.lookup(unit([0, 1]), threshold=tau)


def test_cache_load_reads_each_tfidf_array_once(tmp_path, monkeypatch):
    cache = SemanticCache(dim=2, max_entries=500)
    for i in range(300):
        cache.add(f"question {i}", "a", unit([1, i]), tfidf_row=(np.array([i, i + 1]), np.array([0.6, 0.8])))
    cache.save(tmp_path)

    reads = []
    real_load = np.load

    class CountingNpz:
        def __init__(self, archive):
            self.archive = archive

        def __getitem__(self, name):
            reads.append(name)
            return self.archive[name]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.archive.close()

    def counting_load(path, *args, **kwargs):
        out = real_load(path, *args, **kwargs)
        return CountingNpz(out) if str(path).endswith(".npz") else out

    monkeypatch.setattr(np, "load", counting_load)
    loaded = SemanticCache.load(tmp_path)
    assert len(reads) == 3                              # indptr, indices, data, once each
    slot = loaded._slot_of[loaded.find_exact("question 299").id]
    np.testing.assert_array_equal(loaded._tfidf[slot][0], [299, 300])


def test_ask_keeps_answer_stored_by_a_concurrent_request(settings):
    build_artifacts(settings, verbose=False)
    service = SemanticCacheService.from_settings(settings, verbose=False)
    question = "what colour is the office cat"

    class RacingBackend(StubBackend):
        def generate(self, prompt):
            # Another request for the same question finishes first, while this one waits on the LLM.
            service.add_entry(question, "first answer", source="ollama")
            return super().generate(prompt)

    service.backend = RacingBackend()
    result = service.ask(question)
    assert not result.hit
    stored = service.cache.find_exact(question)
    assert stored.answer == "first answer" and result.stored_id == stored.id
    assert sum(e.query == question for e in service.cache.entries()) == 1


def test_rank_note_names_the_limit_that_applied(faq_embedder):
    items, _ = faq_embedder
    corpus = faq_corpus(items)
    emb = LSAEmbedder(EmbedderConfig(rank=10_000, max_rank_fraction=1.0)).fit(corpus)
    assert any("smaller than both sides" in n for n in emb.report.notes)
    emb = LSAEmbedder(EmbedderConfig(rank=10_000, max_rank_fraction=0.1)).fit(corpus)
    assert any("max_rank_fraction = 0.1" in n for n in emb.report.notes)
