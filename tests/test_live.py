"""Phase C: the faq-live profile, whose vocabulary grows with the cache."""

import numpy as np
import pytest

from semcache.config import PROJECT_ROOT, Settings
from semcache.pipeline import SemanticCacheService, build_artifacts, build_corpus

GANDHINAGAR_Q = "What are the best routes to reach Gandhinagar"
# What an LLM such as llama3.2:3b typically answers: it restates the question, then lists routes.
GANDHINAGAR_A = ("The best routes to reach Gandhinagar depend on where you start. By air: fly to Ahmedabad airport, "
                 "about 20 km away, then travel to Gandhinagar by taxi or bus. By train: travel to Ahmedabad or "
                 "Gandhinagar Capital station. By road: Gandhinagar is well connected by highways, and buses run "
                 "often from Ahmedabad.")


@pytest.fixture()
def live(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path / "artifacts"))
    s = Settings.load(PROJECT_ROOT / "config.toml", profile="faq-live")
    s.llm.backend = "stub"
    s.cache.autosave_every = 1
    s.cache.rebuild_every = 0            # tests start rebuilds themselves unless they say otherwise
    return s


def test_faq_profile_does_not_grow(settings):
    assert not settings.corpus.include_cached
    corpus, info = build_corpus(settings)
    assert len(corpus) == 172 and info["cached_texts"] == 0
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    assert not (settings.model_dir / "corpus.json").exists()
    assert (svc.embedder.dim, len(svc.embedder.vocab)) == (60, 327)
    with pytest.raises(ValueError):
        svc.request_rebuild(admin=True)


def test_faq_live_fits_on_questions_and_answers(live):
    emb, cache = build_artifacts(live, verbose=False)
    assert emb.report.n_docs == 172 + 43                   # FAQ texts + the 43 cached FAQ answers
    assert len(emb.vocab) > 327 and len(cache) == 43
    assert (live.model_dir / "corpus.json").exists()


def test_private_and_placeholder_text_stay_out_of_the_corpus(live):
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    svc.add_entry("where is my secret locker", "zanzibarxyz combination", user="ann")      # private
    svc.ask("what is the best laptop for students")                                        # stub answer
    corpus, _ = build_corpus(live, entries=svc.cache.entries())
    text = " ".join(corpus).lower()
    assert "zanzibarxyz" not in text and "locker" not in text
    assert "placeholder answer" not in text and "what is the best laptop for students" in text


def test_rebuild_swaps_the_model_and_reembeds_zero_rows(live):
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    entry = svc.add_entry("zorblax quintessa fizzwick", "Zorblax quintessa fizzwick is a made-up phrase.")
    assert not np.any(svc.cache.vector_of(entry.id))        # unknown words: a zero row
    before = svc.embedder.fingerprint
    status = svc.request_rebuild(admin=True, wait=True)
    assert svc.rebuild_state["state"] == "done", svc.rebuild_state
    assert svc.embedder.fingerprint != before and status["generation"] == 1
    assert "zorblax" in svc.embedder.vocab
    assert np.any(svc.cache.vector_of(entry.id))            # the row got a real vector
    assert svc.cache.vectors.shape[1] == svc.embedder.dim
    reloaded = SemanticCacheService.from_settings(live, verbose=False)
    assert reloaded.embedder.fingerprint == svc.embedder.fingerprint


def test_rebuild_needs_an_admin(live):
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    with pytest.raises(PermissionError):
        svc.request_rebuild(admin=False)


def test_automatic_rebuild_after_n_new_entries(live):
    live.cache.rebuild_every = 3
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    for q in ("how do magnets work", "who wrote hamlet", "why is the sea salty"):
        assert svc.ask(q).stored_id is not None
    if svc._rebuild_thread is not None:
        svc._rebuild_thread.join(30)
    assert svc.rebuild_state["state"] == "done" and svc.rebuild_state["reason"] == "auto"
    assert svc.model_status()["new_since_build"] == 0


def test_a_vector_from_before_a_swap_is_recomputed(live):
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    old_gen = svc._model_gen
    vector, row, rho = svc._encode("how do magnets work")
    svc.add_entry("zorblax quintessa", "made-up words that make the vocabulary and k change")
    svc.request_rebuild(admin=True, wait=True)
    from semcache.backends import Generation
    gen = Generation("an answer", "stub", None, 0.0)
    entry_id = svc._store_shared("how do magnets work", gen, vector, row, rho, model_gen=old_gen)
    assert np.allclose(svc.cache.vector_of(entry_id), svc._encode("how do magnets work")[0])


def test_gandhinagar_vocabulary_grows_after_a_rebuild(live):
    """Regression: a cached answer about a new place teaches the model the place's name."""
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    svc.add_entry(GANDHINAGAR_Q, GANDHINAGAR_A, shared=True)
    assert "gandhinagar" not in svc.embedder.vocab
    before, _ = svc.lookup("how to travel to Gandhinagar")
    svc.request_rebuild(admin=True, wait=True)
    after, _ = svc.lookup("how to travel to Gandhinagar")
    assert "gandhinagar" in svc.embedder.vocab
    assert after.matches[0].entry.query == GANDHINAGAR_Q
    assert after.score > before.score + 0.3                  # 0.264 -> 0.663 in our run
    reach, _ = svc.lookup("best way to reach Gandhinagar")
    assert reach.hit and reach.best.query == GANDHINAGAR_Q


@pytest.mark.xfail(strict=True, reason=(
    "Not met: after the rebuild 'how to travel to Gandhinagar' scores 0.663 < tau 0.70. The word 'how' "
    "costs 0.056 (without it: 0.718). Latent stopwords would lift it to 0.93, but on faq-live they lower "
    "AUC, recall and add random false hits, so they stay off (see DECISIONS.md C4/C5)."))
def test_gandhinagar_reworded_question_hits_after_a_rebuild(live):
    build_artifacts(live, verbose=False)
    svc = SemanticCacheService.from_settings(live, verbose=False)
    svc.add_entry(GANDHINAGAR_Q, GANDHINAGAR_A, shared=True)
    svc.request_rebuild(admin=True, wait=True)
    after, _ = svc.lookup("how to travel to Gandhinagar")
    assert after.score > live.cache.threshold
