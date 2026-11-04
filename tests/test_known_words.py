"""The known-word guard's topic-word rule (cache.relaxed_known_words; on for faq and quora,
off for faq-live). A wordy question full of everyday words the model never saw gets a
meaning-based match on its known topic words; off-topic questions stay blocked."""

import numpy as np
import pytest

from semcache.config import PROJECT_ROOT, Settings
from semcache.embedder import KnownWordPolicy
from semcache.pipeline import SemanticCacheService, load_embedder

WORDY = "Where do I go to generate a new password if I lost the old one?"
RESET = "How do I reset my password if I forgot it?"


@pytest.fixture()
def svc(settings):
    return SemanticCacheService.from_settings(settings, verbose=False)


def test_wordy_question_hits_the_right_answer(svc):
    adm = svc.embedder.admission(WORDY, 0.5)
    assert (adm.known, adm.total, adm.status) == (2, 8, "relaxed")
    assert adm.topic_known == ["new", "password"] and adm.topic_unknown == ["generate", "lost", "old"]
    assert set(adm.everyday_unknown) == {"where", "go", "one"}
    r = svc.ask(WORDY)
    assert r.hit and r.matched_query == RESET and r.score >= 0.70
    assert "everyday words" in r.note and '"new", "password"' in r.note
    step = next(s for s in r.route["steps"] if s["id"] == "known")
    assert step["status"] == "pass" and "topic 2/5" in step["value"]


@pytest.mark.parametrize("q", ["How do I change my password if I forgot it?", "how to recorrect my password",
                               "forgot Password recovery instructions"])
def test_questions_that_already_passed_are_unchanged(svc, q):
    adm = svc.embedder.admission(q, 0.5)
    assert adm.status == "ok"
    r = svc.ask(q)
    assert r.hit and r.matched_query == RESET


@pytest.mark.parametrize("q", ["what is the capital of france",            # preset 6
                               "Why are some more intelligent than others?",
                               "How do you find someone phone by GPS?"])
def test_off_topic_questions_stay_blocked(svc, q):
    adm = svc.embedder.admission(q, 0.5)
    assert adm.status == "blocked"
    r = svc.ask(q, use_llm=False)
    assert not r.hit and r.route["outcome"]["guard"] == "known words"
    assert "Topic words it doesn't know" in r.note


def test_relaxed_questions_are_matched_on_topic_words_only(svc):
    emb = svc.embedder
    Z, X = emb.encode([WORDY], 0.5)
    terms = {emb.vocab.terms[i] for i in X[0].indices}
    assert terms == {"new", "password"} and np.isclose(np.linalg.norm(X[0].data), 1.0)
    Zt, _ = emb.encode(["new password"], 0.5)
    assert np.allclose(Z[0], Zt[0])


def test_explain_agrees_with_ask(svc):
    x = svc.explain(WORDY)
    assert x["verdict"] == "hit" and x["coverage"]["relaxed"] and x["coverage"]["passed"]
    assert x["matches"][0]["query"] == RESET
    assert x["score"] == pytest.approx(svc.lookup(WORDY)[0].score, abs=1e-4)


def test_the_rule_can_be_turned_off(settings):
    settings.cache.relaxed_known_words = False
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    assert svc.embedder.policy is None
    assert svc.embedder.admission(WORDY, 0.5).status == "blocked"
    assert not svc.ask(WORDY, use_llm=False).hit


def test_cache_is_reembedded_when_the_rule_changes(settings):
    settings.cache.relaxed_known_words = False
    off = SemanticCacheService.from_settings(settings, verbose=False)
    entry = off.add_entry(WORDY, "Use the Forgot password link.")
    assert not np.any(off.cache.vector_of(entry.id))           # blocked: a zero row
    off.save()
    settings.cache.relaxed_known_words = True
    on = SemanticCacheService.from_settings(settings, verbose=False)
    assert on.cache.query_policy == on.embedder.policy_signature != ""
    assert np.any(on.cache.vector_of(entry.id))                # re-embedded under the new rule


def test_profiles():
    for name, on in (("faq", True), ("quora", True), ("faq-live", False)):
        assert Settings.load(PROJECT_ROOT / "config.toml", profile=name).cache.relaxed_known_words is on


def test_policy_signature_tracks_its_settings():
    assert KnownWordPolicy(2, 0.4).signature != KnownWordPolicy(3, 0.4).signature


def test_faq_numbers_with_the_rule(settings, svc):
    from semcache.measure import known_word_guard_study
    r = known_word_guard_study(settings, load_embedder(settings))
    plain, topic = r["rows"]
    assert plain["heldout"]["right"] == topic["heldout"]["right"] == 57    # held-out phrasings unchanged
    assert topic["wordy"]["right"] == topic["wordy"]["asked"] and topic["wordy"]["wrong"] == 0
    if "random" in topic:
        assert topic["random"]["hits"] - plain["random"]["hits"] <= 2
