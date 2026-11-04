"""Phase D: the negation guard (cache.negation_guard: on for faq-live, off on faq)."""

import pytest

from semcache.config import PROJECT_ROOT, Settings
from semcache.guards import TextGuard, names, negation_count, numbers
from semcache.pipeline import SemanticCacheService, build_artifacts

TAP = "Is it safe to drink tap water?"
TAP_A = ("In most cities tap water is treated and safe to drink; check your local water authority's reports, and "
         "boil or filter it when you travel somewhere the tap water is not treated.")


@pytest.fixture()
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path / "artifacts"))
    s = Settings.load(PROJECT_ROOT / "config.toml", profile="faq-live")
    s.llm.backend = "stub"
    s.cache.rebuild_every = 0
    build_artifacts(s, verbose=False)
    return SemanticCacheService.from_settings(s, verbose=False)


def drop(svc, question):
    entry = svc.cache.find_exact(question)
    assert entry is not None
    svc.remove_entry(entry.id)


# ------------------------------------------------------------------ the rules
def test_rules():
    g = TextGuard(["on/off", "enable/disable"])
    assert negation_count("I can't log in") == 1 and negation_count("I cannot log in") == 1
    assert g.check("is it not safe to drink tap water", "is it safe to drink tap water")["rule"] == "negation"
    assert g.check("why isn't it working", "why is it not working") is None          # same parity
    assert g.check("turn on notifications", "turn off notifications")["rule"] == "antonym"
    assert g.check("tips on saving money", "advice on saving money") is None
    assert g.check("top 5 laptops", "top 10 laptops")["rule"] == "number"
    assert numbers("three tips") == {"3"}
    assert g.check("flights to Goa", "flights to Mumbai")["rule"] == "name"
    assert g.check("cheap flight tickets to goa", "How can I find cheap flights to Goa?") is None
    assert names("Is Mac Or Windows Better For Coding") == set()                      # Title Case
    assert names("What is SVD?") == set()                                              # acronyms don't count
    assert g.check("why does python say module not found", "How do I fix ModuleNotFoundError in Python?") is None
    with pytest.raises(ValueError):
        TextGuard(["on-off"])


# ------------------------------------------------------------------ in the cache (faq-live)
def test_negated_question_is_blocked(live):
    live.add_entry(TAP, TAP_A, shared=True)
    live.request_rebuild(admin=True, wait=True)               # the cache teaches the words
    r = live.ask("is it not safe to drink tap water")
    assert not r.hit and r.answer_source == "LLM"
    assert r.score >= live.settings.cache.threshold            # the SVD alone would have served it
    assert "Negation guard (negation)" in r.note
    assert r.route["outcome"]["kind"] == "BLOCKED" and r.route["outcome"]["guard"] == "negation: negation"
    assert next(s for s in r.route["steps"] if s["id"] == "negation")["status"] == "fail"


def test_turn_on_vs_turn_off_is_blocked(live):
    drop(live, "How do I turn on notifications?")             # only the "off" answer is cached
    x = live.explain("turn on notifications")                  # before ask, which caches an answer
    assert x["verdict"] == "refused" and x["refused"]["rule"] == "antonym"
    r = live.ask("turn on notifications")
    assert r.score >= live.settings.cache.threshold and not r.hit     # 0.723 without the guard
    assert r.matches[0]["query"] == "How do I turn off notifications?"
    assert "antonym" in r.note


def test_plain_rewordings_still_hit(live):
    live.add_entry(TAP, TAP_A, shared=True)
    live.request_rebuild(admin=True, wait=True)
    for q, target in [("is tap water safe to drink", TAP),
                      ("cheap flight tickets to goa", "How can I find cheap flights to Goa?"),
                      ("what should I pack for a trip", "What should I pack for a short trip?"),
                      ("turn on notifications", "How do I turn on notifications?"),    # same polarity
                      ("turn off notifications for an app", "How do I turn off notifications?")]:
        r = live.ask(q)
        assert r.hit and r.matched_query == target, (q, r.score, r.matched_query, r.note)
        assert next(s for s in r.route["steps"] if s["id"] == "negation")["status"] == "pass"


def test_guard_is_off_on_faq(settings):
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    assert not settings.cache.negation_guard and svc.cache.text_guard is None
    drop(svc, "How do I turn on notifications?")
    r = svc.ask("turn on notifications")
    assert r.hit and r.matched_query == "How do I turn off notifications?"   # the documented faq weakness
