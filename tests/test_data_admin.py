"""Phase E: import an FAQ CSV as a new profile, forget LLM rows, TTL, "wrong answer"."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from semcache.cache import SemanticCache
from semcache.config import PROJECT_ROOT, Settings
from semcache.pipeline import SemanticCacheService

CAMPUS = """question,answer
Where is the exam hall?,Room 204 on the second floor of the academic block.
When does the library close?,The library closes at 10 pm on weekdays and 6 pm on weekends.
How do I get a student ID card?,Apply at the admin office with a photo and your admission letter.
Where can I park my bike?,Bike parking is behind the hostel blocks and next to the main gate.
How do I reset my campus wifi password?,Use the self-service portal at wifi.campus.example with your student ID.
When is the canteen open?,The canteen is open from 7 am to 9 pm every day.
How do I book a room in the library?,Book library study rooms online up to a week ahead with your student ID.
Where is the medical centre?,The medical centre is next to the sports ground and is open 24 hours.
How do I pay my hostel fees?,Pay hostel fees online through the student portal before the 10th of each month.
Who do I contact about exam results?,Write to the exam office; results queries close two weeks after publication.
Can I borrow a laptop from the library?,The library lends laptops for up to three days with your student ID.
What time does the gym open?,The gym opens at 6 am and closes at 10 pm on weekdays.
"""


@pytest.fixture()
def svc(settings):
    return SemanticCacheService.from_settings(settings, verbose=False)


# ------------------------------------------------------------------ 20. import
def test_import_creates_a_new_profile_and_leaves_faq_alone(settings, svc):
    config_before = (PROJECT_ROOT / "config.toml").read_bytes()
    faq_model = (settings.model_dir / "embedder.npz").read_bytes()
    out = svc.import_csv("campus", CAMPUS, admin=True)
    assert out["profile"] == "campus" and out["entries"] == 12 and out["k"] >= 2
    assert (PROJECT_ROOT / "config.toml").read_bytes() == config_before
    assert (settings.model_dir / "embedder.npz").read_bytes() == faq_model
    campus = Settings.load(PROJECT_ROOT / "config.toml", profile="campus")
    assert "campus" in campus.profiles and campus.artifacts_dir.name == "campus"
    assert not campus.corpus.use_quora
    other = SemanticCacheService.from_settings(campus, verbose=False)
    r = other.ask("when does the library shut")
    assert r.hit and r.matched_query == "When does the library close?"
    assert len(svc.cache) == 43                                  # the served faq cache is unchanged


@pytest.mark.parametrize("name", ["faq", "faq-live", "quora"])
def test_import_never_replaces_a_configured_profile(svc, name):
    with pytest.raises(ValueError):
        svc.import_csv(name, CAMPUS, admin=True)


def test_import_checks(svc):
    with pytest.raises(PermissionError):
        svc.import_csv("campus", CAMPUS, admin=False)
    with pytest.raises(ValueError, match="question and answer"):
        svc.import_csv("campus", "q,a\nx,y\n", admin=True)
    with pytest.raises(ValueError, match="letters"):
        svc.import_csv("../evil", CAMPUS, admin=True)


def test_import_over_the_api(settings, monkeypatch):
    from apps.api import create_app
    monkeypatch.setenv("SEMCACHE_ADMIN_KEY", "k")
    with TestClient(create_app(settings)) as c:
        c.post("/api/session", json={"name": "ada"})
        h = {"X-User": "ada"}
        assert c.post("/api/admin/import-csv", json={"profile": "campus", "csv": CAMPUS}, headers=h).status_code == 403
        r = c.post("/api/admin/import-csv", json={"profile": "campus", "csv": CAMPUS}, headers={**h, "X-Admin-Key": "k"})
        assert r.status_code == 201 and r.json()["serve"] == "python -m semcache serve --profile campus"


# ------------------------------------------------------------------ 21. forget
def test_forget_llm_rows_keeps_faq_and_manual(svc):
    assert svc.ask("what is the best laptop for students").stored_id is not None      # stub answer
    svc.add_entry("where is the exam hall", "Room 204.")                               # by hand
    with pytest.raises(PermissionError):
        svc.forget_llm_rows(admin=False)
    out = svc.forget_llm_rows(admin=True)
    assert out["removed"] == 1 and out["entries"] == 44
    assert {e.source for e in svc.cache.entries()} == {"faq", "manual"}


# ------------------------------------------------------------------ 22. TTL
def test_ttl_is_off_by_default(settings, svc):
    assert settings.cache.ttl_seconds == 0
    assert all(e.expires_at is None for e in svc.cache.entries())


def test_expired_entries_are_misses_and_survive_save(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2, 8))
    cache = SemanticCache(8, threshold=0.9)
    keep = cache.add("kept question", "a", v[0])
    gone = cache.add("short lived", "b", v[1], ttl_seconds=0.05)
    assert gone.expires_at is not None and keep.expires_at is None
    assert cache.lookup(v[1], query_text="short lived", record=False).hit
    cache.save(tmp_path)
    loaded = SemanticCache.load(tmp_path)
    assert loaded.get(gone.id).expires_at == pytest.approx(gone.expires_at)
    time.sleep(0.08)
    for c in (cache, loaded):
        r = c.lookup(v[1], query_text="short lived", record=False)
        assert not r.hit and r.method != "exact"                  # neither exact nor semantic
        assert c.lookup(v[0], query_text="kept question", record=False).hit
    assert cache.purge_expired() == 1 and len(cache) == 1


def test_a_new_answer_replaces_an_expired_one(svc):
    e = svc.add_entry("what time is the bus", "8 am", ttl_seconds=0.05)
    time.sleep(0.08)
    r = svc.ask("what time is the bus")
    assert not r.hit and r.stored_id is not None and svc.cache.get(e.id) is None


# ------------------------------------------------------------------ 23. wrong answer
def test_wrong_answer_evicts_and_logs(svc):
    first = svc.ask("what is the best laptop for students")
    hit = svc.ask("best student laptop")
    assert hit.hit and hit.matched_id == first.stored_id
    out = svc.report_wrong("best student laptop", hit.matched_id, user="bo")
    assert out["evicted"] and svc.cache.get(hit.matched_id) is None
    line = json.loads(svc.feedback_path.read_text().strip().splitlines()[-1])
    assert line["query"] == "best student laptop" and line["score"] == pytest.approx(hit.score, abs=1e-3)
    assert line["tau"] == 0.70 and line["cached_query"] == "what is the best laptop for students"


def test_wrong_answer_on_faq_needs_an_admin_to_evict(svc):
    hit = svc.ask("cheap flight tickets to goa")
    out = svc.report_wrong("cheap flight tickets to goa", hit.matched_id, user="bo")
    assert not out["evicted"] and svc.cache.get(hit.matched_id) is not None
    out = svc.report_wrong("cheap flight tickets to goa", hit.matched_id, user="bo", admin=True)
    assert out["evicted"] and svc.cache.get(hit.matched_id) is None
    assert len(svc.feedback_path.read_text().splitlines()) == 2


def test_wrong_answer_cannot_touch_someone_elses_private_entry(svc):
    e = svc.add_entry("my locker code", "1234", user="ann")
    with pytest.raises(LookupError):
        svc.report_wrong("my locker code", e.id, user="bo")


# ------------------------------------------------------------------ admin: delete cached questions
def test_admin_deletes_several_entries_of_any_owner(svc):
    mine = svc.add_entry("my locker code", "1234", user="ann")                    # private to ann
    learned = svc.cache.get(svc.ask("what is the best laptop for students").stored_id)
    faq = svc.cache.find_exact("How do I delete my account?")
    with pytest.raises(PermissionError):
        svc.remove_entries([learned.id], admin=False)
    out = svc.remove_entries([mine.id, learned.id, faq.id, 99999, learned.id], admin=True)
    assert out["removed"] == 3 and out["missing"] == [99999] and out["entries"] == 42
    reloaded = SemanticCache.load(svc.settings.cache_dir)                        # saved once, at the end
    assert all(reloaded.get(i) is None for i in (mine.id, learned.id, faq.id))


def test_delete_entries_endpoint_needs_the_admin_key(settings, monkeypatch):
    from apps.api import create_app
    monkeypatch.setenv("SEMCACHE_ADMIN_KEY", "k")
    with TestClient(create_app(settings)) as c:
        c.post("/api/session", json={"name": "ada"})
        h = {"X-User": "ada"}
        ids = [e["id"] for e in c.get("/api/entries", headers=h).json()[:2]]
        assert c.post("/api/admin/entries/delete", json={"ids": ids}, headers=h).status_code == 403
        r = c.post("/api/admin/entries/delete", json={"ids": ids}, headers={**h, "X-Admin-Key": "k"})
        assert r.status_code == 200 and r.json()["removed"] == 2
        assert len(c.get("/api/entries", headers=h).json()) == 41
