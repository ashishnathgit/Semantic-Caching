"""Admin passphrase, privacy of other users' questions, and model profiles."""

import pytest

from semcache.config import PROJECT_ROOT, Settings
from semcache.pipeline import HIDDEN_QUESTION, build_artifacts

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from apps.api import create_app  # noqa: E402

KEY = "correct horse battery staple"
ALICE, BOB = {"X-User": "alice"}, {"X-User": "bob"}
ADMIN = {"X-User": "carol", "X-Admin-Key": KEY}


@pytest.fixture()
def make_client(settings, monkeypatch):
    def make(admin_key: str | None):
        if admin_key is None:
            monkeypatch.delenv("SEMCACHE_ADMIN_KEY", raising=False)
        else:
            monkeypatch.setenv("SEMCACHE_ADMIN_KEY", admin_key)
        return TestClient(create_app(settings))
    return make


# --- admin ----------------------------------------------------------------------

def test_admin_actions_are_off_without_the_env_var(make_client):
    with make_client(None) as c:
        assert c.get("/api/health").json()["admin_enabled"] is False
        r = c.post("/api/reset", headers={**ALICE, "X-Admin-Key": "anything"})
        assert r.status_code == 403 and "SEMCACHE_ADMIN_KEY" in r.json()["detail"]
        faq = next(e for e in c.get("/api/entries", headers=ALICE).json() if e["source"] == "faq")
        r = c.delete(f"/api/entries/{faq['id']}", headers=ALICE)
        assert r.status_code == 403 and "turned off" in r.json()["detail"]
        r = c.post("/api/entries", json={"question": "where is block c", "answer": "x", "shared": True},
                   headers=ALICE)
        assert r.status_code == 403


def test_admin_key_is_needed_and_checked(make_client):
    with make_client(KEY) as c:
        assert c.get("/api/health").json()["admin_enabled"] is True
        wrong = c.post("/api/reset", headers={**ALICE, "X-Admin-Key": "nope"})
        assert wrong.status_code == 403 and wrong.json()["detail"] == "wrong admin key"
        named = c.post("/api/reset", headers={"X-User": "admin"})          # the name is not enough
        assert named.json()["scope"] == "private"
        assert c.get("/api/me", headers=ADMIN).json()["is_admin"] is True
        assert c.post("/api/reset", headers=ADMIN).json()["scope"] == "all"
        r = c.post("/api/entries", json={"question": "where is block c", "answer": "x", "shared": True},
                   headers=ADMIN)
        assert r.status_code == 201 and r.json()["owner"] is None
        assert c.delete(f"/api/entries/{r.json()['id']}", headers=ADMIN).status_code == 200


# --- privacy --------------------------------------------------------------------

def test_users_see_faq_and_their_own_questions_only(make_client):
    with make_client(KEY) as c:
        q = "how many moons does jupiter have"
        mine = c.post("/api/ask", json={"query": q}, headers=ALICE).json()
        assert not mine["hit"] and mine["stored_id"]

        alice_ids = {e["id"] for e in c.get("/api/entries", headers=ALICE).json()}
        bob_list = c.get("/api/entries", headers=BOB).json()
        assert mine["stored_id"] in alice_ids
        assert mine["stored_id"] not in {e["id"] for e in bob_list}
        assert all(e["source"] == "faq" for e in bob_list)
        assert mine["stored_id"] in {e["id"] for e in c.get("/api/entries", headers=ADMIN).json()}

        # Bob still gets the shared answer, but not the text of Alice's question.
        bob = c.post("/api/ask", json={"query": q + "?"}, headers=BOB).json()
        assert bob["hit"] and bob["matched_query"] == HIDDEN_QUESTION and bob["matches"][0]["hidden"]
        # (its words aren't in the FAQ vocabulary, so only the exact repeat matches)
        look = c.post("/api/lookup", json={"query": q.upper()}, headers=BOB).json()
        assert look["matches"][0]["hidden"] and look["matches"][0]["query"] == HIDDEN_QUESTION
        x = c.post("/api/explain", json={"query": q.upper()}, headers=BOB).json()
        hidden = [m for m in x["matches"] if m["hidden"]]
        assert hidden and all(m["query"] == HIDDEN_QUESTION and m["pairs"] is None and m["terms"] == []
                              for m in hidden)
        # Alice and the admin see the text.
        assert c.post("/api/ask", json={"query": q + "!"}, headers=ALICE).json()["matched_query"] == q
        assert c.post("/api/lookup", json={"query": q}, headers=ADMIN).json()["matches"][0]["query"] == q


def test_rho_is_returned_everywhere(make_client):
    with make_client(None) as c:
        ask = c.post("/api/ask", json={"query": "my laptop is really slow"}, headers=ALICE).json()
        look = c.post("/api/lookup", json={"query": "my laptop is really slow"}).json()
        x = c.post("/api/explain", json={"query": "my laptop is really slow"}).json()
        assert 0 < ask["rho"] <= 1 and ask["rho"] == look["rho"] == x["rho"]
        assert "ρ =" in ask["note"] and "ρ =" in look["note"]
        assert {"rho", "pair_rho", "scored_by"} <= set(look["matches"][0])
        assert {"rho", "pair_rho", "scored_by", "rho_blocked"} <= set(x["matches"][0])


# --- profiles -------------------------------------------------------------------

def test_profiles_override_settings_and_separate_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path))
    monkeypatch.delenv("SEMCACHE_PROFILE", raising=False)
    faq = Settings.load(PROJECT_ROOT / "config.toml")
    quora = Settings.load(PROJECT_ROOT / "config.toml", profile="quora")
    assert faq.profile == "faq" and set(faq.profiles) >= {"faq", "quora"}
    assert faq.corpus.use_quora is False and quora.corpus.use_quora is True
    assert quora.cache.lsa_weight == 0.5 and faq.cache.lsa_weight == 1.0
    assert faq.model_dir == tmp_path / "faq" / "model" and quora.cache_dir == tmp_path / "quora" / "cache"
    assert faq.users_dir == quora.users_dir == tmp_path / "users"           # users are shared
    monkeypatch.setenv("SEMCACHE_PROFILE", "quora")
    assert Settings.load(PROJECT_ROOT / "config.toml").profile == "quora"
    with pytest.raises(ValueError, match="unknown profile"):
        Settings.load(PROJECT_ROOT / "config.toml", profile="nope")


def test_quora_profile_refuses_to_build_without_the_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path))
    settings = Settings.load(PROJECT_ROOT / "config.toml", profile="quora")
    settings.paths.quora_train = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="quora"):
        build_artifacts(settings, verbose=False)


def test_health_reports_the_profile(make_client):
    with make_client(None) as c:
        body = c.get("/api/health").json()
        assert body["profile"] == "faq" and "quora" in body["profiles"]


def test_backend_can_be_switched_to_the_stub_without_editing_config(monkeypatch):
    monkeypatch.setenv("SEMCACHE_LLM_BACKEND", "stub")
    assert Settings.load(PROJECT_ROOT / "config.toml").llm.backend == "stub"
    monkeypatch.setenv("SEMCACHE_LLM_BACKEND", "gpt")
    with pytest.raises(ValueError, match="SEMCACHE_LLM_BACKEND"):
        Settings.load(PROJECT_ROOT / "config.toml")
