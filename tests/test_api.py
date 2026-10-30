import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from apps.api import create_app  # noqa: E402


ADMIN_KEY = "test-passphrase"


@pytest.fixture()
def client(settings, monkeypatch):
    monkeypatch.setenv("SEMCACHE_ADMIN_KEY", ADMIN_KEY)
    with TestClient(create_app(settings)) as c:     # runs startup: builds the model on first use
        yield c


def test_health_reports_model_and_backend(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model"]["rank"] > 0
    assert body["backend"]["backend"] == "stub"
    assert body["cache"]["entries"] > 0


def test_ask_miss_then_hit(client):
    q = "how many moons does jupiter have"
    first = client.post("/api/ask", json={"query": q}).json()
    assert first["hit"] is False and first["source"] == "stub"
    second = client.post("/api/ask", json={"query": q + "?"}).json()
    assert second["hit"] is True and second["lookup_method"] == "exact"


def test_ask_faq_semantic_hit(client):
    body = client.post("/api/ask", json={"query": "i forgot my password, how do i reset it"}).json()
    assert body["hit"] and body["matched_source"] == "faq"
    assert body["matches"][0]["score"] >= body["threshold"]


def test_lookup_does_not_store(client):
    before = client.get("/api/stats").json()["entries"]
    body = client.post("/api/lookup", json={"query": "what is overfitting", "threshold": 0.5}).json()
    assert body["matches"] and "known_terms" in body
    assert client.get("/api/stats").json()["entries"] == before


def test_explain_endpoint(client):
    before = client.get("/api/stats").json()["entries"]
    body = client.post("/api/explain", json={"query": "how can i reset my password", "threshold": 0.6}).json()
    assert body["threshold"] == 0.6 and body["tokens"] and body["terms"]
    assert len(body["latent"]) == client.get("/api/health").json()["model"]["rank"]
    assert body["matches"] and "reasons" in body["matches"][0]
    assert client.get("/api/stats").json()["entries"] == before
    assert client.post("/api/explain", json={"query": ""}).status_code == 422


ADMIN = {"X-User": "admin", "X-Admin-Key": ADMIN_KEY}
ALICE = {"X-User": "alice"}
BOB = {"X-User": "bob"}


def test_entries_add_delete_and_reset(client):
    created = client.post("/api/entries", json={"question": "where is the library", "answer": "Block C",
                                                "shared": True}, headers=ADMIN)
    assert created.status_code == 201 and created.json()["owner"] is None
    entry_id = created.json()["id"]
    assert any(e["id"] == entry_id for e in client.get("/api/entries", headers=ADMIN).json())
    assert any(e["id"] == entry_id for e in client.get("/api/entries").json())     # admin-added: public
    assert client.delete(f"/api/entries/{entry_id}", headers=ADMIN).status_code == 200
    assert client.delete(f"/api/entries/{entry_id}", headers=ADMIN).status_code == 404
    body = client.post("/api/reset", headers=ADMIN).json()
    assert body["scope"] == "all"
    assert client.get("/api/stats").json()["entries"] == body["entries"]


def test_changes_need_a_user(client):
    assert client.post("/api/entries", json={"question": "q here", "answer": "a"}).status_code == 401
    assert client.delete("/api/entries/1").status_code == 401
    assert client.post("/api/reset").status_code == 401
    assert client.get("/api/threads").status_code == 401
    assert client.get("/api/me", headers={"X-User": "../etc"}).status_code == 400


def test_sign_in_and_profile(client):
    body = client.post("/api/session", json={"name": "  Alice "}).json()
    assert body["name"] == "alice" and body["is_admin"] is False
    assert client.get("/api/me", headers=ADMIN).json()["is_admin"] is True
    assert client.get("/api/me", headers={"X-User": "admin"}).json()["is_admin"] is False   # a name isn't enough
    assert client.post("/api/session", json={"name": "a/b"}).status_code == 422


def test_conversation_with_follow_up(client):
    first = client.post("/api/ask", json={"query": "how many moons does jupiter have"}, headers=ALICE).json()
    tid = first["thread_id"]
    assert tid and first["cache_used"] and not first["follow_up"]

    preview = client.post("/api/lookup", json={"query": "and saturn?", "thread_id": tid}, headers=ALICE).json()
    assert preview["follow_up"] and not preview["cache_used"] and not preview["hit"]

    follow = client.post("/api/ask", json={"query": "and saturn?", "thread_id": tid}, headers=ALICE).json()
    assert follow["follow_up"] and follow["history_used"] == 2 and follow["stored_id"] is None

    threads = client.get("/api/threads", headers=ALICE).json()
    assert [t["id"] for t in threads] == [tid] and threads[0]["message_count"] == 4
    detail = client.get(f"/api/threads/{tid}", headers=ALICE).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][3]["meta"]["follow_up"] is True

    # Bob gets alice's standalone answer from the shared cache, but not her conversation.
    bob = client.post("/api/ask", json={"query": "how many moons does jupiter have?"}, headers=BOB).json()
    assert bob["hit"] and bob["thread_id"] != tid
    assert client.get(f"/api/threads/{tid}", headers=BOB).status_code == 404
    assert client.post("/api/ask", json={"query": "and saturn?", "thread_id": tid}, headers=BOB).status_code == 404
    assert client.post("/api/ask", json={"query": "and saturn?", "thread_id": tid}).status_code == 401
    assert client.get("/api/stats", headers=ALICE).json()["user"]["follow_ups"] == 1


def test_thread_create_rename_delete(client):
    created = client.post("/api/threads", json={"title": "Trip"}, headers=ALICE)
    assert created.status_code == 201 and created.json()["messages"] == []
    tid = created.json()["id"]
    assert client.patch(f"/api/threads/{tid}", json={"title": "Goa trip"}, headers=ALICE).json()["title"] == "Goa trip"
    assert client.delete(f"/api/threads/{tid}", headers=BOB).status_code == 404
    assert client.delete(f"/api/threads/{tid}", headers=ALICE).status_code == 200
    assert client.get(f"/api/threads/{tid}", headers=ALICE).status_code == 404


def test_private_entries_and_permissions_over_http(client):
    mine = client.post("/api/entries", json={"question": "where is the exam hall", "answer": "Room 204"},
                       headers=ALICE).json()
    assert mine["owner"] == "alice"
    assert any(e["id"] == mine["id"] for e in client.get("/api/entries", headers=ALICE).json())
    assert all(e["id"] != mine["id"] for e in client.get("/api/entries", headers=BOB).json())
    assert all(e["id"] != mine["id"] for e in client.get("/api/entries").json())
    assert client.delete(f"/api/entries/{mine['id']}", headers=BOB).status_code == 404
    faq = next(e for e in client.get("/api/entries", headers=BOB).json() if e["source"] == "faq")
    assert client.delete(f"/api/entries/{faq['id']}", headers=BOB).status_code == 403
    assert client.post("/api/entries", json={"question": "q here", "answer": "a", "shared": True},
                       headers=BOB).status_code == 403
    body = client.post("/api/reset", headers=ALICE).json()
    assert body["scope"] == "private" and body["removed"] == 1


def test_validation_errors(client):
    assert client.post("/api/ask", json={"query": ""}).status_code == 422
    assert client.post("/api/ask", json={"query": "hi", "threshold": 1.5}).status_code == 422


def test_model_endpoint_exposes_spectrum_and_directions(client):
    body = client.get("/api/model").json()
    assert len(body["singular_values"]) == body["rank"]
    assert body["directions"] and body["directions"][0]["terms"]
