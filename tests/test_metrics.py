"""Metrics (counters, latency, estimated cost, time series) and force refresh."""

import json

import pytest

from semcache.backends import Generation, StubBackend
from semcache.metrics import MetricsStore, Pricing, parse_bucket
from semcache.pipeline import SemanticCacheService


class FakeOllama(StubBackend):
    """Answers like Ollama would: cacheable, with token counts, ~50 ms."""
    name = "ollama"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, prompt, history=None):
        self.calls += 1
        return Generation(f"answer {self.calls} to {prompt}", "ollama", "fake:1b", 50.0,
                          prompt_tokens=40, output_tokens=60)


class DownOllama(StubBackend):
    """Ollama is down: the stub answers as a fallback."""
    def generate(self, prompt, history=None):
        g = super().generate(prompt, history)
        g.fallback, g.cacheable, g.error = True, False, "connection refused"
        return g


@pytest.fixture()
def service(settings):
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    svc.backend = FakeOllama()
    return svc


# --- the store ------------------------------------------------------------------------

def test_parse_bucket():
    assert parse_bucket("10s") == 10 and parse_bucket("2m") == 120 and parse_bucket("1h") == 3600
    assert parse_bucket("30") == 30
    for bad in ("0s", "10x", "2d", "-5"):
        with pytest.raises(ValueError):
            parse_bucket(bad)


def test_pricing_label_and_cost():
    assert not Pricing().configured and Pricing().label() is None
    p = Pricing(per_1k_input=0.5, per_1k_output=1.5, note="assumed: some list price")
    assert p.cost(1000, 2000) == pytest.approx(3.5)
    assert p.label() == "estimated at $0.50 per 1K input tokens and $1.50 per 1K output tokens"
    assert Pricing(per_call=0.002).cost(None, None) == pytest.approx(0.002)
    assert Pricing(per_1k_input=0.5).cost(None, None) is None              # nothing to price with


def test_series_buckets_and_cumulative_hit_rate(tmp_path):
    store = MetricsStore(tmp_path)
    t0 = 1_000_000.0
    for dt, kind in [(0, "miss"), (3, "hit"), (12, "hit"), (14, "follow_up"), (25, "miss")]:
        store._events.append({"t": t0 + dt, "user": None, "kind": kind})
    s = store.series(10, now=t0 + 29)
    assert [p["requests"] for p in s["points"]] == [2, 2, 1]
    assert [p["total_requests"] for p in s["points"]] == [2, 4, 5]
    assert [round(p["hit_rate"], 3) for p in s["points"]] == [0.5, 0.667, 0.5]   # follow-ups don't count
    short = store.series(10, now=t0 + 29, max_buckets=2)
    assert short["truncated"] and short["points"][0]["total_requests"] == 4      # earlier events still counted


def test_cost_and_savings_are_estimates(tmp_path):
    store = MetricsStore(tmp_path)
    store._events += [
        {"t": 1, "kind": "miss", "llm": True, "fallback": False, "llm_ms": 100, "total_ms": 101,
         "prompt_tokens": 1000, "output_tokens": 1000},
        {"t": 2, "kind": "hit", "llm": False, "llm_ms": 0, "total_ms": 1, "entry_tokens": [2000, 0]},
        {"t": 3, "kind": "hit", "llm": False, "llm_ms": 0, "total_ms": 1, "entry_tokens": None},
    ]
    p = Pricing(per_1k_input=1.0, per_1k_output=2.0, note="assumed")
    s = store.summary(p)
    assert s["cost"]["estimated"] and s["cost"]["llm_cost_usd"] == pytest.approx(3.0)
    assert s["cost"]["savings_usd"] == pytest.approx(2.0 + 3.0)       # from the entry, then the average
    assert (s["cost"]["hits_priced_from_entry"], s["cost"]["hits_priced_from_average"]) == (1, 1)
    assert s["time_saved_ms"] == pytest.approx(2 * (100 - 1))         # hits x (LLM - hit latency)
    assert store.summary(Pricing())["cost"] is None                   # no price set: no cost at all


def test_no_data_means_none_not_zero(tmp_path):
    store = MetricsStore(tmp_path)
    p = Pricing(per_1k_input=1.0)
    empty = store.summary(p)
    assert empty["time_saved_ms"] is None and empty["latency_ms"]["avg_llm"] is None
    assert empty["hit_rate"] is None and empty["hit_rate_of_requests"] is None
    assert empty["cost"]["llm_cost_usd"] is None and empty["cost"]["savings_usd"] is None
    store._events.append({"t": 1, "kind": "hit", "llm": False, "llm_ms": 0, "total_ms": 1})
    store._events.append({"t": 2, "kind": "miss", "llm": True, "fallback": True, "llm_ms": 5, "total_ms": 6})
    s = store.summary(p)
    assert s["time_saved_ms"] is None                 # only a fallback answered: nothing to compare
    assert s["hit_rate"] == 0.5 and s["hit_rate_of_requests"] == 0.5
    assert s["hit_rate_label"] == "Hit rate (of cache lookups)"


# --- through the service ------------------------------------------------------------------

def test_counters_latency_and_restart(service, settings):
    service.ask("how do I reset my password if I forgot it?", user="alice")      # FAQ: hit
    first = service.ask("how many moons does jupiter have", user="alice")        # miss -> LLM
    tid = first.thread_id
    service.ask("and saturn?", user="alice", thread_id=tid)                      # follow-up
    service.ask("how many moons does jupiter have", user="bob", use_cache=False)  # forced refresh
    service.backend = DownOllama()
    service.ask("what is the tallest mountain on mars", user="bob")              # miss, fallback

    s = service.metrics_report()["summary"]
    assert (s["requests"], s["hits"], s["misses"]) == (5, 1, 2)
    assert (s["llm_calls"], s["llm_fallbacks"], s["follow_ups"], s["forced_refreshes"]) == (4, 1, 1, 1)
    assert s["hit_rate"] == pytest.approx(1 / 3) and s["hit_rate_of_requests"] == pytest.approx(1 / 5)
    assert s["latency_ms"]["avg_llm"] == pytest.approx(50.0) and s["latency_ms"]["llm_samples"] == 3
    assert s["time_saved_ms"] == pytest.approx(50.0 - s["latency_ms"]["avg_hit"])
    assert s["tokens"]["input"] == 120 and s["cost"] is None
    alice = service.metrics_report(user="alice")["summary"]
    assert (alice["requests"], alice["hits"], alice["follow_ups"]) == (3, 1, 1)

    lines = (settings.metrics_dir / "events.jsonl").read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["source"] == "FALLBACK"
    again = SemanticCacheService.from_settings(settings, verbose=False)            # after a restart
    assert again.metrics_report()["summary"]["requests"] == 5


def test_every_answer_reports_source_similarity_rho_and_timings(service):
    hit = service.ask("i forgot my password, how do i reset it")
    miss = service.ask("which matrix should i buy")                 # known words, no close answer
    for r in (hit, miss):
        assert r.similarity_pct == pytest.approx(100 * r.score, abs=0.05)
        assert 0 < r.rho <= 1 and r.timings_ms["total"] > 0
    assert hit.answer_source == "CACHE" and miss.answer_source == "LLM" and not miss.hit
    assert miss.prompt_tokens == 40 and service.cache.get(miss.stored_id).output_tokens == 60
    unknown = service.ask("how many moons does jupiter have")       # unknown words: nothing to compare
    assert unknown.similarity_pct is None and unknown.matches == []


def test_hit_on_an_llm_answer_carries_its_token_cost(service):
    service.settings.metrics.price_per_1k_input_tokens = 1.0
    service.settings.metrics.price_per_1k_output_tokens = 1.0
    service.ask("how many moons does jupiter have")
    assert service.ask("how many moons does jupiter have?").hit
    cost = service.metrics_report()["summary"]["cost"]
    assert cost["hits_priced_from_entry"] == 1 and cost["savings_usd"] == pytest.approx(0.1)


def test_reset_metrics_needs_admin_and_keeps_the_old_file(service, settings):
    service.ask("how many moons does jupiter have")
    with pytest.raises(PermissionError):
        service.reset_metrics()
    out = service.reset_metrics(admin=True)
    assert out["removed_events"] == 1 and out["archived_to"].endswith(".jsonl")
    assert service.metrics_report()["summary"]["requests"] == 0
    assert not (settings.metrics_dir / "events.jsonl").exists()


# --- force refresh --------------------------------------------------------------------------

def test_refresh_stores_a_new_answer_when_none_exists(service):
    before = len(service.cache)
    r = service.ask("how many moons does jupiter have", use_cache=False)
    assert r.refreshed and not r.hit and r.answer_source == "LLM" and r.stored_id
    assert len(service.cache) == before + 1 and "Stored as a new" in r.note


def test_refresh_replaces_an_llm_answer(service):
    old = service.ask("how many moons does jupiter have")
    r = service.ask("how many moons does jupiter have", use_cache=False)
    assert r.stored_id != old.stored_id and service.cache.get(old.stored_id) is None
    assert service.cache.get(r.stored_id).answer == r.answer and "Replaced" in r.note
    assert service.ask("how many moons does jupiter have").answer == r.answer


def test_refresh_never_overwrites_faq_or_admin_answers(service):
    faq = next(e for e in service.cache.entries() if e.source == "faq")
    r = service.ask(faq.query, use_cache=False)
    assert r.answer_source == "LLM" and r.stored_id is None and "FAQ" in r.note
    assert service.cache.get(faq.id).answer == faq.answer
    added = service.add_entry("where is the exam hall", "Room 204", user="carol", shared=True, admin=True)
    r = service.ask("where is the exam hall", use_cache=False)
    assert r.stored_id is None and "admin-added" in r.note
    assert service.cache.get(added.id).answer == "Room 204"


def test_refresh_of_a_follow_up_keeps_the_conversation(service):
    tid = service.ask("how many moons does jupiter have", user="alice").thread_id
    before = len(service.cache)
    r = service.ask("and saturn?", user="alice", thread_id=tid, use_cache=False)
    assert r.follow_up and not r.refreshed and r.history_used == 2 and r.answer_source == "FOLLOW-UP"
    assert r.stored_id is None and len(service.cache) == before


def test_fallback_answer_on_refresh_is_not_stored(service):
    service.backend = DownOllama()
    r = service.ask("how many moons does jupiter have", use_cache=False)
    assert r.answer_source == "FALLBACK" and r.stored_id is None and "placeholder" in r.note


# --- API ---------------------------------------------------------------------------------------

def test_metrics_endpoint(settings, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from apps.api import create_app

    monkeypatch.setenv("SEMCACHE_ADMIN_KEY", "k")
    with TestClient(create_app(settings)) as c:
        c.post("/api/ask", json={"query": "i forgot my password, how do i reset it"}, headers={"X-User": "alice"})
        body = c.get("/api/metrics?bucket=10s").json()
        assert body["summary"]["requests"] == 1 and body["series"]["bucket_seconds"] == 10
        assert body["pricing"]["configured"] is False
        assert c.get("/api/metrics?scope=user", headers={"X-User": "alice"}).json()["summary"]["hits"] == 1
        assert c.get("/api/metrics?scope=user").status_code == 401
        assert c.get("/api/metrics?bucket=banana").status_code == 422
        assert c.post("/api/metrics/reset", headers={"X-User": "alice"}).status_code == 403
        assert c.post("/api/metrics/reset", headers={"X-User": "alice", "X-Admin-Key": "k"}).status_code == 200


def test_explain_and_lookup_never_count(service):
    for _ in range(3):
        service.explain("cheap flight tickets to goa", user="alice")
        service.lookup("cheap flight tickets to goa", user="alice")
    s = service.metrics_report()["summary"]
    assert s["requests"] == 0 and not (service.settings.metrics_dir / "events.jsonl").exists()
    assert (service.cache.hits, service.cache.misses) == (0, 0)
    assert service.users.profile("alice").stats()["questions"] == 0
    service.ask("cheap flight tickets to goa", user="alice")
    assert service.metrics_report()["summary"]["requests"] == 1


def test_explain_and_lookup_never_count_over_http(settings):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from apps.api import create_app

    with TestClient(create_app(settings)) as c:
        h = {"X-User": "alice"}
        for _ in range(3):
            c.post("/api/explain", json={"query": "cheap flight tickets to goa"}, headers=h)
            c.post("/api/lookup", json={"query": "cheap flight tickets to goa"}, headers=h)
        assert c.get("/api/metrics").json()["summary"]["requests"] == 0
        assert c.get("/api/stats").json()["hits"] == 0


def test_reset_demo(service, settings):
    service.ask("how many moons does jupiter have", user="alice")
    service.add_entry("where is the exam hall", "Room 204", user="bob")
    with pytest.raises(PermissionError):
        service.reset_demo()
    out = service.reset_demo(admin=True)
    assert out["entries"] == out["faq_entries"] == 43 and out["removed_events"] == 1
    assert all(e.source == "faq" for e in service.cache.entries())
    assert service.users.profile("alice").stats()["questions"] == 0
    assert service.metrics_report()["summary"]["requests"] == 0
    assert service.users.list_threads("alice")                      # conversations are kept


def test_reset_demo_cli(settings, capsys):
    from semcache.cli import main
    from semcache.pipeline import build_artifacts

    build_artifacts(settings, verbose=False)
    assert main(["reset-demo", "--profile", "faq"]) == 0
    assert "demo reset: 43 FAQ answers" in capsys.readouterr().out
