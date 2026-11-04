"""Phase F: the OpenAI-compatible backend, the fallback chain and .env handling."""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from semcache.backends import (BackendError, ChainBackend, FallbackBackend, OllamaBackend,
                               OpenAICompatibleBackend, StubBackend, build_backend)
from semcache.config import LLMConfig, Settings, load_env_file
from semcache.pipeline import SemanticCacheService

KEY = "sk-test-SECRET-1234567890"


class FakeOpenAI(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            data = json.dumps({"error": {"message": f"invalid key {self.headers.get('Authorization')}"}}).encode()
            self.send_response(401)
        else:
            last = body["messages"][-1]["content"]
            data = json.dumps({"model": body["model"], "choices": [{"message": {"role": "assistant",
                               "content": f"online: {last}"}}],
                               "usage": {"prompt_tokens": 42, "completion_tokens": 17}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_openai():
    FakeOpenAI.seen = []
    server = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_generate_sends_the_key_only_in_the_header(fake_openai):
    b = OpenAICompatibleBackend(fake_openai, "tiny-model", KEY)
    g = b.generate("what is svd")
    assert g.text == "online: what is svd" and g.backend == "openai_compatible" and g.model == "tiny-model"
    assert (g.prompt_tokens, g.output_tokens) == (42, 17)
    sent = FakeOpenAI.seen[-1]
    assert sent["path"] == "/v1/chat/completions" and sent["auth"] == f"Bearer {KEY}"
    assert KEY not in json.dumps(sent["body"])
    assert KEY not in json.dumps(b.describe()) and b.describe()["api_key_set"] is True


def test_errors_are_redacted(fake_openai):
    b = OpenAICompatibleBackend(fake_openai, "tiny-model", KEY + "-wrong")
    with pytest.raises(BackendError) as exc:
        b.generate("hello")
    assert "401" in str(exc.value) and KEY not in str(exc.value)


def test_unconfigured_backend_says_what_is_missing():
    b = OpenAICompatibleBackend("", "", None)
    assert not b.available() and "API key" in b.describe()["error"]
    with pytest.raises(BackendError):
        b.generate("hi")


def test_chain_falls_through_in_order(fake_openai):
    down = OllamaBackend("http://127.0.0.1:9", timeout_s=1)
    chain = ChainBackend([OpenAICompatibleBackend("http://127.0.0.1:9/v1", "m", KEY, timeout_s=1), down,
                          StubBackend(cacheable=False)])
    g = chain.generate("q")
    assert g.backend == "stub" and g.fallback and not g.cacheable          # nobody real answered
    assert [t["backend"] for t in g.tried] == ["openai_compatible", "ollama"]
    assert KEY not in g.error
    ok = ChainBackend([OpenAICompatibleBackend("http://127.0.0.1:9/v1", "m", KEY, timeout_s=1),
                       OpenAICompatibleBackend(fake_openai, "second", KEY), StubBackend()])
    g = ok.generate("q")
    assert g.backend == "openai_compatible" and g.model == "second" and g.cacheable and not g.fallback


def test_build_backend_keeps_ollama_the_default(monkeypatch):
    b = build_backend(LLMConfig())
    assert isinstance(b, FallbackBackend) and isinstance(b.primary, OllamaBackend)
    monkeypatch.setenv("SEMCACHE_OPENAI_API_KEY", KEY)
    c = build_backend(LLMConfig(chain=["openai_compatible", "ollama", "stub"], openai_base_url="http://x/v1",
                                openai_model="m"))
    assert isinstance(c, ChainBackend) and [x.name for x in c.backends] == ["openai_compatible", "ollama", "stub"]
    assert not c.backends[2].cacheable                  # the stub at the end only stands in
    with pytest.raises(ValueError):
        build_backend(LLMConfig(backend="nope"))


def test_env_file_is_read_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nSEMCACHE_TEST_A="quoted value"\nexport SEMCACHE_TEST_B=plain\nSEMCACHE_TEST_C=x\n')
    monkeypatch.setenv("SEMCACHE_TEST_C", "already set")
    monkeypatch.delenv("SEMCACHE_TEST_A", raising=False)
    monkeypatch.delenv("SEMCACHE_TEST_B", raising=False)
    assert sorted(load_env_file(env)) == ["SEMCACHE_TEST_A", "SEMCACHE_TEST_B"]
    import os
    assert os.environ["SEMCACHE_TEST_A"] == "quoted value" and os.environ["SEMCACHE_TEST_C"] == "already set"


def test_online_answers_feed_tokens_metrics_and_never_leak_the_key(settings, fake_openai, monkeypatch, caplog):
    monkeypatch.setenv("SEMCACHE_OPENAI_API_KEY", KEY)
    monkeypatch.setenv("SEMCACHE_OPENAI_BASE_URL", fake_openai)
    monkeypatch.setenv("SEMCACHE_OPENAI_MODEL", "tiny-model")
    settings.llm.backend = "ollama"
    settings.llm.chain = ["openai_compatible", "ollama", "stub"]
    settings.metrics.price_per_1k_input_tokens = 0.001
    settings.metrics.price_per_1k_output_tokens = 0.002
    from apps.api import create_app
    caplog.set_level(logging.DEBUG)
    with TestClient(create_app(settings)) as c:
        c.post("/api/session", json={"name": "ola"})
        h = {"X-User": "ola"}
        r = c.post("/api/ask", json={"query": "what is the best laptop for students"}, headers=h).json()
        assert r["answer_source"] == "LLM" and r["backend"] == "openai_compatible" and r["model"] == "tiny-model"
        assert (r["prompt_tokens"], r["output_tokens"]) == (42, 17) and r["stored_id"] is not None
        m = c.get("/api/metrics").json()["summary"]
        assert m["by_backend"] == {"openai_compatible": 1} and m["tokens"]["input"] == 42
        assert m["cost"]["llm_cost_usd"] == pytest.approx(42 / 1000 * 0.001 + 17 / 1000 * 0.002)
        health = c.get("/api/health").text
        for text in (health, json.dumps(r), c.get("/api/metrics").text, caplog.text):
            assert KEY not in text
    svc_events = (settings.metrics_dir / "events.jsonl").read_text()
    assert KEY not in svc_events and KEY not in (settings.cache_dir / "entries.json").read_text()


def test_backend_env_override_accepts_the_online_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMCACHE_LLM_BACKEND", "openai_compatible")
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path))
    s = Settings.load(profile="faq")
    assert s.llm.backend == "openai_compatible" and s.llm.chain == []
