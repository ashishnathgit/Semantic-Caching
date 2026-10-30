"""The Ollama client against a tiny fake Ollama server (no real model needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from semcache.backends import BackendError, FallbackBackend, OllamaBackend, StubBackend


class FakeOllama(BaseHTTPRequestHandler):
    models = ["llama3.2:3b", "qwen2.5:7b"]
    last_body: dict = {}

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    loaded: list = []

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": m} for m in self.models]})
        elif self.path == "/api/ps":
            self._send(200, {"models": [{"name": m} for m in FakeOllama.loaded]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.last_body = body
        if body.get("model") == "broken-json":
            data = b"<html>502 Bad Gateway</html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/generate" and body.get("model") in self.models and body.get("prompt") == "":
            FakeOllama.loaded.append(body["model"])       # an empty prompt loads the model
            self._send(200, {"model": body["model"], "response": "", "done": True})
        elif self.path == "/api/generate" and body.get("model") in self.models:
            self._send(200, {"model": body["model"], "response": f"echo: {body['prompt']}", "done": True,
                             "prompt_eval_count": 31, "eval_count": 12})
        elif self.path == "/api/chat" and body.get("model") in self.models:
            last = body["messages"][-1]["content"]
            self._send(200, {"model": body["model"], "done": True,
                             "message": {"role": "assistant", "content": f"chat echo: {last}"}})
        else:
            self._send(404, {"error": f"model '{body.get('model')}' not found"})


@pytest.fixture()
def fake_ollama():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_auto_model_uses_first_pulled_model(fake_ollama):
    backend = OllamaBackend(fake_ollama, model="auto")
    assert backend.available()
    gen = backend.generate("what is svd")
    assert gen.text == "echo: what is svd" and gen.model == "llama3.2:3b" and gen.backend == "ollama"
    body = FakeOllama.last_body
    assert body["stream"] is False and body["system"] and "temperature" in body["options"]


def test_named_model_and_missing_model_error(fake_ollama):
    assert OllamaBackend(fake_ollama, model="qwen2.5:7b").generate("hi").model == "qwen2.5:7b"
    with pytest.raises(BackendError, match="HTTP 404"):
        OllamaBackend(fake_ollama, model="does-not-exist").generate("hi")


def test_fallback_prefers_ollama_when_it_works(fake_ollama):
    backend = FallbackBackend(OllamaBackend(fake_ollama), StubBackend())
    assert backend.generate("hello").backend == "ollama"
    assert backend.last_error is None


def test_non_json_reply_is_a_backend_error_so_fallback_works(fake_ollama):
    backend = OllamaBackend(fake_ollama, model="broken-json")
    with pytest.raises(BackendError, match="not JSON"):
        backend.generate("hi")
    fallback = FallbackBackend(backend, StubBackend())
    assert fallback.generate("hi").backend == "stub"
    assert "not JSON" in fallback.last_error


def test_history_goes_to_the_chat_endpoint(fake_ollama):
    backend = OllamaBackend(fake_ollama, system_prompt="be brief")
    history = [{"role": "user", "content": "how many moons does jupiter have"},
               {"role": "assistant", "content": "95"}]
    gen = backend.generate("and saturn?", history)
    assert gen.text == "chat echo: and saturn?"
    msgs = FakeOllama.last_body["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "be brief" and msgs[1]["content"] == history[0]["content"]


def test_fallback_error_travels_with_each_answer():
    backend = FallbackBackend(OllamaBackend("http://127.0.0.1:9", timeout_s=1), StubBackend())
    gen = backend.generate("hello", [{"role": "user", "content": "hi"}])
    assert gen.backend == "stub" and gen.error and "cannot reach Ollama" in gen.error
    assert "follow-up" in gen.text


def test_placeholder_answers_are_cached_only_with_the_explicit_stub_backend():
    from semcache.backends import build_backend
    from semcache.config import LLMConfig

    assert build_backend(LLMConfig(backend="stub")).generate("q").cacheable is True
    assert build_backend(LLMConfig(backend="stub", cache_stub_answers=False)).generate("q").cacheable is False
    # Ollama is down (nothing listens on port 9): the stub answers as a fallback, uncached.
    fallback = build_backend(LLMConfig(backend="ollama", ollama_url="http://127.0.0.1:9", timeout_s=2))
    gen = fallback.generate("q")
    assert gen.backend == "stub" and gen.fallback and gen.error and gen.cacheable is False


def test_fallback_answer_never_reaches_the_shared_cache(settings):
    from semcache.pipeline import SemanticCacheService

    settings.llm.backend, settings.llm.ollama_url, settings.llm.timeout_s = "ollama", "http://127.0.0.1:9", 2
    service = SemanticCacheService.from_settings(settings, verbose=False)
    before = len(service.cache)
    r = service.ask("how many moons does jupiter have")
    assert not r.hit and r.source == "stub" and r.backend_error
    assert r.stored_id is None and len(service.cache) == before


def test_ollama_token_counts_are_read(fake_ollama):
    gen = OllamaBackend(fake_ollama, model="auto").generate("hello")
    assert (gen.prompt_tokens, gen.output_tokens) == (31, 12)
    chat = OllamaBackend(fake_ollama, model="auto").generate("hi", [{"role": "user", "content": "x"}])
    assert chat.prompt_tokens is None                      # the fake chat reply has no counts


def test_warm_up_loads_the_model_with_an_empty_prompt(fake_ollama):
    OllamaBackend(fake_ollama, model="auto").warm_up()
    assert FakeOllama.last_body["prompt"] == "" and FakeOllama.last_body["model"] == "llama3.2:3b"


def test_service_warms_up_in_the_background(settings, fake_ollama):
    from semcache.pipeline import SemanticCacheService

    settings.llm.backend, settings.llm.ollama_url = "ollama", fake_ollama
    service = SemanticCacheService.from_settings(settings, verbose=False)
    thread = service.start_warm_up()
    assert thread is not None
    thread.join(5)
    assert service.warmup["state"] == "ready" and service.warmup["ms"] is not None


def test_warm_up_reports_failure_and_stub_needs_none(settings):
    from semcache.pipeline import SemanticCacheService

    settings.llm.backend, settings.llm.ollama_url, settings.llm.timeout_s = "ollama", "http://127.0.0.1:9", 2
    service = SemanticCacheService.from_settings(settings, verbose=False)
    service.start_warm_up().join(5)
    assert service.warmup["state"] == "failed" and service.warmup["error"]
    settings.llm.backend = "stub"
    stub = SemanticCacheService.from_settings(settings, verbose=False)
    assert stub.start_warm_up() is None and stub.warmup["state"] == "not needed"
