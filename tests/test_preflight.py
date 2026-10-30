import threading
from http.server import HTTPServer

import pytest

from semcache.pipeline import SemanticCacheService
from semcache.preflight import run_preflight
from tests.test_backends import FakeOllama


@pytest.fixture()
def fake_ollama():
    FakeOllama.loaded = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def run(settings):
    lines: list[str] = []
    ok = run_preflight(settings, lines.append)
    return ok, lines


def test_all_checks_pass_with_ollama_and_a_fresh_cache(settings, fake_ollama):
    settings.llm.backend, settings.llm.ollama_url = "ollama", fake_ollama
    SemanticCacheService.from_settings(settings, verbose=False).reset_demo(admin=True)
    ok, lines = run(settings)
    assert ok, "\n".join(lines)
    assert any("warmed up now" in line for line in lines)
    assert sum(line.startswith("PASS  preset") for line in lines) == 6
    ok, lines = run(settings)                                   # second time: already warm
    assert ok and any("loaded in memory (warm)" in line for line in lines)


def test_stub_backend_and_a_used_cache_fail(settings):
    settings.llm.backend = "stub"
    service = SemanticCacheService.from_settings(settings, verbose=False)
    service.ask("what is the best laptop for students")         # a rehearsal left this behind
    ok, lines = run(settings)
    assert not ok
    assert any(line.startswith("FAIL  LLM backend is the stub") for line in lines)
    assert any(line.startswith("FAIL  cache has 44 entries") and "reset-demo" in line for line in lines)
    assert any(line.startswith("FAIL  preset 3") for line in lines)     # it hits its own answer now


def test_preflight_stores_and_counts_nothing(settings):
    settings.llm.backend = "stub"
    service = SemanticCacheService.from_settings(settings, verbose=False)
    service.reset_demo(admin=True)
    before = (len(service.cache), service.cache.hits, service.cache.misses)
    run_preflight(settings, lambda _line: None, service=service)
    assert (len(service.cache), service.cache.hits, service.cache.misses) == before
    assert service.metrics_report()["summary"]["requests"] == 0
    assert not (settings.metrics_dir / "events.jsonl").exists()
