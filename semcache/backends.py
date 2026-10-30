"""Who answers a query when the cache misses.

* OllamaBackend   a local LLM served by Ollama (http://localhost:11434)
* StubBackend     a placeholder answer, so the project runs without any LLM
* FallbackBackend tries the first backend, uses the second if it fails

Every backend has ``generate(prompt, history=None)``. ``history`` is a list of
earlier ``{"role": "user" | "assistant", "content": ...}`` messages of the
conversation; it is only passed for follow-up questions.

Only the standard library is used for HTTP, so there is no extra dependency.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    backend: str                 # "ollama" or "stub"
    model: str | None
    latency_ms: float
    cacheable: bool = True       # False: never stored in the shared cache
    error: str | None = None     # why the primary backend failed, when a fallback answered
    fallback: bool = False       # the fallback answered because the primary failed
    prompt_tokens: int | None = None   # Ollama's prompt_eval_count (None: not reported)
    output_tokens: int | None = None   # Ollama's eval_count


class BackendError(RuntimeError):
    pass


class StubBackend:
    """Returns a clearly labelled placeholder instead of calling a model."""

    name = "stub"

    def __init__(self, delay_ms: float = 0.0, cacheable: bool = True):
        self.delay_ms = delay_ms
        self.cacheable = cacheable

    def available(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"backend": "stub", "available": True, "model": None,
                "simulated_delay_ms": self.delay_ms}

    def warm_up(self) -> None:
        """Nothing to load."""

    def generate(self, prompt: str, history: list[dict] | None = None) -> Generation:
        t0 = time.perf_counter()
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        text = ("[Placeholder answer - no LLM connected] Start Ollama to get real answers. "
                f"Your question was: \"{prompt.strip()}\"")
        if history:
            text += f" (a follow-up; {len(history)} earlier messages would be sent along)"
        return Generation(text, "stub", None, (time.perf_counter() - t0) * 1000.0, self.cacheable)


class OllamaBackend:
    """Calls Ollama's REST API. ``model="auto"`` uses the first model you have pulled."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "auto",
                 temperature: float = 0.2, timeout_s: float = 120.0,
                 system_prompt: str = "Answer clearly and concisely in 2-4 sentences."):
        self.base_url = base_url.rstrip("/")
        self.requested_model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        self._resolved_model: str | None = None if model == "auto" else model

    # -- HTTP helpers ------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise BackendError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError(f"cannot reach Ollama at {self.base_url} ({exc})") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError as exc:          # JSONDecodeError and UnicodeDecodeError
            raise BackendError(f"Ollama sent a response that is not JSON: {body[:200]!r}") from exc

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._request("/api/tags", timeout=2.0).get("models", [])]

    # -- backend interface -------------------------------------------------
    def available(self) -> bool:
        try:
            return bool(self.list_models())
        except BackendError:
            return False

    @property
    def model(self) -> str:
        if self._resolved_model is None:
            models = self.list_models()
            if not models:
                raise BackendError("Ollama is running but has no models; run `ollama pull <model>`")
            self._resolved_model = models[0]
        return self._resolved_model

    def describe(self) -> dict:
        try:
            models = self.list_models()
            if self._resolved_model:
                model = self._resolved_model
            elif self.requested_model == "auto":
                model = models[0] if models else None
            else:
                model = self.requested_model
            return {"backend": "ollama", "available": bool(models), "url": self.base_url,
                    "model": model, "installed_models": models}
        except BackendError as exc:
            return {"backend": "ollama", "available": False, "url": self.base_url,
                    "model": self.requested_model, "error": str(exc)}

    def loaded_models(self) -> list[str]:
        """Models Ollama currently holds in memory (GET /api/ps)."""
        return [m.get("name") or m.get("model") for m in self._request("/api/ps", timeout=2.0).get("models", [])]

    def warm_up(self) -> None:
        """Load the model into memory, so the first real question isn't slowed by it.
        Ollama loads a model when it gets an empty prompt, without generating anything."""
        self._request("/api/generate", {"model": self.model, "prompt": "", "keep_alive": "30m"})

    def generate(self, prompt: str, history: list[dict] | None = None) -> Generation:
        t0 = time.perf_counter()
        options = {"temperature": self.temperature}
        if history:
            # /api/chat takes the whole conversation, so the model sees what "it" refers to.
            messages = [{"role": "system", "content": self.system_prompt},
                        *({"role": m["role"], "content": m["content"]} for m in history),
                        {"role": "user", "content": prompt}]
            out = self._request("/api/chat", {"model": self.model, "messages": messages,
                                              "stream": False, "options": options})
            text = ((out.get("message") or {}).get("content") or "").strip()
        else:
            out = self._request("/api/generate", {"model": self.model, "prompt": prompt,
                                                  "system": self.system_prompt, "stream": False,
                                                  "options": options})
            text = (out.get("response") or "").strip()
        if not text:
            raise BackendError("Ollama returned an empty response")
        return Generation(text, "ollama", self.model, (time.perf_counter() - t0) * 1000.0,
                          prompt_tokens=_count(out.get("prompt_eval_count")),
                          output_tokens=_count(out.get("eval_count")))


def _count(value) -> int | None:
    """A token count from Ollama's response, or None when it is missing or not a number."""
    return int(value) if isinstance(value, (int, float)) and value >= 0 else None


class FallbackBackend:
    """Use ``primary``; if it is down or errors, answer with ``fallback`` instead."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+{fallback.name}"
        self.last_error: str | None = None

    def available(self) -> bool:
        return self.primary.available() or self.fallback.available()

    def warm_up(self) -> None:
        self.primary.warm_up()

    def describe(self) -> dict:
        info = self.primary.describe()
        info["fallback"] = self.fallback.name
        return info

    def generate(self, prompt: str, history: list[dict] | None = None) -> Generation:
        try:
            result = self.primary.generate(prompt, history) if history else self.primary.generate(prompt)
            self.last_error = None
            return result
        except BackendError as exc:
            self.last_error = str(exc)
            result = self.fallback.generate(prompt, history) if history else self.fallback.generate(prompt)
            result.error = str(exc)     # per answer: last_error is shared by concurrent requests
            result.fallback = True
            result.cacheable = False    # a stand-in answer must never pollute the shared cache
            return result


def build_backend(llm_settings) -> object:
    """Create the backend described by the [llm] section of config.toml.

    Placeholder answers are cached only when the stub is the chosen backend (and
    ``cache_stub_answers`` is on); as a fallback for Ollama they never are.
    """
    explicit_stub = llm_settings.backend == "stub"
    stub = StubBackend(llm_settings.stub_delay_ms, explicit_stub and llm_settings.cache_stub_answers)
    if llm_settings.backend == "stub":
        return stub
    if llm_settings.backend != "ollama":
        raise ValueError(f"unknown llm backend {llm_settings.backend!r} (use 'ollama' or 'stub')")
    ollama = OllamaBackend(llm_settings.ollama_url, llm_settings.model, llm_settings.temperature,
                           llm_settings.timeout_s, llm_settings.system_prompt)
    return FallbackBackend(ollama, stub) if llm_settings.fallback_to_stub else ollama
