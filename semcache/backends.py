"""Who answers a query when the cache misses.

* OllamaBackend           a local LLM served by Ollama (http://localhost:11434)
* OpenAICompatibleBackend an online model behind an OpenAI-style /chat/completions API
                          (base URL, model and API key from .env; the key never leaves here)
* StubBackend             a placeholder answer, so the project runs without any LLM
* FallbackBackend         tries the first backend, uses the second if it fails
* ChainBackend            [llm] chain: tries each backend in order, e.g. online -> Ollama -> stub

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
    prompt_tokens: int | None = None   # Ollama's prompt_eval_count / the API's usage (None: not reported)
    output_tokens: int | None = None   # Ollama's eval_count / the API's completion_tokens
    tried: list | None = None          # ChainBackend: [{"backend", "error"}] for each backend that failed first


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


class OpenAICompatibleBackend:
    """An online model behind an OpenAI-compatible API (OpenAI, Groq, Together, OpenRouter,
    a local vLLM ...): POST {base_url}/chat/completions with a Bearer key.

    The API key is read from the environment (.env) and is only ever put in the
    Authorization header: describe() never includes it, and error messages are redacted.
    Online mode sends every question that misses the cache to the provider.
    """

    name = "openai_compatible"

    def __init__(self, base_url: str, model: str, api_key: str | None, temperature: float = 0.2,
                 timeout_s: float = 60.0, system_prompt: str = "Answer clearly and concisely in 2-4 sentences.",
                 max_tokens: int | None = None):
        self.base_url = (base_url or "").rstrip("/")
        self.requested_model = model or ""
        self._key = api_key or ""
        self.temperature, self.timeout_s, self.system_prompt = temperature, timeout_s, system_prompt
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self.requested_model

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def missing(self) -> list[str]:
        return [n for n, v in (("base URL", self.base_url), ("model", self.requested_model),
                               ("API key", self._key)) if not v]

    def available(self) -> bool:
        return not self.missing()

    def describe(self) -> dict:
        # Never the key: only whether one is set.
        info = {"backend": self.name, "available": self.available(), "url": self.base_url,
                "model": self.requested_model or None, "api_key_set": bool(self._key)}
        if self.missing():
            info["error"] = "not configured: missing " + ", ".join(self.missing()) + " (see .env.example)"
        return info

    def warm_up(self) -> None:
        """Nothing to load locally."""

    def generate(self, prompt: str, history: list[dict] | None = None) -> Generation:
        if self.missing():
            raise BackendError("online model not configured: missing " + ", ".join(self.missing()))
        t0 = time.perf_counter()
        messages = [{"role": "system", "content": self.system_prompt},
                    *({"role": m["role"], "content": m["content"]} for m in (history or [])),
                    {"role": "user", "content": prompt}]
        payload = {"model": self.requested_model, "messages": messages, "temperature": self.temperature,
                   "stream": False}
        if self.max_tokens:
            payload["max_tokens"] = int(self.max_tokens)
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self._key}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise BackendError(self._redact(f"online model returned HTTP {exc.code}: {detail}")) from None
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError(self._redact(f"cannot reach the online model at {self.base_url} ({exc})")) from None
        try:
            out = json.loads(body.decode("utf-8"))
            text = (out["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError):
            raise BackendError(self._redact(f"unexpected reply from the online model: {body[:200]!r}")) from None
        if not text:
            raise BackendError("the online model returned an empty answer")
        usage = out.get("usage") or {}
        return Generation(text, self.name, out.get("model") or self.requested_model,
                          (time.perf_counter() - t0) * 1000.0,
                          prompt_tokens=_count(usage.get("prompt_tokens")),
                          output_tokens=_count(usage.get("completion_tokens")))


class ChainBackend:
    """Try each backend in order; the first answer wins. A real model further down the chain
    is cached like any answer; a placeholder that answers only because everything before it
    failed is marked as a fallback and never cached."""

    def __init__(self, backends: list):
        if not backends:
            raise ValueError("the [llm] chain is empty")
        self.backends = backends
        self.name = "+".join(b.name for b in backends)
        self.primary = backends[0]

    def available(self) -> bool:
        return any(b.available() for b in self.backends)

    def warm_up(self) -> None:
        for b in self.backends:
            if isinstance(b, OllamaBackend):
                b.warm_up()
                return

    def describe(self) -> dict:
        for b in self.backends:
            if isinstance(b, StubBackend) and b is not self.backends[0]:
                continue
            info = b.describe()
            if info.get("available"):
                break
        else:
            info = self.backends[0].describe()
        return {**info, "chain": [b.name for b in self.backends],
                "chain_status": [{"backend": b.name, "available": b.available() if not isinstance(b, OllamaBackend)
                                  else None} for b in self.backends]}

    def generate(self, prompt: str, history: list[dict] | None = None) -> Generation:
        tried: list[dict] = []
        for i, b in enumerate(self.backends):
            try:
                g = b.generate(prompt, history) if history else b.generate(prompt)
            except BackendError as exc:
                tried.append({"backend": b.name, "error": str(exc)})
                continue
            if tried:
                g.tried = tried
                g.error = "; ".join(f"{t['backend']}: {t['error']}" for t in tried)
                if isinstance(b, StubBackend):            # a stand-in, not a model answer
                    g.fallback, g.cacheable = True, False
            return g
        raise BackendError("every backend in the chain failed: "
                           + "; ".join(f"{t['backend']}: {t['error']}" for t in tried))


BACKENDS = ("ollama", "openai_compatible", "stub")


def _make(name: str, llm_settings, stub_cacheable: bool):
    if name == "stub":
        return StubBackend(llm_settings.stub_delay_ms, stub_cacheable)
    if name == "ollama":
        return OllamaBackend(llm_settings.ollama_url, llm_settings.model, llm_settings.temperature,
                             llm_settings.timeout_s, llm_settings.system_prompt)
    if name == "openai_compatible":
        import os
        return OpenAICompatibleBackend(
            os.environ.get("SEMCACHE_OPENAI_BASE_URL") or llm_settings.openai_base_url,
            os.environ.get("SEMCACHE_OPENAI_MODEL") or llm_settings.openai_model,
            os.environ.get(llm_settings.openai_api_key_env), llm_settings.temperature,
            llm_settings.timeout_s, llm_settings.system_prompt, llm_settings.openai_max_tokens)
    raise ValueError(f"unknown llm backend {name!r} (use one of {', '.join(BACKENDS)})")


def build_backend(llm_settings) -> object:
    """Create the backend described by the [llm] section of config.toml.

    Placeholder answers are cached only when the stub is the chosen backend (and
    ``cache_stub_answers`` is on); as a fallback for Ollama they never are.
    """
    chain = list(getattr(llm_settings, "chain", None) or [])
    if chain:
        # [llm] chain = ["openai_compatible", "ollama", "stub"]: the stub caches its placeholder
        # only when it is first in the chain (then it is the chosen backend).
        return ChainBackend([_make(n, llm_settings, i == 0 and llm_settings.cache_stub_answers)
                             for i, n in enumerate(chain)])
    explicit_stub = llm_settings.backend == "stub"
    stub = StubBackend(llm_settings.stub_delay_ms, explicit_stub and llm_settings.cache_stub_answers)
    if llm_settings.backend == "stub":
        return stub
    if llm_settings.backend not in BACKENDS:
        raise ValueError(f"unknown llm backend {llm_settings.backend!r} (use one of {', '.join(BACKENDS)})")
    primary = _make(llm_settings.backend, llm_settings, False)
    return FallbackBackend(primary, stub) if llm_settings.fallback_to_stub else primary
