"""Pre-flight check, to run right before going on stage:

    python -m semcache preflight --profile faq

Prints PASS or FAIL for each item and returns False if anything failed:
  * Ollama is reachable, has the configured model, and the model is loaded (warm);
  * the cache holds exactly the FAQ answers (as after `reset-demo`);
  * each preset in data/demo_script.json gives its expected result.

The presets are checked with lookups only: nothing is stored, nothing is counted, no
metric event is written and the LLM is not asked. Preset 4 needs the row preset 3 adds
to C; that row is added to a copy of the cache in memory, never to the real one.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Callable

from .backends import OllamaBackend
from .config import Settings
from .context import detect_follow_up
from .data import load_faq
from .pipeline import SemanticCacheService


def load_demo_script(settings: Settings) -> list[dict]:
    path = settings.path("data/demo_script.json")
    return json.loads(path.read_text(encoding="utf-8"))["steps"]


def _ollama(backend) -> OllamaBackend | None:
    inner = getattr(backend, "primary", backend)
    return inner if isinstance(inner, OllamaBackend) else None


def run_preflight(settings: Settings, out: Callable[[str], None] = print,
                  service: SemanticCacheService | None = None) -> bool:
    service = service or SemanticCacheService.from_settings(settings, build_if_missing=False, verbose=False)
    results: list[bool] = []

    def check(ok: bool, text: str) -> bool:
        results.append(bool(ok))
        out(f"{'PASS' if ok else 'FAIL'}  {text}")
        return bool(ok)

    out(f"Preflight: profile {settings.profile!r}, k = {service.embedder.dim}, "
        f"{len(service.embedder.vocab):,} terms, tau = {settings.cache.threshold:.2f}\n")

    # --- the model ---------------------------------------------------------------------
    ollama = _ollama(service.backend)
    if settings.llm.backend == "stub" or ollama is None:
        check(False, "LLM backend is the stub ([llm] backend = \"stub\"): misses get placeholder answers")
    else:
        info = ollama.describe()
        if check(bool(info.get("available")),
                 f"Ollama reachable at {ollama.base_url}"
                 + (f", model {info.get('model')}" if info.get("available") else f": {info.get('error', 'no models')}")):
            try:
                model = ollama.model
                if any(m and m.split(":")[0] == model.split(":")[0] for m in ollama.loaded_models()):
                    check(True, f"{model} is loaded in memory (warm)")
                else:
                    t0 = time.perf_counter()
                    ollama.warm_up()
                    loaded = any(m and m.split(":")[0] == model.split(":")[0] for m in ollama.loaded_models())
                    check(loaded, f"{model} warmed up now ({time.perf_counter() - t0:.1f} s)")
            except Exception as exc:                       # noqa: BLE001 - report, don't crash
                check(False, f"model warm-up failed: {exc}")

    # --- the cache ------------------------------------------------------------------------
    faq = {i.question for i in load_faq(settings.path(settings.paths.faq_csv))}
    entries = service.cache.entries()
    others = [e for e in entries if e.source != "faq"]
    exact = len(entries) == len(faq) and not others and {e.query for e in entries} == faq
    check(exact, f"cache holds exactly the {len(faq)} FAQ answers" if exact else
          f"cache has {len(entries)} entries ({len(entries) - len(others)} FAQ, {len(others)} others): "
          f"run python -m semcache reset-demo --profile {settings.profile}")

    # --- the presets (lookups only) ---------------------------------------------------------
    ratio = settings.cache.min_known_ratio
    shadow = copy.deepcopy(service.cache)          # preset 3's new row goes here, in memory only
    added: dict[int, int] = {}

    def look(cache, text: str):
        vector, row, rho = service._encode(text)
        return cache.lookup(vector, 1, None, record=False, query_text=text, query_tfidf=row, query_rho=rho)

    for n, step in enumerate(load_demo_script(settings), 1):
        q, expect = step["query"], step["expect"]
        if expect == "follow-up":
            f = detect_follow_up(q, has_history=True)
            got, detail = ("follow-up" if f.is_follow_up else "standalone"), f.reason or "no follow-up signal"
        else:
            r = look(shadow if step.get("after") else service.cache, q)
            known, total = service.embedder.coverage(q)
            blocked = total == 0 or known == 0 or known / total < ratio
            got = "hit" if r.hit else "blocked" if blocked else "miss"
            best = r.matches[0] if r.matches else None
            detail = f"{r.score:.3f}" + (f" vs “{best.entry.query}”" if best else "")
            if step.get("after"):
                row_id = added.get(step["after"])
                on_row = r.hit and row_id is not None and best is not None and best.entry.id == row_id
                got = got if on_row or not r.hit else "hit on another row"
                detail += f", on preset {step['after']}'s new row (simulated)" if on_row else ""
            if expect == "miss":                    # what preset 4 needs: preset 3's row in C
                vector, row, rho = service._encode(q)
                added[n] = shadow.add(q, "(preflight: simulated answer)", vector, source="ollama",
                                      tfidf_row=row, rho=rho).id
        check(got == expect, f"preset {n} {step['label']}: expected {expect}, got {got} ({detail})")

    passed = all(results)
    out("")
    out(f"{'All' if passed else 'Only'} {sum(results)} of {len(results)} checks passed"
        + (": ready for the stage." if passed else ": fix the FAIL lines before going on stage."))
    return passed
