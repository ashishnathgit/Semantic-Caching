"""Request metrics: counters, latency, estimated LLM cost and a time series.

Every answered question appends one event to artifacts/<profile>/metrics/events.jsonl
(append-only, so the charts survive a restart). Everything shown in the UI is computed
from these events, for everyone or for one user.

Event fields
    t           unix time
    user        who asked (None = anonymous)
    kind        "hit" | "miss" | "follow_up" | "refresh"
    source      "CACHE" | "LLM" | "FOLLOW-UP" | "FALLBACK" | "NONE" (model not called)
    llm         an LLM backend was called
    fallback    Ollama failed and the stub answered
    backend     "ollama" | "openai_compatible" | "stub" | None (the backend that answered)
    total_ms, llm_ms
    prompt_tokens, output_tokens       reported by Ollama (None otherwise)
    entry_tokens                       for a hit: the tokens the cached answer cost when it
                                       was generated, if known ([prompt, output])
    score, rho

Cost is a what-if (Ollama is free locally): it is computed only when a price is set in
[metrics], and always labelled as an estimate.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MAX_BUCKETS = 720
HIT_RATE_LABEL = "Hit rate (of cache lookups)"
HIT_RATE_BASIS = ("hits / (hits + misses): follow-ups and forced refreshes skip the cache, "
                  "so they are left out")


@dataclass
class Pricing:
    """What-if prices from [metrics] in config.toml; None = not set."""
    per_1k_input: float | None = None
    per_1k_output: float | None = None
    per_call: float | None = None
    note: str = ""

    @property
    def configured(self) -> bool:
        return any(v is not None for v in (self.per_1k_input, self.per_1k_output, self.per_call))

    def label(self) -> str | None:
        if not self.configured:
            return None
        parts = []
        if self.per_1k_input is not None or self.per_1k_output is not None:
            parts.append(f"{_usd(self.per_1k_input or 0.0)} per 1K input tokens and "
                         f"{_usd(self.per_1k_output or 0.0)} per 1K output tokens")
        if self.per_call is not None:
            parts.append(f"{_usd(self.per_call)} per LLM call")
        return "estimated at " + " plus ".join(parts)

    def cost(self, prompt_tokens: float | None, output_tokens: float | None) -> float | None:
        """Cost of one call; None when there is nothing to price it with."""
        total, priced = 0.0, False
        if prompt_tokens is not None and self.per_1k_input is not None:
            total += prompt_tokens / 1000 * self.per_1k_input
            priced = True
        if output_tokens is not None and self.per_1k_output is not None:
            total += output_tokens / 1000 * self.per_1k_output
            priced = True
        if self.per_call is not None:
            total += self.per_call
            priced = True
        return total if priced else None

    def describe(self) -> dict:
        return {"configured": self.configured, "per_1k_input_tokens_usd": self.per_1k_input,
                "per_1k_output_tokens_usd": self.per_1k_output, "per_llm_call_usd": self.per_call,
                "label": self.label(), "note": self.note or None}


def _usd(value: float) -> str:
    return f"${value:,.6f}".rstrip("0").rstrip(".") if value < 0.01 else f"${value:,.2f}"


def parse_bucket(text: str | int) -> int:
    """'10s', '1m', '1h' or a number of seconds -> seconds (1 s to 1 day)."""
    if isinstance(text, int):
        seconds = text
    else:
        m = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", str(text).lower())
        if not m:
            raise ValueError(f"bucket must look like 10s, 1m or 1h, not {text!r}")
        seconds = int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    if not 1 <= seconds <= 86400:
        raise ValueError("bucket must be between 1 second and 1 day")
    return seconds


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class MetricsStore:
    def __init__(self, directory: str | Path, enabled: bool = True):
        self.directory = Path(directory)
        self.path = self.directory / "events.jsonl"
        self.enabled = enabled
        self._lock = threading.Lock()
        self._events: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue                     # a torn last line after a crash: skip it
        return events

    # ----------------------------------------------------------------- writing
    def record(self, event: dict) -> None:
        if not self.enabled:
            return
        event = {"t": time.time(), **event}
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events.append(event)

    def reset(self) -> dict:
        """Start over. The old file is kept as events-<time>.jsonl, not deleted."""
        with self._lock:
            archived = None
            if self.path.exists():
                archived = self.directory / f"events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
                self.path.replace(archived)
            removed = len(self._events)
            self._events = []
        return {"removed_events": removed, "archived_to": str(archived) if archived else None}

    def events(self, user: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._events)
        return items if user is None else [e for e in items if e.get("user") == user]

    # ----------------------------------------------------------------- reading
    def summary(self, pricing: Pricing, user: str | None = None) -> dict:
        ev = self.events(user)
        hits = [e for e in ev if e["kind"] == "hit"]
        misses = [e for e in ev if e["kind"] == "miss"]
        llm = [e for e in ev if e.get("llm")]
        real_llm = [e for e in llm if not e.get("fallback")]
        looked_up = len(hits) + len(misses)
        avg_llm = _mean([e["llm_ms"] for e in real_llm])
        avg_hit = _mean([e["total_ms"] for e in hits])
        # None ("—" in the UI) until a non-fallback LLM answer gives a latency to compare with.
        saved_ms = None if avg_llm is None else len(hits) * max(0.0, avg_llm - (avg_hit or 0.0))

        counted = [e for e in real_llm if e.get("prompt_tokens") is not None or e.get("output_tokens") is not None]
        tokens_in = sum(e.get("prompt_tokens") or 0 for e in counted)
        tokens_out = sum(e.get("output_tokens") or 0 for e in counted)
        avg_in = tokens_in / len(counted) if counted else None
        avg_out = tokens_out / len(counted) if counted else None

        out = {
            "scope": "user" if user else "global", "user": user,
            "requests": len(ev), "hits": len(hits), "misses": len(misses),
            "hit_rate": len(hits) / looked_up if looked_up else None,
            "hit_rate_label": HIT_RATE_LABEL, "hit_rate_basis": HIT_RATE_BASIS,
            "hit_rate_of_requests": len(hits) / len(ev) if ev else None,
            "llm_calls": len(llm), "llm_fallbacks": sum(bool(e.get("fallback")) for e in ev),
            "follow_ups": sum(e["kind"] == "follow_up" for e in ev),
            "forced_refreshes": sum(e["kind"] == "refresh" for e in ev),
            "latency_ms": {"avg": _mean([e["total_ms"] for e in ev]), "avg_hit": avg_hit,
                           "avg_llm": avg_llm, "hit_samples": len(hits), "llm_samples": len(real_llm)},
            "time_saved_ms": saved_ms,
            "time_saved_basis": "cache hits x (average LLM latency - average cache-hit latency)",
            "tokens": {"input": tokens_in, "output": tokens_out, "calls_with_counts": len(counted),
                       "avg_input_per_call": avg_in, "avg_output_per_call": avg_out},
            "first_event": ev[0]["t"] if ev else None,
            # which backend answered the LLM calls (Ollama, an online model, the placeholder)
            "by_backend": {b: sum(e.get("backend") == b for e in llm)
                           for b in sorted({e.get("backend") or "unknown" for e in llm})},
        }
        out["cost"] = self._cost(pricing, real_llm, hits, avg_in, avg_out) if pricing.configured else None
        return out

    @staticmethod
    def _cost(pricing: Pricing, real_llm: list[dict], hits: list[dict],
              avg_in: float | None, avg_out: float | None) -> dict:
        spent = [pricing.cost(e.get("prompt_tokens"), e.get("output_tokens")) for e in real_llm]
        from_entry = from_average = unpriced = 0
        saved = 0.0
        for e in hits:
            tokens = e.get("entry_tokens")
            if tokens and any(t is not None for t in tokens):
                value = pricing.cost(*tokens)
                from_entry += value is not None
            else:
                value = pricing.cost(avg_in, avg_out)
                from_average += value is not None
            if value is None:
                unpriced += 1
            else:
                saved += value
        priced = [v for v in spent if v is not None]
        return {"estimated": True, "label": pricing.label(), "note": pricing.note or None,
                # None ("—") until there is something to price
                "llm_cost_usd": sum(priced) if priced else None, "llm_calls_priced": len(priced),
                "savings_usd": saved if from_entry + from_average else None,
                "hits_priced_from_entry": from_entry,
                "hits_priced_from_average": from_average, "hits_not_priced": unpriced}

    def series(self, bucket_seconds: int, user: str | None = None, now: float | None = None,
               max_buckets: int = MAX_BUCKETS) -> dict:
        """Per-bucket counts plus the cumulative hit rate and request total at the end of
        each bucket. Only the last ``max_buckets`` buckets are returned, but the cumulative
        values count every event."""
        ev = self.events(user)
        b = bucket_seconds
        if not ev:
            return {"bucket_seconds": b, "points": [], "truncated": False}
        now = time.time() if now is None else now
        first = math.floor(ev[0]["t"] / b) * b
        last = math.floor(max(now, ev[-1]["t"]) / b) * b
        start = max(first, last - (max_buckets - 1) * b)
        n = int((last - start) // b) + 1
        req, hit, miss = [0] * n, [0] * n, [0] * n
        cum_req = cum_hit = cum_miss = 0
        for e in ev:
            if e["t"] < start:                    # before the window: only the running totals
                cum_req += 1
                cum_hit += e["kind"] == "hit"
                cum_miss += e["kind"] == "miss"
                continue
            i = min(n - 1, int((e["t"] - start) // b))
            req[i] += 1
            hit[i] += e["kind"] == "hit"
            miss[i] += e["kind"] == "miss"
        points = []
        for i in range(n):
            cum_req, cum_hit, cum_miss = cum_req + req[i], cum_hit + hit[i], cum_miss + miss[i]
            looked = cum_hit + cum_miss
            points.append({"t": start + i * b, "requests": req[i], "hits": hit[i], "misses": miss[i],
                           "total_requests": cum_req,
                           "hit_rate": cum_hit / looked if looked else None})
        return {"bucket_seconds": b, "points": points, "truncated": start > first,
                "hit_rate_basis": HIT_RATE_BASIS}
