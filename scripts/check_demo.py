"""Check the UI's demo-script presets against a fresh FAQ profile after a Reset demo.

    python scripts/check_demo.py            (uses a temporary folder; your artifacts are untouched)

Reads the presets from data/demo_script.json, asks them in order in one
conversation (the stub answers misses, as the LLM would, so the run is repeatable), and
prints what each one did next to what the script expects. Exit code 1 if any differ.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def presets() -> list[dict]:
    return json.loads((ROOT / "data" / "demo_script.json").read_text(encoding="utf-8"))["steps"]


def kind(r) -> str:
    if r.follow_up and not r.cache_used:
        return "follow-up"
    if r.hit:
        return "hit"
    if r.note and "only an exact repeat can hit" in r.note:
        return "blocked"
    return "miss"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SEMCACHE_ARTIFACTS"] = tmp
        from semcache.config import Settings
        from semcache.pipeline import SemanticCacheService

        settings = Settings.load(ROOT / "config.toml", profile="faq")
        settings.llm.backend = "stub"
        service = SemanticCacheService.from_settings(settings, verbose=False)
        service.reset_demo(admin=True)
        thread, failed = None, 0
        print(f"FAQ profile after Reset demo: {len(service.cache)} answers, tau = {settings.cache.threshold}\n")
        print(f"{'#':>2}  {'preset':32} {'expect':10} {'got':10} {'score':>6}  {'rho':>5}  matched")
        for i, p in enumerate(presets(), 1):
            r = service.ask(p["query"], user="demo", thread_id=thread)
            thread = r.thread_id
            got = kind(r)
            failed += got != p["expect"]
            matched = r.matched_query if r.matches else "-"
            print(f"{i:>2}  {p['label']:32} {p['expect']:10} {got:10} {r.score:6.3f}  {r.rho:5.2f}  {matched}"
                  + ("" if got == p["expect"] else "   <-- differs"))
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
