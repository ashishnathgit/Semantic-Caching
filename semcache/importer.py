"""Import an FAQ CSV (columns question, answer) as a NEW profile, e.g. "campus".

The CSV, a profile.toml with the profile's overrides, the model and the warm cache all go
to artifacts/<name>/; config.toml and every existing profile (faq above all) are never
touched. Serve it with  python -m semcache serve --profile <name>.

Optional columns: paraphrases and heldout (" | "-separated, as in data/faq.csv), topic, id.
"""

from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path

from .config import IMPORTED_PROFILE_FILE, Settings, imported_profiles, valid_profile_name

MAX_BYTES = 2_000_000
MAX_ROWS = 5000


def parse_faq_csv(text: str) -> list[dict]:
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"the CSV is larger than {MAX_BYTES // 1_000_000} MB")
    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    fields = {f.strip().lower(): f for f in (reader.fieldnames or [])}
    if not {"question", "answer"} <= set(fields):
        raise ValueError("the CSV needs a header row with the columns question and answer")
    rows, seen = [], set()
    for raw in reader:
        q = (raw.get(fields["question"]) or "").strip()
        a = (raw.get(fields["answer"]) or "").strip()
        if not q or not a:
            continue
        key = " ".join(q.lower().split())
        if key in seen:
            continue
        seen.add(key)
        get = lambda name: (raw.get(fields[name]) or "").strip() if name in fields else ""  # noqa: E731
        rows.append({"id": get("id") or f"q{len(rows) + 1}", "topic": get("topic") or "imported",
                     "question": q, "answer": a, "paraphrases": get("paraphrases"), "heldout": get("heldout")})
        if len(rows) > MAX_ROWS:
            raise ValueError(f"the CSV has more than {MAX_ROWS:,} questions")
    if len(rows) < 3:
        raise ValueError("the CSV needs at least 3 rows with both a question and an answer")
    return rows


def import_faq_csv(base: Settings, name: str, text: str, verbose: bool = False) -> dict:
    """Create profile ``name`` from the CSV and build its model and cache."""
    from .pipeline import build_artifacts
    name = (name or "").strip().lower()
    if not valid_profile_name(name):
        raise ValueError("a profile name may only use letters, digits, '-' and '_' (at most 40)")
    configured = set(base.profiles) - set(imported_profiles(base.artifacts_root))
    if name in configured:
        raise ValueError(f"{name!r} is a profile in config.toml; importing never replaces one. Pick a new name")
    folder = base.artifacts_root / name
    if folder.exists() and any(folder.iterdir()) and not (folder / IMPORTED_PROFILE_FILE).exists():
        raise ValueError(f"{folder} already exists and is not an imported profile")
    rows = parse_faq_csv(text)
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "source.csv"
    with open(source, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "topic", "question", "answer", "paraphrases", "heldout"])
        writer.writeheader()
        writer.writerows(rows)
    has_heldout = any(r["heldout"] for r in rows)
    toml = "\n".join([
        f"# Made by Import FAQ CSV on {time.strftime('%Y-%m-%d %H:%M')}. Delete this folder to remove the profile.",
        "[paths]", f"faq_csv = {_toml_str(source)}",
        "[corpus]", "use_faq = true", "use_quora = false", "require_quora = false",
        "include_cached = true      # no paraphrases here, so the answers teach which words go together",
        "[eval]", 'pairs = "faq"', ""])
    (folder / IMPORTED_PROFILE_FILE).write_text(toml, encoding="utf-8")
    settings = Settings.load(base.root / "config.toml" if (base.root / "config.toml").exists() else None,
                             profile=name)
    embedder, cache = build_artifacts(settings, verbose=verbose, fresh=True)
    return {"profile": name, "questions": len(rows), "entries": len(cache), "k": embedder.dim,
            "vocabulary_size": len(embedder.vocab), "training_documents": embedder.report.n_docs,
            "folder": str(folder), "evaluable": has_heldout,
            "serve": f"python -m semcache serve --profile {name}"}


def _toml_str(path: Path) -> str:
    return '"' + re.sub(r'(["\\])', r"\\\1", str(path)) + '"'
