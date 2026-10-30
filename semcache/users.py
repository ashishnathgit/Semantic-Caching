"""Users and their conversation threads, kept on disk next to the cache.

Users pick a name; there are no passwords. That is enough to keep each person's
conversations, counters and private answers apart on a demo or a trusted
network, but not to stop someone from using another person's name.

    artifacts/users/<name>/profile.json
    artifacts/users/<name>/threads/<thread id>.json

Every change is written straight away (to a temporary file, then renamed), so a
crash never leaves half a file behind.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
THREAD_ID_RE = re.compile(r"^[0-9a-f]{12}$")
TITLE_CHARS = 80


class ThreadNotFound(KeyError):
    """The user has no conversation with this id."""


def normalize_username(name: str) -> str:
    """Lower-cased name; 1-32 characters of a-z, 0-9, '.', '_' or '-', starting with a
    letter or digit (so it is always a safe folder name)."""
    if not isinstance(name, str):
        raise ValueError("user name must be a string")
    clean = name.strip().lower()
    if not USERNAME_RE.match(clean):
        raise ValueError("user name must be 1-32 characters: letters, digits, '.', '_' or '-', "
                         "starting with a letter or digit")
    return clean


def _title_from(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


@dataclass
class Message:
    role: str                        # "user" or "assistant"
    content: str
    created_at: float
    meta: dict = field(default_factory=dict)   # for answers: hit, source, score, follow_up, ...


@dataclass
class Thread:
    id: str
    owner: str
    title: str
    created_at: float
    updated_at: float
    messages: list[Message] = field(default_factory=list)

    def summary(self) -> dict:
        last = self.messages[-1].content if self.messages else ""
        return {"id": self.id, "title": self.title, "created_at": self.created_at,
                "updated_at": self.updated_at, "message_count": len(self.messages),
                "last_message": _title_from(last) if last else None}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Thread":
        msgs = [Message(**m) for m in d.get("messages", [])]
        return cls(d["id"], d["owner"], d["title"], d["created_at"], d["updated_at"], msgs)


@dataclass
class UserProfile:
    name: str
    created_at: float
    last_seen_at: float
    questions: int = 0               # messages asked
    hits: int = 0                    # standalone questions answered from the cache
    misses: int = 0                  # standalone questions that went to the LLM
    follow_ups: int = 0              # follow-ups answered with the thread's history

    def stats(self) -> dict:
        looked_up = self.hits + self.misses
        return {"questions": self.questions, "hits": self.hits, "misses": self.misses,
                "follow_ups": self.follow_ups,
                "hit_rate": round(self.hits / looked_up, 4) if looked_up else None}

    def to_dict(self) -> dict:
        return {"name": self.name, "created_at": self.created_at, "last_seen_at": self.last_seen_at,
                "stats": self.stats()}


class UserStore:
    """Profiles and threads for every user, loaded lazily and saved on each change."""

    def __init__(self, directory: str | Path, max_threads: int = 100, max_messages: int = 200):
        self.directory = Path(directory)
        self.max_threads = max_threads
        self.max_messages = max_messages
        self._lock = threading.RLock()
        self._profiles: dict[str, UserProfile] = {}
        self._threads: dict[str, dict[str, Thread]] = {}

    # ------------------------------------------------------------------ users
    def _user_dir(self, name: str) -> Path:
        return self.directory / name

    def exists(self, name: str) -> bool:
        name = normalize_username(name)
        return name in self._profiles or (self._user_dir(name) / "profile.json").exists()

    def profile(self, name: str, create: bool = True) -> UserProfile:
        """The user's profile; created on first use unless ``create=False`` (then KeyError)."""
        name = normalize_username(name)
        with self._lock:
            if name in self._profiles:
                return self._profiles[name]
            path = self._user_dir(name) / "profile.json"
            if path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                stats = d.get("stats", {})
                profile = UserProfile(name, d["created_at"], d.get("last_seen_at", d["created_at"]),
                                      stats.get("questions", 0), stats.get("hits", 0),
                                      stats.get("misses", 0), stats.get("follow_ups", 0))
            elif create:
                now = time.time()
                profile = UserProfile(name, now, now)
                _write_json(path, profile.to_dict())
            else:
                raise KeyError(name)
            self._profiles[name] = profile
            return profile

    def list_users(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.name for p in self.directory.iterdir()
                      if (p / "profile.json").exists() and USERNAME_RE.match(p.name))

    def record(self, name: str, *, hit: bool | None, follow_up: bool) -> UserProfile:
        """Count one question: ``hit`` is None when the cache wasn't consulted."""
        with self._lock:
            p = self.profile(name)
            p.questions += 1
            p.last_seen_at = time.time()
            if follow_up:
                p.follow_ups += 1
            elif hit is True:
                p.hits += 1
            elif hit is False:
                p.misses += 1
            _write_json(self._user_dir(p.name) / "profile.json", p.to_dict())
            return p

    def reset_stats(self, name: str) -> UserProfile:
        with self._lock:
            p = self.profile(name)
            p.questions = p.hits = p.misses = p.follow_ups = 0
            _write_json(self._user_dir(p.name) / "profile.json", p.to_dict())
            return p

    def delete_user(self, name: str) -> bool:
        name = normalize_username(name)
        with self._lock:
            self._profiles.pop(name, None)
            self._threads.pop(name, None)
            folder = self._user_dir(name)
            if not folder.exists():
                return False
            shutil.rmtree(folder)
            return True

    # ---------------------------------------------------------------- threads
    def _thread_path(self, name: str, thread_id: str) -> Path:
        return self._user_dir(name) / "threads" / f"{thread_id}.json"

    def _load_threads(self, name: str) -> dict[str, Thread]:
        if name not in self._threads:
            threads: dict[str, Thread] = {}
            folder = self._user_dir(name) / "threads"
            if folder.exists():
                for path in folder.glob("*.json"):
                    if THREAD_ID_RE.match(path.stem):
                        t = Thread.from_dict(json.loads(path.read_text(encoding="utf-8")))
                        threads[t.id] = t
            self._threads[name] = threads
        return self._threads[name]

    def _save_thread(self, thread: Thread) -> None:
        _write_json(self._thread_path(thread.owner, thread.id), thread.to_dict())

    def list_threads(self, name: str) -> list[Thread]:
        """The user's threads, most recently active first."""
        name = normalize_username(name)
        with self._lock:
            self.profile(name)
            return sorted(self._load_threads(name).values(), key=lambda t: t.updated_at, reverse=True)

    def create_thread(self, name: str, title: str | None = None) -> Thread:
        name = normalize_username(name)
        with self._lock:
            self.profile(name)
            threads = self._load_threads(name)
            while len(threads) >= self.max_threads:      # drop the least recently used thread
                oldest = min(threads.values(), key=lambda t: t.updated_at)
                self.delete_thread(name, oldest.id)
            now = time.time()
            thread_id = secrets.token_hex(6)
            while thread_id in threads:
                thread_id = secrets.token_hex(6)
            thread = Thread(thread_id, name, _title_from(title) if title and title.strip() else "New conversation",
                            now, now)
            threads[thread_id] = thread
            self._save_thread(thread)
            return thread

    def get_thread(self, name: str, thread_id: str) -> Thread:
        """ThreadNotFound if the user has no such thread."""
        name = normalize_username(name)
        if not isinstance(thread_id, str) or not THREAD_ID_RE.match(thread_id):
            raise ThreadNotFound(thread_id)
        with self._lock:
            thread = self._load_threads(name).get(thread_id)
        if thread is None:
            raise ThreadNotFound(thread_id)
        return thread

    def rename_thread(self, name: str, thread_id: str, title: str) -> Thread:
        if not title or not title.strip():
            raise ValueError("title must not be empty")
        with self._lock:
            thread = self.get_thread(name, thread_id)
            thread.title = _title_from(title)
            thread.updated_at = time.time()
            self._save_thread(thread)
            return thread

    def delete_thread(self, name: str, thread_id: str) -> bool:
        name = normalize_username(name)
        with self._lock:
            try:
                self.get_thread(name, thread_id)
            except ThreadNotFound:
                return False
            del self._load_threads(name)[thread_id]
            self._thread_path(name, thread_id).unlink(missing_ok=True)
            return True

    def append(self, name: str, thread_id: str, *messages: Message) -> Thread:
        """Add messages to a thread. The first question becomes the title of an untitled
        thread; only the newest ``max_messages`` are kept."""
        with self._lock:
            thread = self.get_thread(name, thread_id)
            if thread.title == "New conversation" and not thread.messages:
                first = next((m for m in messages if m.role == "user"), None)
                if first:
                    thread.title = _title_from(first.content)
            thread.messages.extend(messages)
            if len(thread.messages) > self.max_messages:
                thread.messages = thread.messages[-self.max_messages:]
            thread.updated_at = time.time()
            self._save_thread(thread)
            return thread
