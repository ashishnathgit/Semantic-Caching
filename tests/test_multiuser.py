"""Several users sharing one cache: visibility, follow-ups, permissions."""

import numpy as np
import pytest

from semcache import linalg as la
from semcache.backends import StubBackend
from semcache.cache import SemanticCache
from semcache.pipeline import SemanticCacheService, build_artifacts
from semcache.users import ThreadNotFound


def unit(v):
    return la.l2_normalize_rows(np.asarray(v, dtype=float))


# --- cache ---------------------------------------------------------------------

def test_private_entries_are_invisible_to_other_users():
    cache = SemanticCache(dim=2, threshold=0.9)
    shared = cache.add("shared q", "S", unit([0, 1]))
    mine = cache.add("alice q", "A", unit([1, 0]), owner="alice")
    q = unit([1, 0.05])
    assert cache.lookup(q, user="alice").best.id == mine.id
    for other in ("bob", None):
        r = cache.lookup(q, user=other)
        assert not r.hit and all(m.entry.id != mine.id for m in r.matches)
        assert r.compared == 1                           # only the shared row was scored
    assert {e.id for e in cache.entries(user="bob")} == {shared.id}
    assert {e.id for e in cache.entries()} == {shared.id, mine.id}


def test_exact_layer_is_per_owner():
    cache = SemanticCache(dim=2, threshold=0.9)
    shared = cache.add("Where is the library?", "Block C", unit([1, 0]))
    private = cache.add("where is the library", "My notes say B", unit([1, 0]), owner="alice")
    assert len(cache) == 2                               # a private copy doesn't replace the shared one
    assert cache.lookup(unit([0, 1]), query_text="where is the library", user="alice").best.id == private.id
    assert cache.lookup(unit([0, 1]), query_text="where is the library", user="bob").best.id == shared.id
    assert cache.remove_owned_by("alice") == 1
    assert cache.find_exact("where is the library", "alice").id == shared.id


def test_owner_survives_save_load_and_contiguous_removal(tmp_path):
    cache = SemanticCache(dim=2, threshold=0.9)
    a = cache.add("a", "A", unit([1, 0]), owner="alice")
    cache.add("b", "B", unit([0, 1]))
    c = cache.add("c", "C", unit([1, 1]), owner="carol")
    cache.remove(a.id)                                   # the last row moves into a's slot
    assert cache.lookup(unit([1, 1]), user="carol").best.id == c.id
    assert not cache.lookup(unit([1, 1]), user="alice").hit
    cache.save(tmp_path)
    loaded = SemanticCache.load(tmp_path)
    assert loaded.get(c.id).owner == "carol"
    assert loaded.stats()["private_entries"] == 1
    assert not loaded.lookup(unit([1, 1]), user="bob").hit
    assert loaded.lookup(unit([1, 1]), user="carol").hit


def test_lsh_and_hybrid_lookups_respect_visibility():
    rng = np.random.default_rng(1)
    data = unit(rng.standard_normal((300, 16)))
    cache = SemanticCache(dim=16, threshold=0.9, max_entries=1000, use_lsh=True, lsh_bits=6,
                          lsh_tables=8, lsh_min_entries=10, lsa_weight=0.5)
    for i, v in enumerate(data):
        cache.add(str(i), str(i), v, owner="alice" if i % 2 else None,
                  tfidf_row=(np.array([i]), np.array([1.0])))
    for i in range(1, 40, 2):                            # alice's rows
        r = cache.lookup(data[i], user="bob", query_tfidf=(np.array([i]), np.array([1.0])))
        assert all(m.entry.owner is None for m in r.matches)
        r = cache.lookup(data[i], user="alice", query_tfidf=(np.array([i]), np.array([1.0])))
        assert r.matches and r.matches[0].entry.id == i + 1


# --- service --------------------------------------------------------------------

class RecordingBackend(StubBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, prompt, history=None):
        self.calls.append((prompt, history))
        return super().generate(prompt, history)


@pytest.fixture()
def service(settings):
    build_artifacts(settings, verbose=False)
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    svc.backend = RecordingBackend()
    return svc


def test_one_users_answer_is_a_hit_for_another(service):
    first = service.ask("how many moons does jupiter have", user="alice")
    assert not first.hit and first.stored_id and first.thread_id
    second = service.ask("How many moons does Jupiter have?", user="bob")
    assert second.hit and second.matched_id == first.stored_id
    assert second.thread_id != first.thread_id          # each user has their own conversation


def test_follow_up_skips_cache_and_sends_history(service):
    first = service.ask("how many moons does jupiter have", user="alice")
    before = len(service.cache)
    follow = service.ask("and what about saturn?", user="alice", thread_id=first.thread_id)
    assert follow.follow_up and not follow.cache_used and not follow.hit
    assert follow.stored_id is None and len(service.cache) == before     # not cached
    prompt, history = service.backend.calls[-1]
    assert prompt == "and what about saturn?"
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "how many moons does jupiter have"
    assert service.backend.calls[0][1] is None           # standalone answers get no history

    thread = service.users.get_thread("alice", first.thread_id)
    assert len(thread.messages) == 4 and thread.messages[3].meta["follow_up"] is True
    stats = service.stats("alice")["user"]
    assert stats == {"questions": 2, "hits": 0, "misses": 1, "follow_ups": 1, "hit_rate": 0.0}


def test_standalone_question_mid_conversation_uses_the_cache(service):
    first = service.ask("how many moons does jupiter have", user="alice")
    faq = service.ask("i forgot my password, how do i reset it", user="alice", thread_id=first.thread_id)
    assert faq.hit and faq.cache_used and not faq.follow_up
    assert faq.thread_id == first.thread_id


def test_use_cache_overrides_detection(service):
    first = service.ask("how many moons does jupiter have", user="alice")
    forced = service.ask("and what about jupiter", user="alice", thread_id=first.thread_id, use_cache=True)
    assert forced.cache_used and not forced.follow_up
    skipped = service.ask("how many moons does jupiter have", user="alice", thread_id=first.thread_id,
                          use_cache=False)
    assert not skipped.hit and not skipped.cache_used


def test_thread_errors(service):
    with pytest.raises(ValueError, match="thread needs a user"):
        service.ask("hello there friend", thread_id="0123456789ab")
    with pytest.raises(ThreadNotFound):
        service.ask("hello there friend", user="alice", thread_id="0123456789ab")
    t = service.ask("what is the capital of france", user="alice").thread_id
    with pytest.raises(ThreadNotFound):
        service.ask("and germany", user="bob", thread_id=t)


def test_private_entries_and_permissions(service):
    entry = service.add_entry("where is the exam hall", "Room 204", user="alice")
    assert entry.owner == "alice"
    assert service.ask("where is the exam hall", user="alice").hit
    bob = service.ask("where is the exam hall", user="bob")
    assert not bob.hit or bob.matched_id != entry.id

    assert service.remove_entry(entry.id, user="bob") is False        # invisible to bob
    faq_id = next(e.id for e in service.visible_entries("bob") if e.source == "faq")
    with pytest.raises(PermissionError):
        service.remove_entry(faq_id, user="bob")
    with pytest.raises(PermissionError):
        service.add_entry("shared q here", "x", user="bob", shared=True)
    assert service.add_entry("shared q here", "x", user="admin", shared=True, admin=True).owner is None
    assert service.remove_entry(faq_id, user="admin", admin=True)
    assert service.remove_entry(entry.id, user="alice")


def test_reset_scope_depends_on_role(service):
    service.add_entry("where is the exam hall", "Room 204", user="alice")
    service.ask("how many moons does jupiter have", user="alice")
    total = len(service.cache)
    r = service.reset_for("alice")
    assert r == {"scope": "private", "entries": total - 1, "faq_entries": None, "removed": 1}
    assert service.stats("alice")["user"]["questions"] == 0
    assert service.reset_for("admin")["scope"] == "private"           # the name alone isn't enough
    r = service.reset_for("admin", admin=True)
    assert r["scope"] == "all" and r["entries"] == r["faq_entries"]


def test_anonymous_ask_still_works_without_threads(service):
    r = service.ask("how many moons does jupiter have")
    assert r.thread_id is None and r.user is None and r.cache_used
    assert service.users.list_users() == []


def test_concurrent_users_keep_separate_consistent_threads(service):
    from concurrent.futures import ThreadPoolExecutor

    questions = ["how many moons does jupiter have", "and what about saturn?", "tell me more",
                 "what is the capital of france", "and germany?"]

    def converse(name):
        tid = None
        for q in questions:
            tid = service.ask(q, user=name, thread_id=tid).thread_id
        return name, tid

    names = [f"user{i}" for i in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(converse, names))
    for name, tid in results:
        thread = service.users.get_thread(name, tid)
        assert [m.content for m in thread.messages if m.role == "user"] == questions
        assert service.stats(name)["user"]["questions"] == len(questions)
    # The standalone questions were stored once each, however many users raced on them.
    assert sum(e.query == "how many moons does jupiter have" for e in service.visible_entries(None)) == 1
