import json

import pytest

from semcache.users import Message, ThreadNotFound, UserStore, normalize_username


@pytest.mark.parametrize("raw, clean", [("Alice", "alice"), ("  bob.smith ", "bob.smith"), ("a_1-b", "a_1-b")])
def test_usernames_are_normalized(raw, clean):
    assert normalize_username(raw) == clean


@pytest.mark.parametrize("bad", ["", "  ", "../etc", ".hidden", "a/b", "x" * 33, "naïve", "a b"])
def test_unsafe_usernames_are_rejected(bad):
    with pytest.raises(ValueError):
        normalize_username(bad)


def test_threads_persist_and_are_private(tmp_path):
    store = UserStore(tmp_path)
    t = store.create_thread("alice")
    store.append("alice", t.id, Message("user", "how many moons does jupiter have", 1.0),
                 Message("assistant", "95", 1.0, {"hit": False}))
    assert store.get_thread("alice", t.id).title == "how many moons does jupiter have"
    with pytest.raises(ThreadNotFound):
        store.get_thread("bob", t.id)                    # another user can't open it

    reloaded = UserStore(tmp_path)                       # fresh process: read from disk
    thread = reloaded.get_thread("alice", t.id)
    assert [m.content for m in thread.messages] == ["how many moons does jupiter have", "95"]
    assert thread.messages[1].meta == {"hit": False}
    assert reloaded.list_users() == ["alice"]


def test_thread_ids_cannot_escape_the_user_folder(tmp_path):
    store = UserStore(tmp_path)
    store.create_thread("alice")
    for bad in ["../../x", "ABCDEF123456", "", "0" * 11]:
        with pytest.raises(ThreadNotFound):
            store.get_thread("alice", bad)


def test_oldest_thread_is_dropped_beyond_the_limit(tmp_path):
    store = UserStore(tmp_path, max_threads=2)
    first = store.create_thread("alice")
    store.create_thread("alice")
    store.create_thread("alice")
    ids = {t.id for t in store.list_threads("alice")}
    assert len(ids) == 2 and first.id not in ids
    assert not (tmp_path / "alice" / "threads" / f"{first.id}.json").exists()


def test_messages_are_capped_per_thread(tmp_path):
    store = UserStore(tmp_path, max_messages=3)
    t = store.create_thread("alice")
    for i in range(5):
        store.append("alice", t.id, Message("user", f"q{i}", float(i)))
    assert [m.content for m in store.get_thread("alice", t.id).messages] == ["q2", "q3", "q4"]


def test_stats_rename_and_delete(tmp_path):
    store = UserStore(tmp_path)
    store.record("alice", hit=True, follow_up=False)
    store.record("alice", hit=False, follow_up=False)
    store.record("alice", hit=None, follow_up=True)
    st = store.profile("alice").stats()
    assert st == {"questions": 3, "hits": 1, "misses": 1, "follow_ups": 1, "hit_rate": 0.5}
    saved = json.loads((tmp_path / "alice" / "profile.json").read_text())
    assert saved["stats"]["questions"] == 3

    t = store.create_thread("alice", "draft")
    assert store.rename_thread("alice", t.id, "  Trip plans  ").title == "Trip plans"
    assert store.delete_thread("alice", t.id) and not store.delete_thread("alice", t.id)
    assert store.reset_stats("alice").stats()["questions"] == 0
