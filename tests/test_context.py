import pytest

from semcache.context import build_history, detect_follow_up


@pytest.mark.parametrize("text", [
    "and what about saturn?",
    "what about for students",
    "how about the price",
    "tell me more",
    "can you elaborate on the second one",
    "which of those is cheapest",
    "how do I fix it",
    "why?",
    "is that true",
    "you said something about refunds earlier",
    "ok thanks",
    "pricing",
    "the second one",
    "python?",
    "so?",
    "what's the next step?",
    "instead?",
])
def test_follow_ups_are_detected_mid_conversation(text):
    verdict = detect_follow_up(text, has_history=True)
    assert verdict.is_follow_up and verdict.reason


@pytest.mark.parametrize("text", [
    "how do I reset my password",
    "what is the capital of france",
    "my laptop is slow, how do I fix it",
    "cheap flight tickets to goa",
    "is it true that python is slower than c",
    # weak signals (instead, continue, next step, so) next to a subject of their own
    "should i learn java instead of python",
    "how do i continue my studies abroad",
    "so what is svd",
    "what is the next step in my visa process",
])
def test_standalone_questions_stay_standalone(text):
    assert not detect_follow_up(text, has_history=True).is_follow_up


def test_first_message_is_always_standalone():
    assert not detect_follow_up("and what about it?", has_history=False).is_follow_up


def test_history_keeps_the_newest_messages_within_limits():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    h = build_history(msgs, max_messages=4, max_chars=1000)
    assert [m["content"] for m in h] == ["m6", "m7", "m8", "m9"]
    assert h[0]["role"] == "user"


def test_history_char_budget_and_user_first():
    msgs = [{"role": "user", "content": "a" * 50}, {"role": "assistant", "content": "b" * 50},
            {"role": "user", "content": "c" * 50}, {"role": "assistant", "content": "d" * 50}]
    h = build_history(msgs, max_messages=10, max_chars=120)
    # Budget fits two messages (c, d); the history must still start with a user turn.
    assert [m["content"][0] for m in h] == ["c", "d"]
    huge = build_history([{"role": "user", "content": "x" * 500}], 5, 100)
    assert len(huge[0]["content"]) == 100
