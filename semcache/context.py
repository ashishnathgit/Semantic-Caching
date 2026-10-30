"""Conversation context: is a message a follow-up, and which history goes to the LLM.

The cache is shared, so it may only hold answers that make sense without any
conversation around them. A follow-up such as "what about the second one?"
depends on earlier messages: a cached answer would be wrong for it, and its own
answer must not be cached for other users. So follow-ups skip the cache and go
to the LLM together with the recent history of their thread.

The detector is deliberately cautious. Taking a standalone question for a
follow-up only costs one LLM call; the opposite mistake serves a wrong answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .text import STOPWORDS, _TOKEN, normalize

# Opening words that continue the previous turn ("and for students?").
_OPENERS = ("and", "but", "or", "what about", "how about")

# Phrases that point back at something said before.
_BACK_REFERENCES = [re.compile(p) for p in (
    r"\b(first|second|third|fourth|fifth|last|other|previous|former|latter) "
    r"(one|ones|option|options|step|steps|point|thing|answer|method|way|link)\b",
    r"\bnext (one|ones|option|options)\b",
    r"\bthat one\b",
    r"\b(you|u) (said|mentioned|suggested|wrote|told|recommended|listed)\b",
    r"\b(above|earlier|your answer|last answer|previous answer)\b",
    r"\b(tell me more|more details?|elaborate|go on|keep going|expand on)\b",
    r"\bexplain (that|it|this|more|further)\b",
    r"\b(in that case|what else|anything else)\b",
    r"\b(example|examples) of (that|it|this|those)\b",
)]

# Weak signals: words that often continue a conversation ("so?", "what's the next step?")
# but also appear in standalone questions ("should i learn java instead of python",
# "so what is svd", "how do i continue my studies abroad"). They count only when the
# message names nothing of its own.
_WEAK_OPENERS = ("so", "then", "also", "ok", "okay")
_WEAK_SIGNALS = [re.compile(p) for p in (
    r"\binstead\b", r"\bcontinue\b", r"\bnext steps?\b", r"\bthen\b",
)]
_WEAK_WORDS = frozenset("so then also ok okay instead continue next step steps".split())
_QUESTION_WORDS = frozenset("what how why when where which who whom whose".split())

# Pronouns that usually refer to something outside the message.
_REFERRING = frozenset("it its this that these those they them their theirs he him his she her".split())


@dataclass(frozen=True)
class FollowUp:
    is_follow_up: bool
    reason: str | None = None


STANDALONE = FollowUp(False)


def detect_follow_up(text: str, has_history: bool) -> FollowUp:
    """Decide whether ``text`` depends on earlier messages of its conversation.

    The first message of a conversation is always standalone. After that, a message
    is a follow-up when it
      * opens like a continuation ("and ...", "what about ..."),
      * points back explicitly ("the second one", "you said", "tell me more"),
      * leans on a pronoun with little else to go on ("how do I fix it"), or
      * has at most one content word ("why?", "pricing"), or
    * carries a weak signal (so, then, instead, continue, next step) and names
      nothing of its own ("so?", "what's the next step?").
    """
    if not has_history:
        return STANDALONE
    norm = normalize(text)
    words = _TOKEN.findall(norm)
    joined = " ".join(words)
    content = [w for w in words if w not in STOPWORDS and w not in _REFERRING]

    for opener in _OPENERS:
        if joined == opener or joined.startswith(opener + " "):
            return FollowUp(True, f'starts with "{opener}"')
    for pattern in _BACK_REFERENCES:
        m = pattern.search(joined)
        if m:
            return FollowUp(True, f'refers back ("{m.group(0)}")')
    # Words that name something: not question words, pronouns or the weak signals.
    own = [w for w in content if w not in _QUESTION_WORDS and w not in _WEAK_WORDS]
    if not own:
        for opener in _WEAK_OPENERS:
            if joined == opener or joined.startswith(opener + " "):
                return FollowUp(True, f'starts with "{opener}" and names nothing new')
        for pattern in _WEAK_SIGNALS:
            m = pattern.search(joined)
            if m:
                return FollowUp(True, f'"{m.group(0)}" with nothing new named')
    pronoun = next((w for w in words if w in _REFERRING), None)
    if pronoun and len(content) <= 2:
        return FollowUp(True, f'"{pronoun}" refers to something said earlier')
    if len(content) <= 1:
        return FollowUp(True, "too short to stand on its own")
    return STANDALONE


def build_history(messages: Iterable[dict], max_messages: int, max_chars: int) -> list[dict]:
    """The most recent messages that fit both limits, oldest first, as
    ``{"role", "content"}`` dicts ready for a chat API."""
    picked: list[dict] = []
    used = 0
    for m in reversed(list(messages)):
        if len(picked) >= max_messages:
            break
        content = m["content"]
        if picked and used + len(content) > max_chars:
            break
        if used + len(content) > max_chars:          # a single huge message: keep its end
            content = content[-max_chars:]
        picked.append({"role": m["role"], "content": content})
        used += len(content)
    picked.reverse()
    # Chat APIs expect the history to start with a user turn.
    while picked and picked[0]["role"] != "user":
        picked.pop(0)
    return picked
