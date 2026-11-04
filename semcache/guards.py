"""The negation guard: text rules checked before a semantic hit is served.

LSA puts "turn on notifications" and "turn off notifications" at cosine 0.93 (FAQ model):
the two questions share every content word. These rules compare the question with the
matched cached question and refuse the hit when they disagree on

    negation   the number of {not, no, never, cannot, nor, without} differs in parity
               (contractions are expanded first: "isn't" -> "is not", "can't" -> "can not")
    antonym    both use a word of an antonym pair, but not the same words ("on" vs "off")
    number     each side has a number the other lacks ("top 5" vs "top 10")
    name       each side has a capitalised name the other lacks ("Goa" vs "Mumbai");
               a text in Title Case gives no names (every word would look like one), and
               ALL-CAPS acronyms don't count ("AI" vs "LLM" are near-synonyms)

An exact repeat is never checked (it is the same question). Nothing here is linear algebra.
Each check is O(words) with set lookups.
"""

from __future__ import annotations

import re

from .text import STOPWORDS, _TOKEN, fold_plural, normalize

NEGATIONS = frozenset({"not", "no", "never", "cannot", "nor", "without"})
_RAW_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?")
_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
                 "hundred": "100", "thousand": "1000"}
_NOT_NAMES = STOPWORDS | {"what", "how", "why", "when", "where", "which", "who", "whom", "whose", "is", "i",
                          "can", "do", "does", "should", "would", "will"}


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _words(text: str) -> list[str]:
    """Normalised words, with CamelCase split first: "ModuleNotFoundError" -> module not found error,
    so an error name counts its negation the way "module not found" does."""
    return _TOKEN.findall(normalize(_CAMEL.sub(" ", text)))


def negation_count(text: str) -> int:
    return sum(w in NEGATIONS for w in _words(text))


def numbers(text: str) -> set[str]:
    found = set(re.findall(r"\d+(?:\.\d+)?", normalize(text)))
    found |= {_NUMBER_WORDS[w] for w in _words(text) if w in _NUMBER_WORDS}
    return found


def names(text: str) -> set[str]:
    """Capitalised words that are not the first word of a sentence (lower-cased). ALL-CAPS
    acronyms are left out: "AI" and "LLM", or "RAM", name concepts, and two different
    acronyms often mean nearly the same thing."""
    tokens = [(m.group(0), m.start()) for m in _RAW_WORD.finditer(text)]
    words = [t for t, _ in tokens if not _NUMBER.match(t)]
    if not words:
        return set()
    # Title Case ("Is Mac Or Windows Better") makes every word look like a name: no signal.
    # Acronyms (SVD, USA) don't count towards it, and neither does the first word.
    rest = words[1:]
    titled = [w for w in rest if w[0].isupper() and not w.isupper()]
    if len(rest) >= 3 and len(titled) > len(rest) / 2:
        return set()
    out = set()
    for i, (tok, start) in enumerate(tokens):
        if _NUMBER.match(tok) or not tok[0].isupper():
            continue
        before = text[:start].rstrip()
        sentence_start = i == 0 or before.endswith((".", "?", "!", ":", ";", "\n"))
        low = tok.lower()
        if sentence_start or tok.isupper() or low in _NOT_NAMES:
            continue
        out.add(low)
    return out


def parse_antonyms(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        parts = [x.strip().lower() for x in re.split(r"[/,|]", p) if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"an antonym pair looks like 'on/off', not {p!r}")
        out.append((fold_plural(parts[0]), fold_plural(parts[1])))
    return out


class TextGuard:
    def __init__(self, antonyms: list[str] | None = None, check_numbers: bool = True, check_names: bool = True):
        self.pairs = parse_antonyms(antonyms or [])
        self.check_numbers, self.check_names = check_numbers, check_names

    @classmethod
    def from_settings(cls, cache_settings) -> "TextGuard":
        return cls(list(getattr(cache_settings, "antonyms", []) or []))

    def check(self, query: str, cached: str) -> dict | None:
        """None when the hit may be served; else {"rule", "detail"} for the first rule that fires."""
        nq, nc = negation_count(query), negation_count(cached)
        if nq % 2 != nc % 2:
            return {"rule": "negation",
                    "detail": f"your question has {nq} negation word{'s' if nq != 1 else ''} and the cached "
                              f"question {nc}, so one says the opposite of the other"}
        wq = {fold_plural(w) for w in _words(query)}
        wc = {fold_plural(w) for w in _words(cached)}
        for a, b in self.pairs:
            sq, sc = {a, b} & wq, {a, b} & wc
            if sq and sc and sq != sc:
                return {"rule": "antonym",
                        "detail": f'your question says "{"/".join(sorted(sq))}", the cached question '
                                  f'"{"/".join(sorted(sc))}" ({a}/{b} are opposites)'}
        if self.check_numbers:
            xq, xc = numbers(query), numbers(cached)
            if xq - xc and xc - xq:
                return {"rule": "number",
                        "detail": f"the numbers differ: {', '.join(sorted(xq))} vs {', '.join(sorted(xc))}"}
        if self.check_names:
            pq = {fold_plural(n) for n in names(query)}
            pc = {fold_plural(n) for n in names(cached)}
            if pq - wc and pc - wq:
                return {"rule": "name",
                        "detail": f"the names differ: {', '.join(sorted(pq - wc))} vs {', '.join(sorted(pc - wq))}"}
        return None
