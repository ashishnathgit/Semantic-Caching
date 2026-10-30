"""Turning raw text into terms, and terms into vocabulary indices.

Nothing here is linear algebra; it prepares the input for linalg.term_frequency_matrix.

Design choices worth knowing:
* Negations (not, no, never, without) are kept as terms, and contractions are
  expanded ("don't" -> "do not"), so negation is at least visible to the model.
* Question words (what, how, why, when, where, which, who) are kept too:
  "what is SVD" and "how to compute SVD" are different questions. So are
  on/off/up/down ("turn on" vs "turn off").
* A tiny plural folder maps "flights" -> "flight" and "laptops" -> "laptop".
  It is applied identically to the corpus and to queries, so it only needs to
  be consistent, not linguistically perfect.
* Bigrams ("low cost", "reset password") are added next to single words.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

STOPWORDS = frozenset(
    """
    a an the and or but if then so of to in at by for from with about into onto
    again further once
    i me my mine myself we us our ours you your yours he him his she her hers it its they them
    their theirs this that these those
    is am are was were be been being do does did doing have has had having
    will would shall should can could may might must
    there here than too very just also as such some any each every both either other
    own same only more most much many few lot lots get got please thanks thank hi hello
    """.split()
)
"""Common function words. Negations, question words and on/off/up/down are deliberately
absent: "turn on" and "turn off" must not look identical."""

_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'(s|re|ve|ll|d|m)\b"), ""),
]
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lower-case, unify quote characters and expand contractions."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def exact_key(text: str) -> str:
    """Canonical form for exact-match lookups: case, spacing and punctuation ignored."""
    return " ".join(_TOKEN.findall(normalize(text)))


def fold_plural(token: str) -> str:
    """Minimal plural folding: flights -> flight, batteries -> battery, boxes -> box.

    Singular and plural must land on the same term, so a few singular endings are
    folded too: "-ie" -> "-y" (cookie, cookies -> cooky) and "-che" -> "-ch"
    (cache, caches -> cach). The result is a matching key, not always a real word.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "ches", "shes", "xes", "zzes")):
        return token[:-2]
    if token.endswith("ie") and len(token) > 4:
        return token[:-2] + "y"
    if token.endswith("che"):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, stopwords: frozenset[str] = STOPWORDS, stem: bool = True) -> list[str]:
    """Split text into content words."""
    words = _TOKEN.findall(normalize(text))
    words = [w for w in words if w not in stopwords]
    return [fold_plural(w) for w in words] if stem else words


def analyze(text: str, ngram_max: int = 2, stopwords: frozenset[str] = STOPWORDS,
            stem: bool = True) -> list[str]:
    """Words plus n-grams up to ``ngram_max`` (1 = words only, 2 = add bigrams)."""
    words = tokenize(text, stopwords, stem)
    terms = list(words)
    for n in range(2, ngram_max + 1):
        terms.extend(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return terms


class Vocabulary:
    """Bidirectional map between terms and column indices of the TF-IDF matrix."""

    def __init__(self, terms: Sequence[str]):
        self.terms = list(terms)
        self.index = {term: i for i, term in enumerate(self.terms)}

    @classmethod
    def fit(cls, documents: Iterable[Sequence[str]], min_df: int = 1,
            max_df_ratio: float = 1.0, max_features: int | None = None,
            min_df_ngram: int | None = None) -> "Vocabulary":
        """Keep terms that appear in at least ``min_df`` documents (``min_df_ngram`` for
        multi-word terms) and in at most ``max_df_ratio`` of them, most frequent first,
        capped at ``max_features``. A bigram seen once is almost always noise, while a
        rare single word ("disable") can still carry the meaning."""
        df: Counter[str] = Counter()
        n_docs = 0
        for doc in documents:
            n_docs += 1
            df.update(set(doc))
        max_df = max_df_ratio * n_docs
        ngram_min = min_df if min_df_ngram is None else min_df_ngram
        kept = [(t, c) for t, c in df.items()
                if c >= (ngram_min if " " in t else min_df) and c <= max_df]
        kept.sort(key=lambda tc: (-tc[1], tc[0]))
        if max_features is not None:
            kept = kept[:max_features]
        return cls([t for t, _ in kept])

    def encode(self, terms: Iterable[str]) -> list[int]:
        """Column indices of known terms; unknown terms are dropped."""
        index = self.index
        return [index[t] for t in terms if t in index]

    def known(self, terms: Iterable[str]) -> list[str]:
        return [t for t in terms if t in self.index]

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, term: str) -> bool:
        return term in self.index
