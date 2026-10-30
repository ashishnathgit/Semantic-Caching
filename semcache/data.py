"""Data loading: the demo FAQ and Quora Question Pairs.

FAQ CSV columns
    id, topic, question, answer, paraphrases, heldout
``paraphrases`` and ``heldout`` hold several phrasings separated by " | ".
Paraphrases go into the training corpus (they teach the SVD which words mean
the same thing). Held-out phrasings are never trained on; they are used only to
evaluate, so the FAQ evaluation is honest.

Quora CSV columns (written by scripts/download_qqp.py, same as Kaggle's train.csv)
    question1, question2, is_duplicate        (a "label" column also works)
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FAQItem:
    id: str
    topic: str
    question: str
    answer: str
    paraphrases: list[str] = field(default_factory=list)
    heldout: list[str] = field(default_factory=list)


@dataclass
class QuestionPair:
    q1: str
    q2: str
    label: int          # 1 = same meaning (should be a cache hit), 0 = different


def _split(cell: str | None) -> list[str]:
    return [p.strip() for p in (cell or "").split("|") if p.strip()]


def load_faq(path: str | Path) -> list[FAQItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"id", "question", "answer"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"{path} needs the columns {sorted(required)}")
    return [FAQItem(r["id"], r.get("topic", ""), r["question"].strip(), r["answer"].strip(),
                    _split(r.get("paraphrases")), _split(r.get("heldout")))
            for r in rows if r.get("question")]


def faq_corpus(items: list[FAQItem]) -> list[str]:
    """Training text from the FAQ: questions and paraphrases, never the held-out ones."""
    corpus: list[str] = []
    for item in items:
        corpus.append(item.question)
        corpus.extend(item.paraphrases)
    return corpus


def faq_eval_pairs(items: list[FAQItem], negatives_per_item: int = 2, seed: int = 0) -> list[QuestionPair]:
    """Labelled pairs from held-out phrasings.

    Positives: (FAQ question, its held-out phrasing).
    Negatives: (FAQ question, a held-out phrasing of a different FAQ), half from the
    same topic (hard negatives), half from other topics.
    """
    rng = random.Random(seed)
    pairs: list[QuestionPair] = []
    by_topic: dict[str, list[FAQItem]] = {}
    for item in items:
        by_topic.setdefault(item.topic, []).append(item)
    for item in items:
        for h in item.heldout:
            pairs.append(QuestionPair(item.question, h, 1))
        same = [o for o in by_topic.get(item.topic, []) if o is not item and o.heldout]
        other = [o for o in items if o.topic != item.topic and o.heldout]
        for i in range(negatives_per_item):
            pool = same if (i % 2 == 0 and same) else other
            if pool:
                partner = rng.choice(pool)
                pairs.append(QuestionPair(item.question, rng.choice(partner.heldout), 0))
    return pairs


def load_question_pairs(path: str | Path, max_pairs: int | None = None, seed: int = 0) -> list[QuestionPair]:
    """Read Quora-style labelled pairs from .csv or .tsv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_qqp.py` or see the README.")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    pairs: list[QuestionPair] = []
    csv.field_size_limit(10_000_000)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        label_key = "is_duplicate" if "is_duplicate" in (reader.fieldnames or []) else "label"
        for row in reader:
            q1, q2, label = row.get("question1"), row.get("question2"), row.get(label_key)
            if not q1 or not q2 or label in (None, "", "-1"):
                continue
            pairs.append(QuestionPair(q1.strip(), q2.strip(), int(float(label))))
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    return pairs


def unique_questions(pairs: list[QuestionPair], max_questions: int | None = None, seed: int = 0) -> list[str]:
    seen: dict[str, None] = {}
    for p in pairs:
        seen.setdefault(p.q1, None)
        seen.setdefault(p.q2, None)
    questions = list(seen)
    if max_questions is not None and len(questions) > max_questions:
        questions = random.Random(seed).sample(questions, max_questions)
    return questions
