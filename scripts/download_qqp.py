"""Get Quora Question Pairs and save it as CSV for the semantic cache.

Option A - Hugging Face (no account needed):
    pip install datasets
    python scripts/download_qqp.py
    -> data/quora/train.csv       (GLUE QQP train split, ~364k labelled pairs)
       data/quora/validation.csv  (GLUE QQP validation split, ~40k labelled pairs)

Option B - Kaggle's train.csv (from the "Quora Question Pairs" competition page):
    python scripts/download_qqp.py --from-kaggle path/to/train.csv
    -> the same two files, split 90/10 at random

Both write the columns question1, question2, is_duplicate. The GLUE test split has no
labels, so it is not downloaded.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "quora"


def write_pairs(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question1", "question2", "is_duplicate"])
        for q1, q2, label in rows:
            if q1 and q2 and label in (0, 1):
                writer.writerow([q1.strip(), q2.strip(), label])
                count += 1
    return count


def from_huggingface() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("The `datasets` package is needed: pip install datasets")
    print("Downloading GLUE / QQP from Hugging Face (about 40 MB)...")
    ds = load_dataset("nyu-mll/glue", "qqp")
    for split, name in (("train", "train.csv"), ("validation", "validation.csv")):
        part = ds[split]
        missing = {"question1", "question2", "label"} - set(part.column_names)
        if missing:
            sys.exit(f"Unexpected columns in the {split} split: missing {sorted(missing)}")
        n = write_pairs(OUT_DIR / name, zip(part["question1"], part["question2"], part["label"]))
        print(f"  {split}: {n:,} pairs -> {OUT_DIR / name}")


def from_kaggle(path: Path, eval_share: float, seed: int) -> None:
    if not path.exists():
        sys.exit(f"{path} not found")
    csv.field_size_limit(10_000_000)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [(r["question1"], r["question2"], int(r["is_duplicate"]))
                for r in csv.DictReader(fh) if r.get("question1") and r.get("question2")]
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * (1 - eval_share))
    n_train = write_pairs(OUT_DIR / "train.csv", rows[:cut])
    n_eval = write_pairs(OUT_DIR / "validation.csv", rows[cut:])
    print(f"train: {n_train:,} pairs, validation: {n_eval:,} pairs -> {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-kaggle", type=Path, help="path to Kaggle's train.csv")
    parser.add_argument("--eval-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.from_kaggle:
        from_kaggle(args.from_kaggle, args.eval_share, args.seed)
    else:
        from_huggingface()
    print("Next: python -m semcache build   (refits the SVD on Quora + FAQ)")


if __name__ == "__main__":
    main()
