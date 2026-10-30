"""Make a clean zip of the project for teammates.

    python scripts/package_zip.py                 -> ../semantic-cache-YYYYMMDD.zip
    python scripts/package_zip.py --out ~/Desktop/semantic-cache.zip

Leaves out what each teammate recreates or should never share: .venv, node_modules,
data/quora (download with scripts/download_qqp.py), artifacts (python -m semcache build),
caches (__pycache__, .pytest_cache, __MACOSX, .DS_Store), .env files, and the review
screenshots in docs/screenshots except the six-stage GIF. frontend/dist is kept, so the
UI runs with Python alone. Everything sits under one top-level folder, semantic-cache/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TOP = "semantic-cache"

EXCLUDED_DIRS = {".venv", "venv", "node_modules", "artifacts", "__pycache__", ".pytest_cache", "__MACOSX",
                 ".ipynb_checkpoints", ".git", ".idea", ".vscode", ".mypy_cache", ".ruff_cache"}
EXCLUDED_PATHS = {"data/quora"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
KEEP_IN_SCREENSHOTS = {"docs/screenshots/six-stages.gif"}


def excluded(rel: PurePosixPath) -> bool:
    parts = rel.parts
    if any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in parts):
        return True
    text = rel.as_posix()
    if any(text == p or text.startswith(p + "/") for p in EXCLUDED_PATHS):
        return True
    if rel.name in EXCLUDED_NAMES or rel.suffix in EXCLUDED_SUFFIXES:
        return True
    if rel.name == ".env" or rel.name.startswith(".env."):
        return True
    if text.startswith("docs/screenshots/") and text not in KEEP_IN_SCREENSHOTS:
        return True
    return False


def files_to_pack(root: Path = ROOT) -> list[Path]:
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if not excluded(rel):
                out.append(path)
    return out


def package(out: Path, root: Path = ROOT) -> tuple[int, int]:
    files = files_to_pack(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, f"{TOP}/{path.relative_to(root).as_posix()}")
    return len(files), out.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT.parent / f"{TOP}-{dt.date.today():%Y%m%d}.zip")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if ROOT in out.parents:
        raise SystemExit("write the zip outside the project folder, or it would include itself next time")
    n, size = package(out)
    print(f"{out}: {n} files, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
