import shutil
from pathlib import Path

import pytest

from semcache.config import PROJECT_ROOT, Settings


@pytest.fixture()
def settings(tmp_path, monkeypatch) -> Settings:
    """Project settings with artifacts in a temp folder, the FAQ-only corpus and the stub LLM."""
    monkeypatch.setenv("SEMCACHE_ARTIFACTS", str(tmp_path / "artifacts"))
    s = Settings.load(PROJECT_ROOT / "config.toml")
    s.corpus.use_quora = False
    s.llm.backend = "stub"
    s.cache.autosave_every = 1
    return s


@pytest.fixture()
def faq_path() -> Path:
    return PROJECT_ROOT / "data" / "faq.csv"
