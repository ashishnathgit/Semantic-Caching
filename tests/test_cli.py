import argparse
import json

import pytest

from semcache.cli import main, threshold_arg
from semcache.pipeline import build_artifacts


def test_threshold_arg_accepts_zero_to_one():
    assert threshold_arg("0.75") == 0.75 and threshold_arg("1") == 1.0


@pytest.mark.parametrize("text", ["0", "-1", "1.5", "abc"])
def test_threshold_arg_rejects_bad_values(text):
    with pytest.raises(argparse.ArgumentTypeError):
        threshold_arg(text)


def test_ask_with_zero_threshold_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["ask", "hello", "--threshold", "0"])
    assert exc.value.code == 2 and "threshold must be in (0, 1]" in capsys.readouterr().err


@pytest.fixture()
def cli_config(settings, tmp_path):
    """A config.toml for the CLI that uses the FAQ only and the stub LLM."""
    build_artifacts(settings, verbose=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[corpus]\nuse_quora = false\n[llm]\nbackend = "stub"\n[paths]\n'
                   f'faq_csv = "{settings.path(settings.paths.faq_csv).as_posix()}"\n', encoding="utf-8")
    return str(cfg)


def test_ask_as_a_user_continues_a_conversation(cli_config, capsys):
    assert main(["--config", cli_config, "ask", "how many moons does jupiter have", "--user", "alice", "--json"]) == 0
    tid = json.loads(capsys.readouterr().out)["thread_id"]
    assert main(["--config", cli_config, "ask", "and", "saturn?", "--user", "alice", "--thread", tid, "--json"]) == 0
    follow = json.loads(capsys.readouterr().out)
    assert follow["follow_up"] and follow["thread_id"] == tid and follow["history_used"] == 2
    assert main(["--config", cli_config, "threads", "--user", "alice"]) == 0
    assert tid in capsys.readouterr().out
    assert main(["--config", cli_config, "users"]) == 0
    assert "alice" in capsys.readouterr().out


def test_unknown_thread_is_a_clean_error(cli_config, capsys):
    assert main(["--config", cli_config, "ask", "hello there", "--user", "alice", "--thread", "0123456789ab"]) == 1
    assert "no conversation" in capsys.readouterr().err


def test_chat_with_user_commands(cli_config, capsys, monkeypatch):
    lines = iter(["what is the capital of france", "and germany?", ":history", ":threads", ":new", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(lines))
    assert main(["--config", cli_config, "chat", "--user", "alice"]) == 0
    out = capsys.readouterr().out
    assert "[FOLLOW-UP]" in out and "you> what is the capital of france" in out
    assert "new conversation" in out


def test_warm_restores_missing_faq_answers_and_keeps_the_rest(cli_config, capsys):
    from semcache.config import Settings
    from semcache.pipeline import SemanticCacheService

    settings = Settings.load(cli_config)
    service = SemanticCacheService.from_settings(settings, verbose=False)
    faq = [e for e in service.cache.entries() if e.source == "faq"]
    for e in faq[:40]:
        service.cache.remove(e.id)
    kept = service.add_entry("where is the exam hall", "Room 204")          # not an FAQ entry
    service.save()

    assert main(["--config", cli_config, "warm", "--profile", settings.profile]) == 0
    assert "added 40 FAQ answers (3 were already there)" in capsys.readouterr().out
    after = SemanticCacheService.from_settings(Settings.load(cli_config), verbose=False)
    assert sum(e.source == "faq" for e in after.cache.entries()) == len(faq)
    assert after.cache.get(kept.id) is not None
    assert main(["--config", cli_config, "warm"]) == 0                      # idempotent
    assert "added 0 FAQ answers" in capsys.readouterr().out
