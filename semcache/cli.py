"""Command-line interface:  python -m semcache <command>

Every command takes --profile faq|quora (default: [profiles] default in config.toml,
or the SEMCACHE_PROFILE environment variable).

    build       fit the TF-IDF + SVD model and create a warm cache
    ask         ask one question
    chat        interactive session (type :help inside); --user keeps a conversation
    stats       cache statistics
    entries     list cached entries
    threads     list a user's conversations
    users       list the users who have signed in
    reset       empty the cache and preload the FAQ again
    warm        put back missing FAQ answers without touching other entries
    reset-demo  exactly the FAQ answers, metrics archived, every user's counters zeroed
    preflight   PASS/FAIL checks to run right before going on stage
    evaluate    precision/recall, rank sweep, latency report
    measure     headline numbers, traffic mix, tau cross-validation, LSH sweep, fact checks
    backend     check whether Ollama is reachable
    serve       start the FastAPI server (for the React UI)
    gradio      start the Gradio demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import BACKEND_ENV, PROFILE_ENV, Settings
from .users import ThreadNotFound


def threshold_arg(text: str) -> float:
    """A threshold tau in (0, 1]; tau <= 0 would turn every query into a hit."""
    try:
        tau = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 < tau <= 1.0:
        raise argparse.ArgumentTypeError(f"threshold must be in (0, 1], got {tau}")
    return tau


def _bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, round(score * width)))
    return "#" * filled + "." * (width - filled)


def _print_result(r, show_matches: bool = True) -> None:
    status = "HIT " if r.hit else "FOLLOW-UP" if r.follow_up else "MISS"
    print(f"\n[{status}] cos = {r.score:.3f}  (tau = {r.threshold:.2f}, {r.lookup_method})")
    if r.hit:
        print(f"  matched: {r.matched_query!r}  [{r.matched_source}]")
    print(f"\n{r.answer}\n")
    if show_matches and r.matches:
        print("  closest cached queries:")
        for m in r.matches:
            print(f"    {m['score']:.3f} {_bar(m['score'])} {m['query'][:70]}")
    t = r.timings_ms
    print(f"  time: embed {t['embed']:.2f} ms, lookup {t['lookup']:.2f} ms, "
          f"LLM {t['llm']:.0f} ms, total {t['total']:.1f} ms")
    if r.backend:
        print(f"  answered by: {r.backend}{' / ' + r.model if r.model else ''}")
    if r.note:
        print(f"  note: {r.note}")
    if r.backend_error:
        print(f"  Ollama unavailable, so the placeholder answered: {r.backend_error}")


def _settings(args) -> Settings:
    return Settings.load(args.config, getattr(args, "profile", None))


def _service(args):
    from .pipeline import SemanticCacheService
    return SemanticCacheService.from_settings(_settings(args))


def cmd_build(args) -> int:
    from .pipeline import build_artifacts
    settings = _settings(args)
    if args.no_quora:
        settings.corpus.use_quora = False
    build_artifacts(settings, verbose=True, fresh=args.fresh)
    return 0


def cmd_ask(args) -> int:
    service = _service(args)
    # The CLI runs on the server's own disk, so it is trusted to see every question.
    result = service.ask(" ".join(args.query), threshold=args.threshold, top_k=args.top_k,
                         use_llm=not args.no_llm, user=args.user, thread_id=args.thread, admin=True)
    service.save()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_result(result)
        if result.thread_id:
            print(f"  conversation: {result.thread_id} (continue it with --thread {result.thread_id})")
    return 0


def _print_threads(service, user: str, current: str | None = None) -> None:
    threads = service.users.list_threads(user)
    if not threads:
        print("  no conversations yet")
    for t in threads:
        mark = "*" if t.id == current else " "
        print(f" {mark} {t.id}  {len(t.messages):>3} msgs  {t.title[:60]}")


def cmd_chat(args) -> int:
    service = _service(args)
    tau = args.threshold
    user = args.user
    thread_id = args.thread
    if thread_id and not user:
        raise ValueError("--thread needs --user")
    if user:
        profile = service.profile(user)
        user = profile["name"]
        if thread_id:
            thread = service.users.get_thread(user, thread_id)
            print(f"Continuing {thread.title!r} ({len(thread.messages)} messages).")
        print(f"Semantic cache chat as {user}. Follow-ups use this conversation's history. "
              "Type :help for commands.")
    else:
        print("Semantic cache chat. Type a question, or :help for commands. "
              "(Start with --user NAME to keep a conversation.)")
    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.startswith(":"):
                cmd, _, rest = line[1:].partition(" ")
                if cmd in {"q", "quit", "exit"}:
                    break
                elif cmd == "help":
                    print(":tau 0.75   set the threshold   :stats   cache statistics\n"
                          ":entries    recent entries      :reset   reload the FAQ only\n"
                          ":save       save the cache      :quit    leave")
                    if user:
                        print(":new        new conversation    :threads list conversations\n"
                              ":open ID    switch to one       :history this conversation")
                elif user and cmd == "new":
                    thread_id = None
                    print("new conversation (it starts with your next question)")
                elif user and cmd == "threads":
                    _print_threads(service, user, thread_id)
                elif user and cmd == "open":
                    try:
                        thread = service.users.get_thread(user, rest.strip())
                    except ThreadNotFound:
                        print("no such conversation; :threads lists them")
                    else:
                        thread_id = thread.id
                        print(f"switched to {thread.title!r} ({len(thread.messages)} messages)")
                elif user and cmd == "history":
                    if not thread_id:
                        print("  (empty)")
                    else:
                        for m in service.users.get_thread(user, thread_id).messages:
                            who = "you" if m.role == "user" else "bot"
                            print(f"  {who}> {m.content[:200]}")
                elif cmd == "tau":
                    try:
                        tau = threshold_arg(rest.strip())
                        print(f"threshold set to {tau:.2f}")
                    except argparse.ArgumentTypeError as exc:
                        print(f"{exc}; usage: :tau 0.75")
                elif cmd == "stats":
                    print(json.dumps(service.stats(user), indent=2))
                elif cmd == "entries":
                    for e in service.visible_entries(user, 15, everyone=not user):
                        print(f"  #{e.id:<4} [{e.source:<6}] hits {e.hits:<3} {_owner(e)}{e.query[:70]}")
                elif cmd == "reset":
                    if not user:
                        print(f"cache reset with {service.reset()} FAQ entries")
                    else:
                        r = service.reset_for(user, admin=True)       # the CLI is trusted
                        print(f"cache reset with {r['faq_entries']} FAQ entries" if r["scope"] == "all"
                              else f"removed {r['removed']} private answers and reset your counters")
                elif cmd == "save":
                    service.save()
                    print("saved")
                else:
                    print("unknown command; try :help")
                continue
            try:
                result = service.ask(line, threshold=tau, user=user, thread_id=thread_id, admin=True)
                _print_result(result)
                thread_id = result.thread_id or thread_id
            except ValueError as exc:
                print(f"error: {exc}")
    except KeyboardInterrupt:
        print()
    finally:
        service.save()
    return 0


def cmd_stats(args) -> int:
    print(json.dumps(_service(args).describe(), indent=2, ensure_ascii=False))
    return 0


def _owner(entry) -> str:
    return f"(private: {entry.owner}) " if entry.owner else ""


def cmd_entries(args) -> int:
    service = _service(args)
    for e in service.visible_entries(args.user, args.limit, everyone=not args.user):
        print(f"#{e.id:<5} [{e.source:<6}] hits {e.hits:<4} {_owner(e)}{e.query[:80]}")
    return 0


def cmd_threads(args) -> int:
    _print_threads(_service(args), args.user)
    return 0


def cmd_users(args) -> int:
    from .pipeline import new_user_store
    settings = _settings(args)
    store = new_user_store(settings)
    names = store.list_users()
    if not names:
        print("nobody has signed in yet")
    for name in names:
        st = store.profile(name).stats()
        rate = f"{100 * st['hit_rate']:.0f}% hits" if st["hit_rate"] is not None else "no lookups"
        print(f"{name:<20} {st['questions']:>4} questions, {rate}, "
              f"{len(store.list_threads(name))} conversations")
    return 0


def cmd_reset(args) -> int:
    print(f"cache reset with {_service(args).reset()} FAQ entries")
    return 0


def cmd_reset_demo(args) -> int:
    r = _service(args).reset_demo(admin=True)          # the CLI is trusted, like warm
    print(f"demo reset: {r['entries']} FAQ answers in the cache, {r['removed_events']} metric events "
          f"archived{' to ' + r['archived_to'] if r['archived_to'] else ''}, "
          f"counters zeroed for {r['users_reset']} users")
    return 0


def cmd_preflight(args) -> int:
    from .preflight import run_preflight
    return 0 if run_preflight(_settings(args)) else 1


def cmd_warm(args) -> int:
    r = _service(args).warm()
    print(f"added {r['added']} FAQ answers ({r['already_there']} were already there); "
          f"the cache now has {r['entries']} entries")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import guard_sweep, run_evaluation
    if args.neural:
        from .neural import available, neural_comparison, neural_markdown
        if not available():
            print("error: the neural baseline needs: pip install -r requirements-neural.txt", file=sys.stderr)
            return 1
        settings = _settings(args)
        result = neural_comparison(settings)
        settings.eval_dir.mkdir(parents=True, exist_ok=True)
        (settings.eval_dir / "neural_baseline.md").write_text(neural_markdown(result), encoding="utf-8")
        print(f"  report: {settings.eval_dir / 'neural_baseline.md'}")
        return 0
    if args.guard_sweep:
        guard_sweep(_settings(args), max_pairs=args.max_pairs)
        return 0
    run_evaluation(_settings(args), pairs_path=args.pairs, max_pairs=args.max_pairs,
                   charts=not args.no_charts)
    return 0


def cmd_measure(args) -> int:
    from .measure import run_measurements
    only = [x.strip() for x in args.only.split(",")] if args.only else None
    run_measurements(_settings(args), only=only)
    return 0


def cmd_backend(args) -> int:
    from .backends import build_backend
    info = build_backend(_settings(args).llm).describe()
    print(json.dumps(info, indent=2))
    if not info.get("available"):
        print("\nOllama isn't reachable. Install it from https://ollama.com, run `ollama serve`, "
              "then `ollama pull llama3.2:3b` (or any model). Until then the stub answers misses.")
    return 0


def _export_settings(args) -> Settings:
    """Check the profile, then pass it (and --config) to code that loads its own settings."""
    if getattr(args, "backend", None):
        os.environ[BACKEND_ENV] = args.backend
    settings = _settings(args)
    os.environ[PROFILE_ENV] = settings.profile
    if args.config:
        os.environ["SEMCACHE_CONFIG"] = os.path.abspath(args.config)
    return settings


def cmd_serve(args) -> int:
    import uvicorn
    settings = _export_settings(args)
    print(f"Serving profile {settings.profile!r} from {settings.artifacts_dir} "
          f"(LLM backend: {settings.llm.backend})")
    uvicorn.run("apps.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_gradio(args) -> int:
    _export_settings(args)
    from apps.gradio_app import launch
    launch(port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semcache", description="Semantic cache for NLP queries")
    parser.add_argument("--config", help="path to config.toml (default: project root)")
    parser.add_argument("--profile", help="model profile, e.g. faq or quora (see config.toml)")
    # --profile may also come after the command:  semcache build --profile quora
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", default=argparse.SUPPRESS,
                        help="model profile, e.g. faq or quora (see config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, **kw) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], **kw)

    p = add("build", help="fit the model and create a warm cache")
    p.add_argument("--no-quora", action="store_true", help="fit on the FAQ only")
    p.add_argument("--fresh", action="store_true", help="discard cached answers; keep only the FAQ")
    p.set_defaults(func=cmd_build)

    p = add("ask", help="ask one question")
    p.add_argument("query", nargs="+")
    p.add_argument("--threshold", type=threshold_arg)
    p.add_argument("--top-k", type=int)
    p.add_argument("--no-llm", action="store_true", help="only look up; never call the LLM")
    p.add_argument("--json", action="store_true")
    p.add_argument("--user", help="ask as this user (keeps the question in a conversation)")
    p.add_argument("--thread", help="continue this conversation (needs --user)")
    p.set_defaults(func=cmd_ask)

    p = add("chat", help="interactive session")
    p.add_argument("--threshold", type=threshold_arg)
    p.add_argument("--user", help="chat as this user; follow-ups use the conversation history")
    p.add_argument("--thread", help="continue this conversation (needs --user)")
    p.set_defaults(func=cmd_chat)

    add("stats", help="cache statistics").set_defaults(func=cmd_stats)
    p = add("entries", help="list cached entries")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--user", help="only what this user can see (shared + their private entries)")
    p.set_defaults(func=cmd_entries)
    p = add("threads", help="list a user's conversations")
    p.add_argument("--user", required=True)
    p.set_defaults(func=cmd_threads)
    add("users", help="list users").set_defaults(func=cmd_users)
    add("reset", help="empty the cache and preload the FAQ").set_defaults(func=cmd_reset)
    add("reset-demo", help="exactly the FAQ answers, metrics archived, user counters zeroed").set_defaults(
        func=cmd_reset_demo)
    add("preflight", help="PASS/FAIL checks before going on stage (nothing is stored)").set_defaults(
        func=cmd_preflight)
    add("warm", help="put back missing FAQ answers; other entries are kept").set_defaults(
        func=cmd_warm)

    p = add("evaluate", help="evaluation report with charts")
    p.add_argument("--pairs", help="CSV/TSV of question pairs (default: Quora validation file)")
    p.add_argument("--max-pairs", type=int)
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--neural", action="store_true",
                   help="compare LSA with a MiniLM baseline (needs requirements-neural.txt)")
    p.add_argument("--guard-sweep", action="store_true",
                   help="refit for latent stopwords on/off x k = 200/300/400 x rho cutoffs (slow)")
    p.set_defaults(func=cmd_evaluate)

    p = add("measure", help="headline numbers, traffic mix, tau cross-validation, LSH sweep, fact checks")
    p.add_argument("--only", help="comma-separated: headline,traffic,tau_cv,lsh,facts,latent,incremental,thresholds,neural,guard")
    p.set_defaults(func=cmd_measure)

    add("backend", help="check the LLM backend").set_defaults(func=cmd_backend)

    p = add("serve", help="start the FastAPI server")
    p.add_argument("--backend", choices=["ollama", "openai_compatible", "stub"],
                   help="override [llm] backend; stub = placeholder answers, cached (the stage fallback)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = add("gradio", help="start the Gradio demo")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public link")
    p.set_defaults(func=cmd_gradio)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ThreadNotFound as exc:
        print(f"error: no conversation {exc.args[0]!r}; `semcache threads --user NAME` lists them",
              file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
