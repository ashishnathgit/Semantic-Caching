"""Gradio demo of the semantic cache, for several people at once.

    python -m semcache gradio            (or: python -m apps.gradio_app)
    python -m semcache gradio --share    (temporary public link, handy in Colab)

Each browser tab signs in with a name (no password) and keeps its own
conversations. Typing updates the angle picture live (a lookup only; nothing is
stored). "Ask" runs the full pipeline: a hit returns the shared cached answer, a
miss asks the LLM and stores the answer for everyone, and a follow-up ("what
about the second one?") goes to the LLM with the conversation so far.
"""

from __future__ import annotations

import html
import math

import gradio as gr

from semcache.config import Settings
from semcache.pipeline import SemanticCacheService
from semcache.users import ThreadNotFound
from semcache.viz import angle_fan_svg, spectrum_svg

EXAMPLES = [
    "i forgot my password, how can i reset it",
    "my laptop is really slow",
    "cheap flight tickets to goa",
    "who won the cricket world cup",
]
SOURCES = {"faq": "FAQ", "ollama": "Ollama", "stub": "Placeholder", "manual": "Added by hand"}
CSS = """
.verdict h3 { margin: 0 0 6px; font-size: 20px; }
.verdict .hit { color: #2f6f5e; } .verdict .miss { color: #8a5a12; } .verdict .follow { color: #3a4fd8; }
.verdict blockquote { margin: 8px 0; padding-left: 12px; border-left: 2px solid #d8dce2; }
.verdict .meta { font-size: 13px; opacity: .8; }
"""
INTRO = ("<p>Ask something. A paraphrase of a cached question is answered from the shared cache; "
         "anything new goes to the model and is remembered for everyone. Follow-ups use your "
         "conversation instead.</p>")

_service: SemanticCacheService | None = None


def service() -> SemanticCacheService:
    global _service
    if _service is None:
        _service = SemanticCacheService.from_settings(Settings.load())
    return _service


def _angle(score: float) -> float:
    return math.degrees(math.acos(min(1.0, max(0.0, score))))


def _match_rows(matches: list[dict]) -> list[list]:
    return [[f"c{i + 1}", m["query"], round(m["score"], 3), round(_angle(m["score"]), 1),
             SOURCES.get(m["source"], m["source"]), "private" if m.get("owner") else "shared"]
            for i, m in enumerate(matches)]


def _status(user: str | None) -> str:
    svc = service()
    s, b = svc.stats(user), svc.backend.describe()
    rate = f", {round(100 * s['hit_rate'])}% hits overall" if s["hit_rate"] is not None else ""
    llm = f"Ollama ({b.get('model')})" if b.get("available") else "placeholder answers (Ollama isn't running)"
    mine = ""
    if user:
        u = s["user"]
        mine_rate = f", {round(100 * u['hit_rate'])}% from the cache" if u["hit_rate"] is not None else ""
        mine = f" **You:** {u['questions']} questions{mine_rate}, {u['follow_ups']} follow-ups."
    return (f"**Model:** k = {svc.embedder.dim}, {len(svc.embedder.vocab):,} terms. "
            f"**Cache:** {s['entries']} answers{rate}. **On a miss:** {llm}.{mine}")


def _entries(user: str | None) -> list[list]:
    return [[e.id, e.query, SOURCES.get(e.source, e.source), "private" if e.owner else "shared", e.hits]
            for e in service().listed_entries(user)]


def _chat(user: str | None, thread_id: str | None) -> list[dict]:
    if not user or not thread_id:
        return []
    try:
        thread = service().users.get_thread(user, thread_id)
    except ThreadNotFound:
        return []
    return [{"role": m.role, "content": m.content} for m in thread.messages]


def _threads(user: str | None, selected: str | None):
    choices = [(t.title, t.id) for t in service().users.list_threads(user)] if user else []
    return gr.update(choices=choices, value=selected if selected in {c[1] for c in choices} else None)


def _require(user: str | None) -> str:
    if not user:
        raise gr.Error("Sign in with a name first.")
    return user


# ----------------------------------------------------------------- handlers
def sign_in(name: str):
    try:
        profile = service().profile(name or "")
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    user = profile["name"]
    return (user, None, f"Signed in as **{user}**.", _threads(user, None), [], _status(user),
            _entries(user), INTRO)


def preview(query: str, tau: float, user: str | None, thread_id: str | None):
    """Live lookup while typing: scores only, nothing stored."""
    if not query or not query.strip():
        return angle_fan_svg([], tau), [], ""
    svc = service()
    result, _ = svc.lookup(query, threshold=tau, record=False, user=user)
    hint = ""
    if user and thread_id:
        try:
            cache_used, follow = svc.follow_up_for(query, user, thread_id)
        except ThreadNotFound:
            cache_used, follow = True, None
        if not cache_used and follow and follow.is_follow_up:
            hint = f"Looks like a follow-up ({follow.reason}): it will skip the cache and use this conversation."
    matches = svc.match_dicts(result, user)
    return angle_fan_svg(matches, tau, result.method == "exact"), _match_rows(matches), hint


def ask(query: str, tau: float, user: str | None, thread_id: str | None):
    user = _require(user)
    if not query or not query.strip():
        raise gr.Error("Type a question first.")
    try:
        r = service().ask(query, threshold=tau, user=user, thread_id=thread_id)
    except ThreadNotFound:
        raise gr.Error("That conversation no longer exists; start a new one.") from None
    t = r.timings_ms
    if r.hit:
        how = "Exact repeat of" if r.lookup_method == "exact" else "Same meaning as"
        detail = (f"{how} <q>{html.escape(r.matched_query or '')}</q> "
                  f"(cos θ = {r.score:.3f}, θ = {_angle(r.score):.1f}°). "
                  f"Found in {t['embed'] + t['lookup']:.2f} ms with no model call.")
        title = '<h3 class="hit">Answered from the cache</h3>'
    elif not r.cache_used:
        title = '<h3 class="follow">Follow-up, answered with the conversation</h3>'
        detail = ""                                       # the note below explains it
    else:
        if r.source == "ollama":
            detail = f"Answered by Ollama ({html.escape(r.model or '')}) in {t['llm']:.0f} ms, now cached."
        else:
            detail = "Placeholder answer, because Ollama isn't running. It's cached so the next ask is a hit."
        title = '<h3 class="miss">New question</h3>'
    extra = "".join(f'<p class="meta">{html.escape(x)}</p>' for x in (r.note, r.backend_error) if x)
    verdict = (f'<div class="verdict">{title}<p class="meta">at τ = {r.threshold:.2f}</p>'
               f'<blockquote>{html.escape(r.answer)}</blockquote><p class="meta">{detail}</p>{extra}</div>')
    fan = angle_fan_svg(r.matches, tau, r.lookup_method == "exact")
    return (verdict, fan, _match_rows(r.matches), _status(user), _entries(user),
            _chat(user, r.thread_id), r.thread_id, _threads(user, r.thread_id), "")


def open_thread(thread_id: str | None, user: str | None):
    return _chat(user, thread_id), thread_id


def new_thread(user: str | None):
    _require(user)
    return [], None, gr.update(value=None), INTRO


def delete_thread(thread_id: str | None, user: str | None):
    user = _require(user)
    if thread_id:
        service().users.delete_thread(user, thread_id)
    return [], None, _threads(user, None)


def add_entry(question: str, answer: str, user: str | None):
    user = _require(user)
    if not question.strip() or not answer.strip():
        raise gr.Error("Enter both a question and its answer.")
    service().add_entry(question, answer, user=user)
    gr.Info("Added. Only you can see this answer.")
    return "", "", _entries(user), _status(user)


def reset(user: str | None):
    user = _require(user)
    r = service().reset_for(user)
    if r["scope"] == "all":
        gr.Info(f"Shared cache reset to {r['faq_entries']} FAQ answers.")
    else:
        gr.Info(f"Removed {r['removed']} of your private answers and reset your counters.")
    return _entries(user), _status(user)


def model_html() -> str:
    emb = service().embedder
    rows = "".join(f"<tr><td>v<sub>{d['direction']}</sub>, σ {d['sigma']:.2f}</td>"
                   f"<td>{html.escape(', '.join(d['terms']))}</td></tr>" for d in emb.direction_terms(8, 6))
    return (f"<p>The TF-IDF matrix of {emb.report.n_docs:,} training questions and {len(emb.vocab):,} terms "
            f"is reduced to its top k = {emb.dim} singular directions, keeping "
            f"{100 * emb.report.energy_kept:.1f}% of its energy.</p>{spectrum_svg(emb.s)}"
            f"<p>Words with the largest weights in the first columns of V<sub>k</sub>:</p>"
            f"<table>{rows}</table>")


def build_demo() -> gr.Blocks:
    tau0 = Settings.load().cache.threshold
    with gr.Blocks(title="Semantic cache") as demo:
        user = gr.State(None)          # per browser session
        thread = gr.State(None)
        gr.Markdown(f"# Semantic cache · {service().settings.profile} profile\n"
                    "Questions that mean the same thing get the same answer, "
                    "without asking the model again.")
        with gr.Row(equal_height=True):
            name = gr.Textbox(label="Your name", placeholder="e.g. priya", scale=3, max_lines=1,
                              info="No password: names only keep people's conversations apart.")
            sign_btn = gr.Button("Sign in", scale=1)
        who = gr.Markdown("Not signed in.")
        status = gr.Markdown(_status(None))
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                threads = gr.Dropdown(label="Conversations", choices=[], value=None, interactive=True)
                with gr.Row():
                    new_btn = gr.Button("New conversation", size="sm")
                    del_btn = gr.Button("Delete", size="sm", variant="stop")
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=320)
                query = gr.Textbox(label="Your question", placeholder="How do I reset a forgotten password?",
                                   lines=1, max_lines=3)
                hint = gr.Markdown("")
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                tau = gr.Slider(0.30, 0.99, value=tau0, step=0.01, label="Similarity threshold τ",
                                info="Lower: more hits, more wrong answers. Higher: fewer, safer hits.")
                gr.Examples(EXAMPLES, inputs=query)
            with gr.Column(scale=2):
                verdict = gr.HTML(INTRO)
                fan = gr.HTML(angle_fan_svg([], tau0))
                gr.Markdown("Your question *q* points up; cached questions *c* sit at their angle θ from it. "
                            "Inside the green cone, cos θ ≥ τ, the cached answer is reused.")
        matches = gr.Dataframe(headers=["", "Closest cached question", "cos θ", "θ (°)", "Source", "Seen by"],
                               label="Closest cached questions", interactive=False, wrap=True)
        with gr.Accordion("Cached answers", open=False):
            entries = gr.Dataframe(value=_entries(None), headers=["id", "Question", "Source", "Seen by", "Hits"],
                                   interactive=False, wrap=True, max_height=360)
            with gr.Row():
                new_q = gr.Textbox(label="Question", placeholder="Where is the exam hall?")
                new_a = gr.Textbox(label="Answer", placeholder="Room 204, second floor of the academic block.")
            with gr.Row():
                add_btn = gr.Button("Add a private answer")
                reset_btn = gr.Button("Reset", variant="stop")
            gr.Markdown("Reset removes your private answers and zeroes your counters. "
                        "For an admin it resets the shared cache to the FAQ.")
        with gr.Accordion("What the SVD learned", open=False):
            gr.HTML(model_html())

        for trigger in (sign_btn.click, name.submit):
            trigger(sign_in, [name], [user, thread, who, threads, chat, status, entries, verdict])
        for trigger in (query.change, tau.change):
            trigger(preview, [query, tau, user, thread], [fan, matches, hint],
                    trigger_mode="always_last", show_progress="hidden")
        for trigger in (ask_btn.click, query.submit):
            trigger(ask, [query, tau, user, thread],
                    [verdict, fan, matches, status, entries, chat, thread, threads, query])
        threads.input(open_thread, [threads, user], [chat, thread])
        new_btn.click(new_thread, [user], [chat, thread, threads, verdict])
        del_btn.click(delete_thread, [thread, user], [chat, thread, threads])
        add_btn.click(add_entry, [new_q, new_a, user], [new_q, new_a, entries, status])
        reset_btn.click(reset, [user], [entries, status])
    return demo


def launch(port: int = 7860, share: bool = False) -> None:
    demo = build_demo()
    try:
        demo.launch(server_port=port, share=share, css=CSS,
                    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"))
    finally:
        service().save()


if __name__ == "__main__":
    launch()
