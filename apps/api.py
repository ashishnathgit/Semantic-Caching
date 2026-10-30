"""REST API over the semantic cache, used by the React frontend.

    python -m semcache serve            (or: uvicorn apps.api:app --reload)

Users identify themselves with an ``X-User: <name>`` header (no passwords; see
semcache/users.py). Without it a request is anonymous: it can ask one-off
questions against the shared cache, but has no threads and can't change entries.

Admin actions (reset the shared cache, add or delete shared entries) also need an
``X-Admin-Key`` header matching the SEMCACHE_ADMIN_KEY environment variable. Without
that variable they are turned off. Non-admins see only FAQ questions and their own;
other users' questions appear as "another user's question".

Endpoints, all under /api:
    GET    /health              profile, model, cache and LLM backend status
    POST   /session             {name}: sign in (creates the user on first use)
    GET    /me                  the signed-in user, with their counters
    GET    /threads             the user's conversations, most recent first
    POST   /threads             {title?}: start an empty conversation
    GET    /threads/{id}        one conversation with all its messages
    PATCH  /threads/{id}        {title}: rename it
    DELETE /threads/{id}        delete it
    POST   /ask                 {query, threshold?, top_k?, use_llm?, thread_id?, use_cache?}:
                                hit -> cached answer, miss -> LLM answer (stored in the cache),
                                follow-up -> LLM answer with the thread's history (not cached)
    POST   /lookup              same body, but only scores: nothing is stored, no LLM call
    POST   /explain             same body: every number behind the decision (tokens, TF-IDF
                                weights, latent vector, per-word and per-direction parts of
                                each match's score); nothing is stored, no LLM call
    GET    /stats               cache counters (plus the user's own)
    GET    /entries             FAQ and admin-added entries plus the user's own (admins: all),
                                newest first (?limit=100)
    POST   /entries             {question, answer, shared?}: add an entry (private to the user;
                                shared=true needs an admin)
    DELETE /entries/{id}        remove one entry (your own; shared ones need an admin)
    GET    /metrics             counters, latency, estimated cost, time series (?bucket=10s&scope=global|user)
    POST   /metrics/reset       admin: start the metrics over (the old events file is kept)
    POST   /reset-demo          admin: exactly the FAQ answers, metrics archived, user counters zeroed
    POST   /reset               admin: reset the shared cache to the FAQ;
                                anyone else: remove their private entries, zero their counters
    GET    /model               embedder details, singular values, top terms per direction

Interactive docs: http://localhost:8000/docs
After `npm run build` in frontend/, the built UI is served at http://localhost:8000/ too.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from semcache.config import PROJECT_ROOT, Settings
from semcache.pipeline import (MAX_QUERY_CHARS, SemanticCacheService, admin_key_configured,
                               check_admin_key)
from semcache.users import Thread, ThreadNotFound, normalize_username

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_llm: bool = True
    thread_id: str | None = Field(default=None, max_length=32)
    use_cache: bool | None = None      # None = decide (follow-ups skip the cache)


class EntryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    answer: str = Field(min_length=1, max_length=20000)
    shared: bool = False


class SessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ThreadRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _thread_dict(thread: Thread) -> dict:
    return {**thread.summary(), "messages": [
        {"role": m.role, "content": m.content, "created_at": m.created_at, "meta": m.meta}
        for m in thread.messages]}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = SemanticCacheService.from_settings(settings)
        app.state.service.start_warm_up()           # load the LLM before the first question
        yield
        app.state.service.save()

    app = FastAPI(title="Semantic Cache API", version="1.1.0", lifespan=lifespan,
                  description="TF-IDF + SVD (LSA) semantic cache in front of an LLM, for many users.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.api.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])

    def service(request: Request) -> SemanticCacheService:
        return request.app.state.service

    def optional_user(x_user: str | None = Header(default=None)) -> str | None:
        if x_user is None or not x_user.strip():
            return None
        try:
            return normalize_username(x_user)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def required_user(user: str | None = Depends(optional_user)) -> str:
        if user is None:
            raise HTTPException(status_code=401, detail="sign in first: send your name in an X-User header")
        return user

    def is_admin(x_admin_key: str | None = Header(default=None)) -> bool:
        try:
            return check_admin_key(x_admin_key)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def get_thread_or_404(svc: SemanticCacheService, user: str, thread_id: str) -> Thread:
        try:
            return svc.users.get_thread(user, thread_id)
        except ThreadNotFound:
            raise HTTPException(status_code=404, detail=f"no conversation with id {thread_id}") from None

    # ------------------------------------------------------------- status
    @app.get("/api/health")
    def health(request: Request, user: str | None = Depends(optional_user)):
        svc = service(request)
        return {"status": "ok", "profile": settings.profile, "profiles": settings.profiles,
                "model": svc.embedder.describe(), "cache": svc.stats(user),
                "backend": {**svc.backend.describe(), "warmup": svc.warmup},
                "defaults": {"threshold": settings.cache.threshold, "top_k": settings.cache.top_k,
                             "min_latent_coverage": settings.cache.min_latent_coverage},
                "follow_up_detection": settings.users.detect_follow_ups,
                "admin_enabled": admin_key_configured()}

    @app.get("/api/model")
    def model(request: Request):
        emb = service(request).embedder
        energy = emb.energy_profile()
        return {**emb.describe(), "singular_values": [round(float(x), 5) for x in emb.s],
                "energy_cumulative": [round(float(x), 5) for x in energy.cumsum()],
                "directions": emb.direction_terms(n_directions=8, n_terms=6)}

    # -------------------------------------------------------------- users
    @app.post("/api/session")
    def session(body: SessionRequest, request: Request, admin: bool = Depends(is_admin)):
        try:
            return service(request).profile(body.name, admin)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/me")
    def me(request: Request, user: str = Depends(required_user), admin: bool = Depends(is_admin)):
        return service(request).profile(user, admin)

    # ------------------------------------------------------------ threads
    @app.get("/api/threads")
    def list_threads(request: Request, user: str = Depends(required_user)):
        return [t.summary() for t in service(request).users.list_threads(user)]

    @app.post("/api/threads", status_code=201)
    def create_thread(request: Request, body: ThreadRequest | None = None,
                      user: str = Depends(required_user)):
        return _thread_dict(service(request).users.create_thread(user, body.title if body else None))

    @app.get("/api/threads/{thread_id}")
    def get_thread(thread_id: str, request: Request, user: str = Depends(required_user)):
        return _thread_dict(get_thread_or_404(service(request), user, thread_id))

    @app.patch("/api/threads/{thread_id}")
    def rename_thread(thread_id: str, body: RenameRequest, request: Request,
                      user: str = Depends(required_user)):
        svc = service(request)
        get_thread_or_404(svc, user, thread_id)
        try:
            return svc.users.rename_thread(user, thread_id, body.title).summary()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/threads/{thread_id}")
    def delete_thread(thread_id: str, request: Request, user: str = Depends(required_user)):
        if not service(request).users.delete_thread(user, thread_id):
            raise HTTPException(status_code=404, detail=f"no conversation with id {thread_id}")
        return {"deleted": thread_id}

    # ------------------------------------------------------------ asking
    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request, user: str | None = Depends(optional_user),
            admin: bool = Depends(is_admin)):
        if body.thread_id is not None and user is None:
            raise HTTPException(status_code=401, detail="sign in to use conversations")
        try:
            return service(request).ask(body.query, body.threshold, body.top_k, body.use_llm,
                                        user=user, thread_id=body.thread_id,
                                        use_cache=body.use_cache, admin=admin).to_dict()
        except ThreadNotFound:
            raise HTTPException(status_code=404, detail=f"no conversation with id {body.thread_id}") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/lookup")
    def lookup(body: AskRequest, request: Request, user: str | None = Depends(optional_user),
               admin: bool = Depends(is_admin)):
        svc = service(request)
        try:
            result, known = svc.lookup(body.query, body.threshold, body.top_k, record=False, user=user)
            cache_used, follow = (svc.follow_up_for(body.query, user, body.thread_id, body.use_cache)
                                  if user is not None else (body.use_cache is not False, None))
        except ThreadNotFound:
            raise HTTPException(status_code=404, detail=f"no conversation with id {body.thread_id}") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        best = result.matches[0] if result.matches else None
        return {"hit": result.hit and cache_used, "score": round(result.score, 4),
                "threshold": result.threshold, "method": result.method,
                "matches": svc.match_dicts(result, user, admin), "known_terms": known,
                "rho": round(result.rho, 4), "scored_by": best.scored_by if best else None,
                "note": svc.rho_note(result) if known else None,
                "cache_used": cache_used,
                "follow_up": bool(follow and follow.is_follow_up),
                "follow_up_reason": follow.reason if follow else None}

    @app.post("/api/explain")
    def explain(body: AskRequest, request: Request, user: str | None = Depends(optional_user),
                admin: bool = Depends(is_admin)):
        """The math behind one lookup, for the step-by-step view. Nothing is stored."""
        try:
            return service(request).explain(body.query, body.threshold, body.top_k, user=user, admin=admin,
                                            thread_id=body.thread_id)
        except ThreadNotFound:
            raise HTTPException(status_code=404, detail=f"no conversation with id {body.thread_id}") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ----------------------------------------------------------- entries
    @app.get("/api/stats")
    def stats(request: Request, user: str | None = Depends(optional_user)):
        return service(request).stats(user)

    @app.get("/api/entries")
    def entries(request: Request, limit: int = 100, user: str | None = Depends(optional_user),
                admin: bool = Depends(is_admin)):
        limit = max(1, min(limit, 1000))
        return [e.to_dict() for e in service(request).listed_entries(user, admin, limit)]

    @app.post("/api/entries", status_code=201)
    def add_entry(body: EntryRequest, request: Request, user: str = Depends(required_user),
                  admin: bool = Depends(is_admin)):
        try:
            return service(request).add_entry(body.question, body.answer, user=user,
                                              shared=body.shared, admin=admin).to_dict()
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/entries/{entry_id}")
    def delete_entry(entry_id: int, request: Request, user: str = Depends(required_user),
                     admin: bool = Depends(is_admin)):
        try:
            removed = service(request).remove_entry(entry_id, user=user, admin=admin)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail=f"no entry with id {entry_id}")
        return {"deleted": entry_id}

    @app.get("/api/metrics")
    def metrics(request: Request, bucket: str | None = None, scope: str = "global",
                user: str | None = Depends(optional_user)):
        """Counters, latency, estimated cost (what-if) and the time series."""
        if scope not in ("global", "user"):
            raise HTTPException(status_code=422, detail="scope must be global or user")
        if scope == "user" and user is None:
            raise HTTPException(status_code=401, detail="sign in to see your own metrics")
        try:
            return service(request).metrics_report(bucket, user if scope == "user" else None)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/metrics/reset")
    def reset_metrics(request: Request, user: str = Depends(required_user), admin: bool = Depends(is_admin)):
        try:
            return service(request).reset_metrics(admin)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/api/reset-demo")
    def reset_demo(request: Request, user: str = Depends(required_user), admin: bool = Depends(is_admin)):
        try:
            return service(request).reset_demo(admin)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/api/reset")
    def reset(request: Request, user: str = Depends(required_user), admin: bool = Depends(is_admin)):
        return service(request).reset_for(user, admin)

    if Path(FRONTEND_DIST, "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


app = create_app()
