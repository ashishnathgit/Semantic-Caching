"""Phase B: the Matrix Lab and the route trace."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from semcache import linalg as la
from semcache.lab import model_params_billion
from semcache.pipeline import SemanticCacheService


@pytest.fixture()
def client(settings):
    from apps.api import create_app
    with TestClient(create_app(settings)) as c:
        c.post("/api/session", json={"name": "lab"})
        c.headers.update({"X-User": "lab"})
        yield c


def test_jacobi_trace_leaves_the_result_unchanged():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30))
    S = A @ A.T
    w0, W0 = la.jacobi_eigh(S)
    trace: dict = {}
    w1, W1 = la.jacobi_eigh(S, trace=trace)
    assert np.array_equal(w0, w1) and np.array_equal(W0, W1)
    assert trace["converged"] and trace["sweeps"] >= 1 and trace["rotations"] > 0
    offs = trace["off_norms"]
    assert len(offs) == trace["sweeps"] + 1 and offs[-1] <= trace["tol"] * trace["frobenius"]
    assert offs[0] > offs[-1]


def test_route_trace_for_the_presets(settings):
    svc = SemanticCacheService.from_settings(settings, verbose=False)
    exact = svc.ask("How do I reset my password if I forgot it?").route
    assert exact["outcome"]["kind"] == "HIT" and [s["id"] for s in exact["steps"]] == ["exact"]
    para = svc.ask("cheap flight tickets to goa").route
    ids = [s["id"] for s in para["steps"]]
    assert ids == ["exact", "known", "rho", "negation", "score", "angle"] and para["outcome"]["kind"] == "HIT"
    assert next(s for s in para["steps"] if s["id"] == "negation")["status"] == "off"    # off on faq
    assert "45.6°" in next(s for s in para["steps"] if s["id"] == "angle")["value"]
    blocked = svc.ask("what is the capital of france").route
    assert blocked["outcome"]["kind"] == "BLOCKED" and blocked["outcome"]["guard"] == "known words"
    miss = svc.ask("what is the best laptop for students").route
    assert miss["outcome"]["kind"] == "LLM"


def test_lab_correctness(client):
    c = client.get("/api/lab/correctness").json()
    assert c["shape"] == [172, 327] and c["k"] == 60
    assert c["eckart_young"]["equal"] and c["orthogonality"]["ok"]
    assert c["jacobi"]["converged"] and c["jacobi"]["sweeps"] == len(c["jacobi"]["off_norms"]) - 1
    assert len(c["spectrum"]["sigma"]) == 172 and c["spectrum"]["full"]


def test_lab_worked_numbers_add_up(client):
    w = client.post("/api/lab/worked", json={"query": "cheap flight tickets to goa"}).json()
    assert w["rarest"]["term"] == "cheap" and w["rarest"]["df"] == 2
    assert np.isclose(np.linalg.norm(w["e"]), 1.0, atol=1e-3)
    m = w["match"]
    assert m["score"] == pytest.approx(0.858, abs=1e-3)
    assert m["lsa"] == pytest.approx(sum(t["product"] for t in m["top"]) + m["rest"], abs=2e-3)
    assert w["theta_tau_deg"] == pytest.approx(45.57, abs=0.01)


def test_lab_ops_and_param_parsing(client):
    o = client.post("/api/lab/ops", json={"query": "cheap flight tickets to goa"}).json()
    assert o["hit"]["flops"] == sum(p["flops"] for p in o["hit"]["parts"])
    assert o["llm"]["params_billion"] == 3 and o["ratio"] > 1e5
    assert model_params_billion("llama3.2:3b") == 3
    assert model_params_billion("qwen2.5-7b-instruct") == 7
    assert model_params_billion("gpt-4o-mini") is None


def test_sandbox_refit_never_touches_the_served_model(client):
    before = client.get("/api/health").json()["model"]["fingerprint"]
    r = client.post("/api/lab/refit", json={"k": 30, "query": "cheap flight tickets to goa"})
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 30 and body["eckart_young"]["equal"]
    assert body["served"]["k"] == 60 and body["served_headline"]["precision"] == pytest.approx(0.983, abs=5e-4)
    assert client.get("/api/health").json()["model"]["fingerprint"] == before
    assert client.post("/api/lab/refit", json={"k": 500}).status_code == 422


def test_lab_latency_lists_answered_questions(client):
    client.post("/api/ask", json={"query": "cheap flight tickets to goa"})
    pts = client.get("/api/lab/latency").json()["points"]
    assert pts and pts[-1]["kind"] == "hit" and "user" not in pts[-1]
