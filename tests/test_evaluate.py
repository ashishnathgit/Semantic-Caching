import json

import numpy as np
import pytest

from semcache.data import QuestionPair, faq_corpus, load_faq
from semcache.embedder import LSAEmbedder
from semcache.evaluate import (_json_safe, cache_simulation, latency_benchmark, roc_auc,
                               run_evaluation, threshold_sweep)
from semcache.pipeline import build_artifacts


def test_roc_auc_matches_definition_with_ties():
    scores = np.array([0.9, 0.8, 0.8, 0.3, 0.1])
    labels = np.array([1, 1, 0, 0, 1])
    pos, neg = scores[labels == 1], scores[labels == 0]
    expected = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg])
    assert roc_auc(scores, labels) == pytest.approx(expected)


def test_threshold_sweep_counts():
    rows = threshold_sweep(np.array([0.9, 0.6, 0.4]), np.array([1, 0, 1]), taus=np.array([0.5]))
    assert rows[0]["precision"] == 0.5 and rows[0]["recall"] == 0.5


def test_full_evaluation_on_faq_writes_report(settings, tmp_path):
    build_artifacts(settings, verbose=False)
    report = run_evaluation(settings, out_dir=tmp_path, verbose=False)
    assert 0.5 < report["similarity"]["auc"] <= 1.0
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "report.json").read_text())["rank_sweep"]
    pytest.importorskip("matplotlib")
    assert (tmp_path / "threshold_sweep.png").exists()


def test_latency_benchmark_survives_lsh_lookups_with_no_candidates():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((1400, 50))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # 16 bits x 2 tables over random vectors: many probes share no bucket with anything.
    rows = latency_benchmark(vectors, [1000], lsh_bits=16, lsh_tables=2)
    assert len(rows) == 1 and 0.0 <= rows[0]["lsh_agreement"] < 1.0


def test_json_safe_replaces_nan_and_inf():
    report = {"auc": float("nan"), "rows": [{"x": float("inf")}, {"x": 0.5}], "name": "ok"}
    text = json.dumps(_json_safe(report), allow_nan=False)
    assert json.loads(text) == {"auc": None, "rows": [{"x": None}, {"x": 0.5}], "name": "ok"}


def test_cache_simulation_finds_partner_stored_under_another_spelling(faq_path):
    emb = LSAEmbedder().fit(faq_corpus(load_faq(faq_path)))
    # Both first questions are the same exact-match key, so the cache keeps one entry for them.
    pairs = [QuestionPair("How do I reset my password?", "How can I reset my password?", 1),
             QuestionPair("how do i reset my password", "How can I reset my password?", 1)]
    rows = cache_simulation(emb, pairs, min_known_ratio=0.0, taus=np.array([0.3]))
    assert rows[0]["duplicate_hit_partner"] == 1.0
