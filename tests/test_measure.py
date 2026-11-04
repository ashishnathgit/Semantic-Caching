"""Phase A measurements: the FAQ headline numbers stay put, and the helpers behave."""

import numpy as np
import pytest

from semcache import linalg as la
from semcache.cache import SemanticCache
from semcache.embedder import LSAEmbedder
from semcache.measure import (eckart_young, fact_checks, headline, tau_cross_validation,
                              traffic_simulation)
from semcache.pipeline import build_artifacts


@pytest.fixture()
def faq_model(settings):
    build_artifacts(settings, verbose=False)
    return LSAEmbedder.load(settings.model_dir)


def test_faq_headline_numbers_are_the_slide_numbers(settings, faq_model):
    h = headline(settings, faq_model, with_random=False)
    assert (h["k"], h["vocabulary_size"], settings.cache.threshold) == (60, 327, 0.70)
    assert (h["accepted_hits_correct"], h["accepted_hits"]) == (58, 59)
    assert h["precision"] == pytest.approx(0.983, abs=5e-4)
    assert h["recall"] == pytest.approx(0.674, abs=5e-4)
    assert h["auc"] == pytest.approx(0.973, abs=5e-4)
    assert h["pairs_refused_by_text_guard"] == 0          # the text guard is off on faq


def test_traffic_simulation_counts(settings, faq_model):
    t = traffic_simulation(settings, faq_model, n_queries=500)
    by = t["by_kind"]
    assert sum(c["queries"] for c in by.values()) == 500
    assert by["exact"]["hits"] == by["exact"]["queries"] == by["exact"]["right"]
    assert by["new"]["right"] == 0                         # a held-back item can't be answered right
    assert 0 < t["hit_rate"] < 1
    expected = (t["c_lookup_usd"] + (1 - t["hit_rate"]) * t["assumed"]["llm_usd_per_call"]) * 1e6
    assert t["cost_per_million_usd"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        traffic_simulation(settings, faq_model, mix=(0.5, 0.5, 0.5))


def test_tau_cross_validation_keeps_items_apart(settings, faq_model):
    cv = tau_cross_validation(settings, faq_model, folds=5)
    assert sum(r["test_pairs"] for r in cv["per_fold"]) == cv["pairs"]
    fixed = cv["pooled"]["fixed 0.70"]
    assert (fixed["tp"], fixed["fp"]) == (58, 1)          # same pairs, same tau: same counts


def test_eckart_young_residual_matches_dropped_sigmas(settings, faq_model):
    ey = eckart_young(settings, faq_model)
    assert ey["difference"] < 1e-8
    assert ey["energy_kept"] == pytest.approx(faq_model.report.energy_kept, abs=1e-9)
    assert ey["sigma_match_max_abs"] < 1e-8


def test_fact_checks_name_missing_terms(settings, faq_model):
    f = fact_checks(settings, faq_model)
    low_cost = next(c for c in f["term_cosines"] if c["b"] == "low cost")
    assert low_cost["cosine"] is None and not low_cost["b_in_vocab"]
    assert f["on_off"]["lsa_cosine"] > 0.7                # why the negation guard exists


def test_lsh_keyed_by_row_survives_removals():
    """The cache keys LSH tables by row of C; removing a row moves the last one into its
    slot, and the LSH label must move with it."""
    rng = np.random.default_rng(3)
    V = la.l2_normalize_rows(rng.standard_normal((300, 16)))
    cache = SemanticCache(16, threshold=0.99, max_entries=1000, use_lsh=True, lsh_bits=4,
                          lsh_tables=6, lsh_min_entries=0)
    ids = [cache.add(f"q{i}", f"a{i}", V[i]).id for i in range(300)]
    for entry_id in ids[::3]:
        cache.remove(entry_id)
    assert len(cache.lsh) == len(cache)
    for i in range(300):
        if i % 3 == 0:
            continue
        r = cache.lookup(V[i], 1, record=False)
        assert r.hit and r.best.query == f"q{i}"
