"""Per-entry thresholds (experimental, vCache-style; cache.per_entry_thresholds, off by default).

Every cached entry keeps its last observations (score, correct?) : a hit that nobody reports
counts as correct, a "Wrong answer" report as wrong. Its own threshold is

    tau_i = the smallest t >= tau (on a 0.01 grid) such that, among its observations with
            score >= t, the share that was correct is at least ``precision_target``

(no observation above t counts as precise). So tau_i only ever tightens the global tau, for
entries that attract wrong hits; an entry nobody complained about keeps tau. A lookup is a
hit when the best match's score >= max(tau, tau_i). O(obs) per update, O(1) per lookup.
"""

from __future__ import annotations

import random

import numpy as np

MAX_OBSERVATIONS = 50


def learn_threshold(observations, base_tau: float, target: float, step: float = 0.01) -> float | None:
    """tau_i for one entry, or None when it equals base_tau (nothing to tighten)."""
    if not observations:
        return None
    obs = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
    scores, labels = obs[:, 0], obs[:, 1]
    for t in np.round(np.arange(base_tau, 1.0 + 1e-9, step), 4):
        above = scores >= t - 1e-12
        if not above.any() or labels[above].mean() >= target:
            return None if t <= base_tau + 1e-12 else float(t)
    return 1.0


def add_observation(entry, score: float, correct: bool, base_tau: float, target: float,
                    replace_positive: bool = False) -> float | None:
    """Record one observation on ``entry`` and recompute its threshold. With
    ``replace_positive`` a wrong report replaces the positive its hit recorded."""
    obs = list(entry.feedback or [])
    s = round(float(score), 4)
    if replace_positive and not correct:
        for i in range(len(obs) - 1, -1, -1):
            if obs[i][1] == 1 and abs(obs[i][0] - s) < 1e-4:
                del obs[i]
                break
    obs.append([s, 1 if correct else 0])
    entry.feedback = obs[-MAX_OBSERVATIONS:]
    entry.threshold = learn_threshold(entry.feedback, base_tau, target)
    return entry.threshold


# ---------------------------------------------------------------------------
# offline study: does it help on the FAQ?
# ---------------------------------------------------------------------------

def per_entry_threshold_study(settings, embedder, targets=(0.9, 0.95, 0.99), seed: int = 5) -> dict:
    """Cache: the 43 FAQ answers. Tune stream: one held-out phrasing per FAQ item plus 500
    random Quora questions (any hit on them is wrong); the user reports every wrong hit.
    Test stream: the other held-out phrasing per item plus 500 other Quora questions, with
    the learned thresholds frozen. Compared with the global tau alone."""
    from .data import load_faq
    from .evaluate import sample_quora_questions
    from .measure import cache_for, fill

    items = [i for i in load_faq(settings.path(settings.paths.faq_csv)) if len(i.heldout) >= 2]
    ratio, tau = settings.cache.min_known_ratio, settings.cache.threshold
    randoms: list[str] = []
    if settings.path(settings.paths.quora_eval).exists():
        randoms = sample_quora_questions(settings, 1000)
        random.Random(seed).shuffle(randoms)
    tune = [(i.heldout[0], i.question) for i in items] + [(q, None) for q in randoms[:500]]
    test = [(i.heldout[1], i.question) for i in items] + [(q, None) for q in randoms[500:]]

    def run(stream, cache, learn: bool, target: float | None):
        hits = right = wrong = 0
        texts = [q for q, _ in stream]
        V, X, rho = embedder.encode_with_rho(texts, ratio)
        for i, (q, want) in enumerate(stream):
            r = cache.lookup(V[i], 1, record=False, query_text=q, query_tfidf=X[i], query_rho=float(rho[i]))
            if not r.hit:
                continue
            hits += 1
            ok = want is not None and r.best.query == want
            right += ok
            wrong += not ok
            if learn and r.method != "exact":
                add_observation(r.best, r.score, ok, tau, target)
        positives = sum(w is not None for _, w in stream)
        return {"hits": hits, "right": right, "wrong": wrong, "recall": right / positives if positives else float("nan"),
                "precision": right / hits if hits else float("nan")}

    base_cache = cache_for(settings, embedder)
    fill(base_cache, embedder, [i.question for i in items], ratio)
    rows = [{"rule": f"global tau {tau:.2f}", "tune": run(tune, base_cache, False, None),
             "test": run(test, base_cache, False, None), "entries_tightened": 0}]
    for target in targets:
        cache = cache_for(settings, embedder, per_entry_thresholds=True, entry_precision_target=target)
        fill(cache, embedder, [i.question for i in items], ratio)
        tuned = run(tune, cache, True, target)
        tight = [e for e in cache.entries() if e.threshold]
        rows.append({"rule": f"per-entry, precision target {target:.2f}", "tune": tuned,
                     "test": run(test, cache, False, None), "entries_tightened": len(tight),
                     "examples": sorted([{"query": e.query, "threshold": e.threshold} for e in tight],
                                        key=lambda d: -d["threshold"])[:5]})
    return {"tau": tau, "items": len(items), "random_tune": min(500, len(randoms)),
            "random_test": max(0, len(randoms) - 500), "rows": rows}


def thresholds_markdown(r: dict) -> str:
    lines = ["## Per-entry thresholds (experimental)", "",
             f"Cache: {r['items']} FAQ answers. Tune: one held-out phrasing per item + {r['random_tune']} random "
             f"Quora questions, every wrong hit reported. Test: the other phrasing + {r['random_test']} other "
             "Quora questions, thresholds frozen.", "",
             "| rule | entries tightened | test hits | right | wrong | test precision | test recall |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in r["rows"]:
        t = row["test"]
        lines.append(f"| {row['rule']} | {row['entries_tightened']} | {t['hits']} | {t['right']} | {t['wrong']} | "
                     f"{t['precision']:.3f} | {t['recall']:.3f} |")
    ex = next((row.get("examples") for row in r["rows"][1:] if row.get("examples")), None)
    if ex:
        lines += ["", "Most tightened entries: " + "; ".join(f"\"{e['query']}\" → {e['threshold']:.2f}" for e in ex)]
    return "\n".join(lines) + "\n"
