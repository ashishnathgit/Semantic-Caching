"""The route one question takes through the cache, step by step, for the answer card:

    exact repeat?  ->  guards (known words, rho, negation)  ->  max score vs tau
                   ->  angle vs arccos(tau)  ->  HIT / LLM / BLOCKED (which guard)

Every step is a dict: {"id", "label", "value", "status", "detail"} where status is
"yes", "no", "pass", "fail", "info", "off" or "skipped". The last item is the outcome.
"""

from __future__ import annotations

import math


def _deg(score: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, score))))


def route_trace(lookup, *, cache_used: bool = True, follow_reason: str | None = None,
                refreshed: bool = False, known: int = 0, total: int = 0, min_known_ratio: float = 0.0,
                min_latent_coverage: float = 0.0, text_guard_on: bool = False,
                use_llm: bool = True, admission=None) -> dict:
    """Steps and outcome for one LookupResult (semcache.cache)."""
    tau = float(lookup.threshold)
    steps: list[dict] = []
    then = "LLM" if use_llm else "no answer (LLM off)"

    def step(id_, label, value, status, detail=""):
        steps.append({"id": id_, "label": label, "value": value, "status": status, "detail": detail})

    def done(kind: str, detail: str, guard: str | None = None):
        return {"steps": steps, "outcome": {"kind": kind, "guard": guard, "detail": detail,
                                            "then": None if kind == "HIT" else then}}

    if not cache_used:
        why = "forced refresh" if refreshed else f"follow-up ({follow_reason})" if follow_reason else "skipped"
        step("cache", "Cache used?", "no", "no", why)
        return done("LLM", f"The cache was skipped: {why}.")

    exact = lookup.method == "exact"
    step("exact", "Exact repeat?", "yes" if exact else "no", "yes" if exact else "no",
         "case, spacing and punctuation ignored")
    if exact:
        return done("HIT", "An exact repeat is answered before any similarity is computed.")

    ratio = known / total if total else 0.0
    status = admission.status if admission is not None else (
        "ok" if total > 0 and known > 0 and ratio >= min_known_ratio else "blocked")
    if status == "relaxed":
        tk = len(admission.topic_known)
        step("known", "Known words", f"{known}/{total} · topic {tk}/{tk + len(admission.topic_unknown)}", "pass",
             "too few words known, but unknown everyday words don't count: matched on the known topic words "
             + ", ".join(admission.topic_known))
    else:
        step("known", "Known words", f"{known}/{total}", "pass" if status == "ok" else "fail",
             f"need at least {min_known_ratio:.0%} of the words in the vocabulary"
             + (f"; unknown: {', '.join(admission.topic_unknown)}" if admission is not None and status == "blocked"
                and admission.topic_unknown else ""))
    if status == "blocked":
        return done("BLOCKED", "Too few known words: only an exact repeat could hit.", guard="known words")

    rho = float(lookup.rho)
    best = lookup.matches[0] if lookup.matches else None
    if min_latent_coverage > 0:
        if rho < min_latent_coverage:
            step("rho", "ρ guard", f"{rho:.2f}", "info",
                 f"below {min_latent_coverage:.2f}: scored by word overlap (TF-IDF), not the SVD")
        elif best is not None and best.blocked:
            step("rho", "ρ guard", f"{rho:.2f}", "info",
                 f"the cached question's ρ {best.rho:.2f} is below {min_latent_coverage:.2f}: "
                 "that pair is scored by word overlap")
        else:
            step("rho", "ρ guard", f"{rho:.2f}", "pass", f"at least {min_latent_coverage:.2f}: the SVD score is used")
    else:
        step("rho", "ρ guard", f"{rho:.2f}", "off", "min_latent_coverage = 0")

    refused = getattr(lookup, "refused", None)
    if not text_guard_on:
        step("negation", "Negation guard", "off", "off", "cache.negation_guard = false")
    elif refused:
        step("negation", "Negation guard", refused["rule"], "fail", refused["detail"])
    elif best is not None and best.score >= tau:
        step("negation", "Negation guard", "pass", "pass", "no negation, antonym, number or name differs")
    else:
        step("negation", "Negation guard", "not needed", "skipped", "only checked before a semantic hit")

    if best is None:
        step("score", "max score vs τ", "no rows", "no", "the cache is empty")
        return done("LLM", "Nothing cached yet: the model answers.")
    score = float(best.score)
    own = getattr(lookup, "threshold_used", None)
    if own is not None:                       # per-entry thresholds (experimental) raised this entry's tau
        tau = float(own)
    above = score >= tau
    step("score", "max score vs τ", f"{score:.3f} {'≥' if above else '<'} {tau:.2f}", "pass" if above else "fail",
         f"scored by {best.scored_by}" + (" · this entry's own τ, learned from feedback" if own is not None else ""))
    theta, theta_tau = _deg(score), _deg(tau)
    step("angle", "angle vs arccos τ", f"{theta:.1f}° {'≤' if above else '>'} {theta_tau:.1f}°",
         "pass" if above else "fail", "θ = arccos(score)")
    if refused:
        return done("BLOCKED", f"The best match was refused by the negation guard ({refused['rule']}).",
                    guard=f"negation: {refused['rule']}")
    if lookup.hit:
        return done("HIT", "Close enough in meaning: the cached answer is reused, no model call.")
    return done("LLM", f"{tau - score:.3f} short of τ: the model answers, and the answer is cached.")
