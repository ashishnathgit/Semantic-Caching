# Report: phases A–G

Nothing was committed. The `faq` profile is unchanged: k = 60, 327 terms, 43 answers, tau = 0.70, the same
evaluation numbers, the same preflight output and pixel-identical presentation mode (checked against the committed
version). Every new behaviour is behind a flag that is off on `faq`, or in the new `faq-live` profile. All linear
algebra is still in `semcache/linalg.py`. Decisions, with their reasons and costs, are in [DECISIONS.md](DECISIONS.md).

**Final state:** 277 tests pass, 1 expected failure (marked strict; see C below), 0 skipped (the optional
scikit-learn comparison now runs because the MiniLM install brought scikit-learn). `npm run build` passes.
`python -m semcache preflight --profile faq` prints exactly what it printed before any change (byte-identical).
On this machine it ends "7 of 8" on fresh artifacts and "5 of 8" on your current artifacts. The failures are not
code: Ollama isn't running, and your faq cache holds 12 learned answers. I did not run `reset-demo`, because that
would delete them.

Every evaluation row below reports vocabulary size, k, AUC, precision and recall at tau = 0.70, and false hits among
1,000 random Quora questions asked against the 43 FAQ answers. The FAQ AUCs use 172 held-out pairs (86 positives),
so they are a smoke test, not a benchmark.

## Headline numbers

| profile / variant | vocabulary | k | AUC | P @ 0.70 | R @ 0.70 | accepted hits correct | best F1 (tau) | random false hits / 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **faq** (served, unchanged) | 327 | 60 | 0.973 | 0.983 | 0.674 | 58 of 59 | 0.928 (0.40) | 53 |
| faq, negation guard switched on (not the default) | 327 | 60 | 0.963 | 0.983 | 0.663 | 57 of 58 | — | 45 |
| **faq-live** (answers in the corpus + negation guard) | 794 | 75 | 0.976 | 1.000 | 0.616 | 53 of 53 | 0.927 (0.32) | 55 |
| faq-live without the guard | 794 | 75 | 0.986 | 1.000 | 0.616 | 53 of 53 | — | 60 |
| faq-live with latent stopwords | 794 | 75 | 0.966 | 1.000 | 0.605 | 52 of 52 | 0.921 (0.31) | 60 |
| MiniLM on the faq pairs (baseline, evaluation only) | subword | 384 | 0.991 | 1.000 | 0.814 | 70 of 70 | 0.970 (0.54) | 3 |

## Phase A: measurements (no behaviour changes)

**1. evaluate on faq.** AUC 0.973 for LSA and 0.973 for TF-IDF (0.9730 vs 0.9728). Best F1 is 0.928 at tau 0.40.
At tau 0.70, 58 of 59 accepted hits were correct (precision 0.983, recall 0.674).

The slide's **0.967 vs 0.963** and the earlier **0.973** come from the same model scored on different random
negatives. Negative-sampling seed 0 (the function's default) gives 0.967 / 0.963; the config's `eval.seed = 7`
gives 0.973. Across 12 seeds, LSA scores 0.973 ± 0.004 (range 0.962–0.979) and TF-IDF 0.968 ± 0.004. LSA is higher
for all 12 seeds.

**2. Traffic mix** (18% exact repeats, 47% rewordings, 35% new; 10,000 queries; 28 FAQ items cached, 15 held back as
the "new" questions):

| profile | h | wrong hits | lookup | E[cost] per million (estimate) | without the cache |
|---|---:|---:|---:|---:|---:|
| faq | 0.524 | 218 (all on new questions) | 0.12 ms | $475.70 | $1,000 |
| faq-live | 0.486 | 106 | 0.13 ms | $513.90 | $1,000 |

The costs assume $0.001 per LLM call and $0.05 per CPU core-hour, which puts c_lookup at about $1.7e-9. Both
prices are assumptions and can be changed in `[eval]`.

**3. Choosing tau by group 5-fold cross-validation** (folds split by FAQ item). Pooled results on the held-out folds:

| profile | rule | tau per fold | test precision | test recall |
|---|---|---|---:|---:|
| faq | best F1 | 0.40–0.42 | 0.962 | 0.884 |
| faq | precision ≥ 0.99 | 0.50–0.76 | 0.984 | 0.698 |
| faq | fixed 0.70 | 0.70 | 0.983 | 0.674 |
| faq-live | best F1 | 0.32 | 0.974 | 0.884 |
| faq-live | fixed 0.70 | 0.70 | 1.000 | 0.616 |

**4. LSH sweep at tau 0.70.** Setup: Quora model (k = 200), 400 lookups whose true best match is ≥ 0.70. Each
cell is recall of the true best match / ms per lookup.

| b, L | 10k recall / ms | 50k recall / ms |
|---|---:|---:|
| old default 12, 8 | 0.75 / 0.04 | 0.82 / 0.11 |
| **new default 10, 24** | 0.95 / 0.12 | 0.96 / 0.48 |
| brute force | 1.00 / 0.12 | 1.00 / 0.48 |

The full 6 × 6 grid is in `artifacts/faq/eval/measurements.md`. The rule was: recall ≥ 0.95 at both sizes, then the
fastest at 50k. Up to 50k entries, one brute-force product is as fast as LSH at that recall, so `lsh_min_entries`
is now 50,000. LSH stays off on every profile.

Along the way, LSH lookups got about 25% faster: the tables now store row positions instead of entry ids (same
candidates, fewer Python steps).

**5. Fact checks on the faq model:**
- "How do I turn on notifications?" vs "How do I turn off notifications?" have an LSA cosine of **0.930**, well
  above tau. The V_k rows of "on" and "off" have cosine 0.015, but the shared words dominate. This is what Phase D
  addresses.
- V_k row cosines: "cheap"–"budget" **0.525**. "low cost" is **not in the vocabulary**: the bigram appears once,
  below `min_df_ngram = 2`. Separately, "cheap"–"low" and "cheap"–"cost" are both 0.273.
- ρ("cheap flight tickets to goa") = **0.831**.
- Eckart–Young at k = 60 (172 × 327): ‖X − X_k‖_F² = **48.931068**, computed explicitly. The sum of the dropped
  σᵢ² from an independent full SVD is **48.931068**; the difference is 1.5e-12. Energy kept: **71.55%**.

## Phase B: Matrix Lab tab

A **Matrix Lab** switch sits above the normal view. It is not part of presentation mode, all its text is at least
18 px, and it uses the current theme. It contains:
- **Correctness:** Eckart–Young with a check mark, ‖V_kᵀV_k − I‖_F = 3e-14, and the Jacobi solver's work (10 sweeps,
  145,678 rotations, about 0.2 s), with a log-scale plot of the off-diagonal norm per sweep.
- **Singular values and cumulative energy:** two charts rather than one dual-axis chart, each with the served k
  marked.
- **Worked maths** in KaTeX with real numbers: the idf of the rarest word, the TF-IDF row and its norm, e and ρ, the
  best cosine written as Σ e_j f_j, and θ = arccos(score) vs arccos(τ) = 45.6°.
- **Route trace**, also on the answer card: exact? → known words → ρ → negation guard → max score vs τ → angle →
  HIT / LLM / BLOCKED (naming the guard).
- **Latency per request** on a log scale, cache answers vs LLM calls, with a small summary table.
- **k slider + "Refit SVD (sandbox)":** refits a copy and shows AUC, P/R, accepted hits correct and random false hits
  for the sandbox next to the served model. The served model's fingerprint is checked unchanged in the tests.
- **Operation count:** a cache hit costs about 6 kFLOP; an LLM answer costs about 1.1 TFLOP using
  2 · 3×10⁹ · (prompt + output tokens). That is about 2×10⁸ times more, labelled an estimate. Without measured
  token counts it assumes 40 + 150 tokens, and says so.

## Phase C: a vocabulary that grows (`faq-live`)

Built:
- `corpus.include_cached`: the model is also fitted on every shared cached question joined with its answer.
  Placeholder answers add only their question; private entries add nothing.
- Background rebuild every 20 new entries, plus an admin **Rebuild model** button. The rebuild fits with no lock
  held, re-embeds every stored question (zero rows get real vectors; tested), then swaps the model atomically using
  a generation counter.
- A corpus snapshot is saved with the model.

**Item 15, latent stopwords** (what, how, which, best, way, good): **kept off**. Turning them on lowered AUC
0.976 → 0.966 and recall 0.616 → 0.605, and raised random false hits 55 → 60. Top words of the first five SVD
directions:

| direction | off | on |
|---:|---|---|
| 1 | what, how, matrix, python, learning, rank | matrix, python, rank, orthogonal, determinant, find |
| 2 | how, what, python, matrix, notification, install | python, matrix, install, version, python version, package |
| 3 | matrix, python, explain, learning, model, laptop | notification, explain, python, model, app, matrix |
| 4 | notification, app, turn, python, matrix, explain | notification, explain, app, turn, off, model |
| 5 | ticket, notification, book, explain, mumbai, python | explain, ticket, recall, precision, book, similarity |

**Item 16, Gandhinagar regression: not met as written.** After caching "What are the best routes to reach
Gandhinagar" with a realistic LLM-style answer and rebuilding:
- "gandhinagar" becomes a known word, and "how to travel to Gandhinagar" rises from **0.264 to 0.663**. That is
  still below tau = 0.70.
- "best way to reach Gandhinagar" does hit, at 0.927.
- The gap is the word "how": it costs 0.056, and "travel to Gandhinagar" alone scores 0.718.
- The only settings that clear tau were latent stopwords with a ρ cutoff ≤ 0.20 (0.932), and those raise random
  false hits to 88–109 per 1,000. I also swept k (0.35 / 0.5 / 0.7 of the corpus) and σ-power (1 / 0.5 / 0); none
  passed safely.
- I did not tune the test to pass. It asserts what the rebuild does achieve; the strict "> tau" check is kept as a
  strict expected failure with these numbers in its reason, so it will flag when it starts passing.

## Phase D: negation guard (`cache.negation_guard`; on for faq-live, off on faq)

Before a semantic hit (never an exact repeat), the question is compared with the matched cached question:
- **Negation:** the parity of not/no/never/cannot/nor/without, after contraction expansion and CamelCase splitting.
- **Antonyms:** an editable list (the seven pairs asked for; faq-live adds on/disable and off/enable).
- **Numbers and capitalised names:** fires when each side has one the other lacks. ALL-CAPS acronyms and Title Case
  texts are ignored.

When a rule fires, the hit is refused, the question goes to the LLM, and the note, route trace and math panel name
the rule. The math panel's BLOCKED verdict can only appear when the guard is on, so presentation on faq is
unaffected.

The required tests pass:
- "is it not safe to drink tap water" vs "is it safe to drink tap water" is blocked (negation; it scored above tau
  without the guard).
- "turn on notifications" vs "turn off notifications" is blocked (antonym; 0.723 without the guard).
- Plain rewordings still hit.

Cost and benefit on faq-live: random false hits fall 60 → 55 and precision stays 1.000. AUC falls 0.986 → 0.976
because one correct pair is refused ("why can't I evaluate my model…" vs "Why do we split data…").

## Phase E: data and admin

- **Import FAQ CSV** (admin) builds a new profile in `artifacts/<name>/` (for example `campus`). It refuses names in
  config.toml and the profile being served; config.toml and faq are verified byte-identical afterwards. Imported
  profiles also fit on the answers, because a bare question/answer CSV has no paraphrases. Very small imports (around
  12 questions) still produce some wrong hits.
- **Forget LLM-learned rows** (admin) removes every entry whose source is a backend; FAQ and hand-added answers stay.
- **TTL:** `cache.ttl_seconds` (0 = off everywhere), plus a per-entry lifetime when adding by hand. Expired entries
  are misses, exact repeats included, and are replaced by the next answer.
- **Wrong answer** on hits evicts the entry and logs (question, cached question, server-computed score, tau) to
  `feedback/wrong_answers.jsonl`. FAQ and admin answers can only be evicted by an admin, but every report is logged.

## Phase F: online LLM

No Groq or online backend existed, so I added `openai_compatible`:
- It takes the base URL, model and API key from `.env`. `.env` stays ignored; `.env.example` was added and
  un-ignored.
- The key is only ever put in the Authorization header. A test checks it is absent from `/api/health`, responses,
  metrics, the saved cache and the logs.
- A fallback chain is available (e.g. `["openai_compatible", "ollama", "stub"]`); Ollama stays the default.
- The answer card says which backend answered and its token counts, and `/api/metrics` returns `by_backend`. The
  API's real token counts feed the cost cards.
- The README has a privacy note: in online mode, questions that miss the cache go to the provider.

## Phase G: experimental (all off by default)

**27. Per-entry thresholds** (`cache.per_entry_thresholds`). Each entry learns its own τᵢ ≥ τ from hits and
"Wrong answer" reports. In this offline test every wrong hit is reported during tuning, then the thresholds are
frozen for testing:

| profile | rule | test precision | test recall | wrong hits |
|---|---|---:|---:|---:|
| faq | global 0.70 | 0.517 | 0.698 | 28 |
| faq | per-entry (19 entries tightened) | 0.750 | 0.558 | 8 |
| faq-live | global 0.70 | 0.464 | 0.605 | 30 |
| faq-live | per-entry (16 tightened) | 0.806 | 0.581 | 6 |

Precision targets of 0.90, 0.95 and 0.99 gave identical results: every tightened entry had only wrong observations
above τ. The low global precision here comes from 500 random Quora questions in the test stream.

**28. Incremental SVD.** Brand's update is in `linalg.brand_update_rows`. Starting from the full SVD it matches a
full SVD exactly (σ within 7e-14, subspace distance 4e-13). It is **not adopted** for faq-live:

| comparison | max σ difference | subspace distance | time |
|---|---:|---:|---:|
| from the full SVD, same vocabulary | 6.8e-14 | 3.7e-13 | 609 ms |
| from the served rank-75 SVD, same vocabulary | 0.25 | 0.84 | 53 ms |
| vs the real rebuild (182 new terms, new IDF) | 0.25 | 0.96 | 53 ms vs 517 ms refit |

It is 10× faster, but a real rebuild also changes the vocabulary and every IDF weight, which an update of the old
SVD can't reproduce. It fails the 1e-6 rule.

**29. Neural baseline.** MiniLM was installed (exact pins in `requirements-neural.txt`) and runs behind the embedder
interface, but only in `evaluate --neural` and `measure`; the cache never uses it.

| model | vocabulary | k / dim | AUC | P @ 0.70 | R @ 0.70 | random false hits |
|---|---:|---:|---:|---:|---:|---:|
| LSA, faq | 327 | 60 | 0.973 | 0.983 | 0.674 | 53 |
| MiniLM | subword | 384 | 0.991 | 1.000 | 0.814 | 3 |
| MiniLM at LSA's precision (τ 0.51) | subword | 384 | 0.991 | 0.988 | 0.942 | 17 |

MiniLM is clearly stronger on these pairs, at about 0.4–1.7 ms per text vs about 0.1 ms for LSA.

## Skipped or reverted

Nothing was reverted: no phase broke tests it couldn't fix. Items that are partial or deliberately not adopted:
- **16 (Gandhinagar > tau):** not met. It reaches 0.663; see Phase C.
- **28 (incremental SVD):** implemented and measured, not adopted.
- **Preflight "PASS everywhere":** not achievable on this machine without starting Ollama and resetting your cache.
  The output is unchanged from before.

## Where things are

| | |
|---|---|
| Measurements | `artifacts/faq/eval/measurements.md`, `artifacts/faq-live/eval/measurements.md` (`python -m semcache measure --profile …`) |
| Decisions and complexity | [DECISIONS.md](DECISIONS.md) |
| New code | `semcache/measure.py`, `lab.py`, `route.py`, `guards.py`, `importer.py`, `thresholds.py`, `neural.py`; `linalg.py` gained the Jacobi trace, `brand_update_rows`, `subspace_distance` and `RandomHyperplaneLSH.rename` |
| New tests | `tests/test_measure.py`, `test_lab.py`, `test_live.py`, `test_negation_guard.py`, `test_data_admin.py`, `test_online.py`, `test_experimental.py` |
| UI | `frontend/src/components/lab/`, `RouteTrace.tsx`, `AdminTools.tsx`; `frontend/dist` rebuilt |

## Follow-up: wordy questions and the known-word guard

"Where do I go to generate a new password if I lost the old one?" was blocked on faq (2 of 8 words known). The model
would have answered it correctly (0.776 against "How do I reset my password if I forgot it?"); six everyday words it
had never seen ("where", "go", "one", …) counted against it. Fix: unknown everyday words no longer count, and a
question admitted that way is matched on its known topic words only (DECISIONS.md H1–H6).
`python -m semcache measure --only guard` re-measures it:

| profile | rule | vocabulary | k | wordy rewordings right (of 12) | held-out right (of 86) | random false hits / 1,000 |
|---|---|---:|---:|---:|---:|---:|
| faq | before | 327 | 60 | 9 | 57 | 53 |
| faq | **topic-word rule (on)** | 327 | 60 | **12** (0 wrong) | 57 | 55 |
| faq-live | before (kept: rule off) | 794 | 75 | 11 | 52 | 55 |
| quora | either | 40,000 | 200 | — | — | 0 |

faq's AUC (0.973) and precision / recall at 0.70 (0.983 / 0.674) are unchanged: they score pairs without the
known-word guard. Preflight presets and presentation mode are unchanged (pixel-identical); "what is the capital of
france" stays blocked. Not fixed here: on faq-live this question passes the guard but scores 0.10, because answer
text links "generate" to the strong-password FAQ; that is a faq-live model weakness, not the guard.
