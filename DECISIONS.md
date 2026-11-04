# Decisions

Every open choice made while adding phases A–G, why it was made, and its cost.
n = cached entries, k = latent dimensions, m = vocabulary size, N = training texts,
nnz(q) = terms in one question, L = LSH tables, b = bits per table.

Ground rules kept throughout: the `faq` profile keeps k = 60, 327 terms, 43 answers and
tau = 0.70, and its numbers are re-checked by `tests/test_measure.py`; every new behaviour
is behind a flag that is off on `faq` or lives in a new profile; presentation mode is not
touched; all linear algebra stays in `semcache/linalg.py`.

## Phase A: measurements

| # | Decision | Why | Complexity |
|---|---|---|---|
| A1 | New `python -m semcache measure` command (`semcache/measure.py`) instead of growing `evaluate` | Measurements only; `evaluate`'s report stays as it was. Results go to `artifacts/<profile>/eval/measurements.{md,json}` | — |
| A2 | Report AUC across 12 negative-sampling seeds | The slide's 0.967 / 0.963 is seed 0 (the function default); config `eval.seed = 7` gives 0.973 / 0.973. Same model, different random negatives, so one number is misleading | 12 × O(P·nnz·k), P = 172 pairs |
| A3 | Traffic mix: split FAQ items 65 / 35 into cached / held back; "new" questions are held-out phrasings of held-back items | Keeps the whole stream "built from the FAQ held-out phrasings" and makes new questions in-domain (harder than random Quora noise): any hit on them is a wrong answer we can count | unique questions scored once, then reused: O(U·(nnz·k + n·k)) instead of O(Q·…) for Q = 10,000 queries |
| A4 | c_lookup = measured lookup time × an assumed CPU price ($0.05 / core-hour); c_LLM = assumed $0.001 / call; both in `[eval]` | No real prices exist for local Ollama; both are labelled estimates everywhere | — |
| A5 | Tau chosen by group 5-fold CV (folds by FAQ item), rules: best F1, lowest tau with tune precision ≥ 0.95 / 0.99, fixed 0.70 | Grouping stops an item's held-out phrasings leaking between tune and test | O(folds × |TAUS| × P) |
| A6 | LSH sweep uses the Quora model (k = 200) on Quora validation questions | A 50,000-entry cache needs 50,000 real questions; the FAQ has 172 texts. Hyperplane LSH collision depends only on the angle, so b and L transfer | one brute-force pass per size, then per (b, L): O(n·L·b·k) to hash + lookups |
| A7 | LSH tables keyed by row of C instead of entry id; a removal relabels the moved last row (`RandomHyperplaneLSH.rename`) | The profile showed the per-candidate dict lookup id → row dominated an LSH lookup. Same candidates, fewer Python steps: 50k-entry LSH lookups got ~25% faster (b = 10, L = 24: 0.63 → 0.46 ms) | rename O(L); lookup drops an O(candidates) Python loop |
| A8 | LSH defaults b = 10, L = 24 (were 12, 8); `lsh_min_entries` 1,000 → 50,000 | Rule: recall of the true best match ≥ 0.95 at both 10k and 50k, then fastest at 50k. The old (12, 8) had recall 0.75–0.82. Brute force (one BLAS product) was as fast as any LSH setting with that recall up to 50k, so LSH only starts past 50k. `use_lsh` stays false on every profile | memory O(n·L) bucket entries |

## Phase B: Matrix Lab tab

| # | Decision | Why | Complexity |
|---|---|---|---|
| B1 | Lab lives in `semcache/lab.py` + `/api/lab/*`; the tab is a radio switch above the normal view and is never rendered in presentation mode (the `present` branch is checked first and untouched) | Presentation mode must not change. The switch is `role=radio`, not `role=tab`, because `scripts/screenshots.mjs` finds the stage player by its tabs | — |
| B2 | Jacobi instrumentation is an optional `trace` dict on `jacobi_eigh` / `exact_svd` / `randomized_svd` / `truncated_svd` | Results are bit-identical with or without it (tested); the solver already computed the off-diagonal norm every sweep, so recording it is free | O(sweeps) extra memory |
| B3 | Correctness recomputes the fit's own SVD path once per model fingerprint and caches it | The served model stores only V_k and σ; the check needs X and the full spectrum. Small matrices (min side ≤ 300): full exact SVD + explicit dense residual, so the two sides of Eckart–Young are computed independently. Large: ‖X‖² − ‖XV_k‖², never a dense copy | small: O(N·m·min(N,m)) once; large: O(nnz·k) |
| B4 | Worked maths computed server-side (`/api/lab/worked`), not from `/api/explain` | Keeps `/api/explain` (which the presentation stages read) unchanged | O(nnz·k + n·k) per call |
| B5 | Sandbox refit: a fresh `LSAEmbedder` with `max_rank_fraction = 1` (so k can go to min(N, m) − 1), one refit at a time, corpora ≤ 20,000 texts; the user's question is scored against a deep copy of the cache re-embedded with the sandbox | The served embedder and cache objects are never referenced for writing; the copy is discarded with the response | refit O(fit); copy O(n·k) |
| B6 | FLOPs: hit = 4·nnz + 2·nnz·k + 5k + 2·n·k + n (+ 2·postings for word overlap); LLM = 2·P·(prompt + output tokens), always labelled an estimate | 2 FLOPs per parameter per token is the standard forward-pass estimate; attention is left out, so it is a lower bound. P comes from `[llm] model_params_billion`, else the model tag ("llama3.2:3b" → 3B), else the README's demo model, labelled as assumed | O(1) |
| B7 | Charts: σ_i and cumulative energy are two charts, not one dual-axis chart; latency uses one log axis with both series, a legend and a summary table | Dual-axis charts mislead; amber (#d98a00) is below 3:1 contrast, so it gets visible labels and a table | — |
| B8 | Route trace built once in `semcache/route.py`, used by `/api/ask` (answer card) and the lab | One definition of exact → guards → score vs τ → angle → outcome | O(1) |

## Phase C: a vocabulary that grows (profile `faq-live`)

| # | Decision | Why | Complexity |
|---|---|---|---|
| C1 | `corpus.include_cached` adds one training text per **shared** cached entry: question + " " + answer. Placeholder (stub) answers add only their question; private entries add nothing; the FAQ answers the cache is warmed with count from the first build | Placeholder text would teach "placeholder", "ollama"; private entries' words would surface in the shared model's directions (`/api/model`) | corpus grows by n texts |
| C2 | The corpus snapshot is saved as `model/corpus.json` (only when `include_cached`) | The lab must rebuild exactly the X the model was fitted on; on `faq` nothing new is written | O(N) disk |
| C3 | Rebuild = fit in a background thread with no lock held → embed the snapshot's questions (still no lock) → under the lock: re-embed only entries added meanwhile, swap the embedder, bump a model generation, save. Any vector computed outside the lock (ask, add, store after an LLM call) is recomputed inside it if the generation changed | The slow part (TF-IDF + SVD) never blocks requests; the swap is O(late entries) under the lock, so every reader sees either the old model with old vectors or the new model with new vectors, never a mix (k can change between models) | fit O(N·m·min(N,m)) off-lock; swap O(n_late·(nnz·k)) + O(n·k) copy |
| C4 | Auto-rebuild every `rebuild_every` = 20 new entries on faq-live (0 elsewhere); admin "Rebuild model" button; `/api/admin/rebuild` refuses on profiles that don't include the cache | A refit on `faq` would reproduce the same model; refusing keeps `faq` untouched | — |
| C5 | Latent stopwords (what, how, which, best, way, good) stay **off** on faq-live | Measured with the guard on: AUC 0.976 → 0.966, recall 0.616 → 0.605, random false hits 55 → 60, precision equal. The rule ("on only if nothing is worse and AUC or recall is better") says off | — |
| C6 | Gandhinagar regression: the test asserts what the rebuild achieves (the word becomes known, the score rises 0.264 → 0.663, "best way to reach Gandhinagar" hits at 0.927); the strict "how to travel to Gandhinagar > tau" is a **strict xfail** with the numbers | Not met honestly: "how" costs 0.056 (0.718 without it). Every setting that lifts it above tau (latent stopwords with a ρ cutoff ≤ 0.20: 0.932) costs random false hits (88–109 vs 60). The answer text matters: a terse answer gave 0.599, an LLM-style answer that restates the question 0.663 | — |
| C7 | faq-live keeps faq's k-cap (`max_rank_fraction` 0.35, so k = 75 for 215 texts) and sigma_power 1 | Swept 0.35 / 0.5 / 0.7 × latent off/on × ρ cutoff × sigma_power 1 / 0.5 / 0: larger k or smaller sigma_power cut recall at tau to 0.31–0.52 | — |

## Phase D: negation guard

| # | Decision | Why | Complexity |
|---|---|---|---|
| D1 | The guard runs inside `SemanticCache.lookup`, after the best match is found and only when it would be a semantic hit (exact repeats are never checked) | One place, so ask, explain (the math panel), preflight, measure and the route trace all see the same decision. A refused hit becomes a miss; `LookupResult.refused` names the rule | O(words) per lookup, set lookups |
| D2 | Rules in order: negation parity, antonym pairs, numbers, names. Antonyms fire when both texts use a word of the pair but not the same words; numbers and names fire only when **each** side has one the other lacks | "on" is also a preposition: "tips on X" vs "advice on X" must not fire; "install Python on Mac" vs "install Python" must not fire | — |
| D3 | CamelCase is split inside the guard; ALL-CAPS acronyms and Title Case texts give no names | Measured on the 172 FAQ pairs: "ModuleNotFoundError" vs "module not found" and "AI" vs "LLM" were refused as false conflicts. After the change one positive pair is still refused ("why can't I evaluate my model…" vs "Why do we split data…") — the cost of the negation rule | — |
| D4 | faq-live's antonym list adds two cross pairs, on/disable and off/enable | "disable push notifications" vs "turn on notifications" uses one word of each pair; the list is editable in config.toml | — |
| D5 | A pair the guard refuses scores −1 in `measure` | Precision/recall and AUC then describe what the cache would really serve | — |
| D6 | In the live math panel the guard only adds a BLOCKED verdict when it fires (`verdict = "refused"`) | Presentation on `faq` is unchanged: the guard is off there, so that branch can't run | — |

## Phase E: data and admin

| # | Decision | Why | Complexity |
|---|---|---|---|
| E1 | "Import FAQ CSV" writes `artifacts/<name>/source.csv` + `profile.toml` (overrides) and builds that profile; `Settings.load` also discovers these imported profiles. Names that exist in config.toml, and the profile being served, are refused | config.toml and `faq` are never edited; deleting the folder removes the profile; `artifacts/` is git-ignored, so imported data is never committed | build O(fit) once; discovery O(profiles) |
| E2 | Imported profiles fit on question + answer (`include_cached = true`) | A plain question/answer CSV has no paraphrases: with 5 disjoint questions k was 2 and most questions got a zero vector. With answers, 12 campus questions gave k = 8 and sensible hits. Small imports still give some wrong hits (12 questions: "where is the doctor" → bike parking at 0.80); paraphrases fix that | — |
| E3 | CSV limits: 2 MB, 5,000 questions, at least 3; duplicate questions are dropped | Bounded synchronous build time on the request path (an admin action) | O(rows) |
| E4 | "Forget LLM-learned rows" removes every entry whose source is not `faq` or `manual` (Ollama, online models, placeholders) | Placeholders came from a backend too | O(n) |
| E5 | TTL: `cache.ttl_seconds` (0 = off, every profile) + per-entry `ttl_seconds` when adding by hand. An `_expires` array per row of C is folded into the visibility mask only when some entry can expire; exact repeats skip expired rows; a new answer to the same question replaces the expired row; `purge_expired()` drops them | With TTL off the lookup path is unchanged (no mask); expiry is a vectorised comparison, not a per-entry Python check | O(n) mask only while some entry can expire |
| E7 | Admins can delete any cached question (FAQ and other users' private ones too): a Remove button on every row, plus checkboxes and "Delete selected" with a confirm step, backed by `POST /api/admin/entries/delete {ids}` | One request and one save for many rows instead of N requests and N autosaves; unknown ids are reported, not errors. FAQ answers come back with Reset demo | O(k) removals (each O(1): the last row of C moves into the gap) + one save |
| E6 | "Wrong answer": logs {question, cached question, server-computed score, tau, model} to `artifacts/<profile>/feedback/wrong_answers.jsonl` and evicts the entry. FAQ answers and shared hand-added answers are evicted only by an admin (anyone's report is still logged); someone else's private entry can't be reported | Anyone could otherwise wipe the curated FAQ; the score is recomputed server-side, not trusted from the client | O(n·k) for the score |

## Phase F: online LLM

| # | Decision | Why | Complexity |
|---|---|---|---|
| F1 | No Groq or other online backend existed; added one `OpenAICompatibleBackend` (POST `{base}/chat/completions`, Bearer key) using only the standard library | One client covers OpenAI, Groq, OpenRouter, Together and local vLLM; no new dependency to pin | — |
| F2 | Base URL, model and key come from `.env` (`SEMCACHE_OPENAI_BASE_URL`, `SEMCACHE_OPENAI_MODEL`, `SEMCACHE_OPENAI_API_KEY`), read by a 20-line loader in `config.py`; variables already in the environment win. `.env` stays git-ignored; `!.env.example` was added so only the example can be committed | python-dotenv is not a project dependency; the key is never in config.toml | O(lines) once |
| F3 | The key is only put in the Authorization header: `describe()` reports `api_key_set: true/false`, every error message is redacted, and a test checks the key is absent from `/api/health`, responses, metrics, the saved cache and the logs | Requirement: never to the frontend or logs | — |
| F4 | `[llm] chain = [...]` builds a `ChainBackend`; empty (the default everywhere) keeps today's Ollama + stub fallback. A real model later in the chain is cached normally; the stub answering only because everything before it failed is a FALLBACK and never cached; the stub caches its placeholder only when it is first in the chain | Same caching rules as before, extended to N backends | O(chain length) per miss |
| F5 | The backend that answered is stored on each metric event; `/api/metrics` returns `by_backend`; the answer card shows "Answered by …, model, prompt + output tokens" and what failed first | Requirement 25; real token counts (`usage.prompt_tokens`, `usage.completion_tokens`) flow into the existing cost cards unchanged | — |
| F6 | `describe()` of the online backend does not call the network | `/api/health` is polled; a network call per poll would cost latency and possibly money | — |

## Phase G: experimental (all off by default)

| # | Decision | Why | Complexity |
|---|---|---|---|
| G1 | Per-entry thresholds: each entry keeps its last 50 (score, correct?) observations; an unreported semantic hit counts as correct, a "Wrong answer" as wrong (it replaces the positive its hit recorded). τ_i = the smallest t ≥ τ on a 0.01 grid where precision among its observations scoring ≥ t reaches `entry_precision_target`. Only the best match is checked against max(τ, τ_i) | Without asking the LLM for every hit there are no labels below τ, so τ_i can only tighten. Stored on the entry, so it survives restarts | update O(50 × grid); lookup O(1) |
| G2 | With the flag on, "Wrong answer" on a semantic hit tightens τ_i instead of evicting; a wrong exact repeat is still evicted | No threshold can fix an exact repeat; eviction would also throw away the feedback the flag exists to learn from | — |
| G3 | Measured offline (`measure --only thresholds`): on faq, test precision 0.517 → 0.750 and recall 0.698 → 0.558 (19 entries tightened, e.g. "How do I delete my account?" → 0.96); faq-live 0.464 → 0.806, recall 0.605 → 0.581. The 0.90 / 0.95 / 0.99 targets give the same result because every tightened entry only had wrong observations above τ | Fewer wrong answers for some lost recall: worth trying behind the flag, not a default | — |
| G4 | Brand's update (`linalg.brand_update_rows`, built on our Gram–Schmidt QR and Jacobi SVD) is **not** the faq-live rebuild path | Exact from a full-rank SVD (σ within 7e-14, subspace 4e-13), but the served model is truncated (k = 75) and a real rebuild adds 182 new terms and changes every IDF weight; against the real refit σ differs by 0.25 and the subspaces by sin θ = 0.96. It is 10× faster (51 vs 499 ms) but fails the 1e-6 rule | Brand O((N + m)(k + p)² + (k + p)³) vs refit O(N·m·min(N, m)) |
| G5 | `subspace_distance` = largest singular value of (I − V₁V₁ᵀ)V₂ | √(1 − cos²) loses half the digits near 0 (it gave 2e-7 for identical subspaces) | O(n·k²) |
| G6 | MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) behind the embedder interface in `semcache/neural.py`, used only by `evaluate --neural` and `measure --only neural`; dependencies pinned in `requirements-neural.txt` (sentence-transformers 6.2.0, transformers 5.22.1, torch 2.15.0), installed into the project venv | The served cache never imports it; its tests skip when the package or the weights are missing | ~1 ms per text vs ~0.1 ms for an LSA embedding |
| G7 | MiniLM is reported at the profile's τ and at the τ that gives it LSA's precision, with guards off | Its cosines have another scale; ρ and the known-word guard are LSA ideas | — |

## Follow-up: wordy questions blocked by the known-word guard

Report: "Where do I go to generate a new password if I lost the old one?" was blocked ("only 2 of 8 words are known")
although it means the same as the reset-password FAQ. The faq model scores it 0.776 against the right answer once it
is allowed through; it was stopped only because six everyday words it never saw ("where", "go", "one", …) counted
against it.

| # | Decision | Why | Complexity |
|---|---|---|---|
| H1 | Topic-word rule (`cache.relaxed_known_words`): when the plain rule fails, unknown **everyday** words (`text.EVERYDAY_WORDS`: question words, light verbs, indefinites, adverbs, greetings; no adjectives) are not counted; ≥ 2 known topic words making up ≥ 40% of the topic words admit the question | The guard's job is to catch questions about something else ("capital", "france" are topic words, so those stay blocked). Everyday words say nothing about the topic | O(words) per question |
| H2 | An admitted question is matched on its **known topic words only**: everyday terms' columns are zeroed in its TF-IDF row before the fold-in | The first prototype (only "don't count unknown everyday words") added 11 wrong hits per 1,000 random questions, almost all carried by known everyday words ("why", "how", "find"). Matching on topic words and requiring two of them cut that to 2 | O(nnz) per admitted question |
| H3 | Thresholds 2 words / 40% chosen on: 12 hand-written wordy rewordings (`data/eval_wordy_paraphrases.csv`), the 86 FAQ held-out phrasings and 1,000 random Quora questions. faq: wordy 9 → 12 of 12 right (0 wrong), held-out unchanged (57 of 86), random false hits 53 → 55; "capital of france" stays blocked | Variants measured: 1 / 50% (11 of 12, +1), 2 / 50% (11, +1), **2 / 40% (12, +2)**, 2 / 34% (12, +5), 2 / 0% (12, +35). The wordy set includes the reported question and is small: a sanity check, not a benchmark | — |
| H4 | On for faq and quora, **off for faq-live** | Measured per profile: faq-live already learns everyday words from answer text (wordy 11 of 12 either way) and the rule only added 4 random false hits; quora's random false hits are 0 either way | — |
| H5 | It is a query-time policy on the embedder (`KnownWordPolicy`, set from `[cache]` by `pipeline.configure_embedder`), not part of the fitted model: the faq fingerprint, vocabulary (327), k (60) and evaluation numbers are unchanged. The cache records the policy its vectors were made under and re-embeds once when it changes | Changing a guard setting must never need a rebuild, and stale vectors must never survive one | re-embed O(n·nnz·k) once |
| H6 | The note on a blocked answer now names the unknown topic words and the everyday words that were ignored; relaxed admissions say what they were matched on; the route trace shows "topic 2/5"; `measure --only guard` re-measures the trade-off | So the next "why was this blocked?" is answered by the UI, and any change to the list or thresholds can be checked in one command | — |
