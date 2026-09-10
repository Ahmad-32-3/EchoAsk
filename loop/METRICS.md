# Metrics (real, reproducible — see the commands to regenerate each)

## 1. Threshold sweep (`python -m eval.threshold_sweep`)

Data: 20,000 pairs sampled from Quora Question Pairs (37.1% duplicate rate),
embedded with this project's real embedding backend (spaCy `en_core_web_md`,
mean-pooled word vectors — see DESIGN.md for why not a trained sentence
encoder). Full curve: `eval/results/threshold_sweep.csv` and
`.png`.

**Headline finding: no threshold from 0.50 to 0.99 reaches 97% precision.**
Precision peaks at **0.539** (threshold 0.97, recall 0.257) and *declines*
above that. The max-F1 operating point is threshold 0.90 (F1 0.599,
precision 0.461, recall 0.856).

| threshold | precision | recall | f1 | predicted positive / 20000 |
|---|---|---|---|---|
| 0.80 | 0.393 | 0.994 | 0.563 | 18,780 |
| 0.85 | 0.415 | 0.970 | 0.581 | 17,380 |
| 0.90 | 0.461 | 0.856 | 0.599 | 13,799 |
| 0.93 | 0.498 | 0.665 | 0.569 | 9,912 |
| 0.95 | 0.523 | 0.475 | 0.498 | 6,741 |
| 0.97 | **0.539 (max)** | 0.257 | 0.348 | 3,548 |
| 0.99 | 0.473 | 0.065 | 0.115 | 1,025 |

I also tried a hybrid guard — cosine similarity AND lexical (Jaccard)
token-overlap, grid-searched over both thresholds — to see if a cheap
lexical check could rescue precision. It cannot: no (cosine, Jaccard)
combination in the grid reaches 97% precision either; the best combinations
top out in the same ~0.53-0.55 range as cosine alone. Script:
the grid search in this file's git history / re-derivable from
`app/embeddings.py` + the QQP sample; not kept as a standalone script since
it changed nothing about the conclusion.

**What this means, stated plainly:** mean-pooled static word vectors do not
separate "same intent, different wording" from "different intent, similar
wording" well enough to safely gate a cache on similarity alone, on
open-domain question pairs. This is not primarily a threshold-tuning problem
— at every threshold tried, roughly half of predicted matches are wrong.
The fix is a better similarity signal (a trained sentence encoder — bge-small
or MiniLM — was the original plan; blocked by this sandbox's network
allowlist, see DESIGN.md), not a better threshold.

**Demo configuration:** `similarity_threshold = 0.90` (see `app/config.py`).
Chosen for recall high enough to visibly demonstrate a semantic hit in the
live proxy, explicitly logged here as *not* production-safe at the measured
~46% precision. A reader should not copy this default into a real system
without either swapping the embedder or independently validating precision
on their own traffic.

## 2. Adversarial near-duplicate set (`python -m eval.adversarial_audit`)

10 hand-built pairs, scoped to this project's own support-bot scenario
(refund policy monthly/annual, order-id swap, currency swap, upgrade vs
downgrade, etc — full list and similarities in
`eval/results/adversarial_summary.txt`).

**Raw embedding similarity alone: 10/10 false positives.** Similarities
ranged 0.962-1.000 — i.e. this embedder scores these adversarial pairs *as
similar as or more similar than* genuine duplicates typically score, because
mean-pooling washes out the one or two tokens that actually carry the
distinguishing information (a plan cadence, a currency code, an order
number) against a much longer shared sentence frame. This is the sharper,
domain-specific version of the QQP finding: not a quirk of one open-domain
dataset. A cosine+Jaccard hybrid (tried during the threshold sweep) doesn't
save it either — lexical overlap is exactly what's near-identical in these
pairs.

**Response: added `app/conflict_guard.py`, a real shipped feature, not just
an eval note.** A cheap rule-based check applied after a semantic-similarity
hit clears the threshold: differing numbers, differing members of a handful
of watched conflict groups (billing cadence, currency, role, direction,
support channel, temporal reference), or mismatched negation all veto the
hit and fall through to the next-best candidate (or a genuine miss).
Re-running the same 10 pairs through the real cache (`python -m
eval.adversarial_audit`, `app/conflict_guard.py` enabled) drops the
false-positive rate from **100% to 20% (2/10)**.

The two that remain are the honest limit of a keyword-based guard, and the
audit script documents them rather than hiding them: a scope-addition pair
("cancel my subscription" vs "...and still get a refund") where the second
question needs a longer answer with no lexical conflict to detect, and a
genuine open-vocabulary entity swap ("a paid plan" vs "a support contract")
that isn't in any watched group. `tests/test_conflict_guard.py` pins this
gap down as an explicit `test_known_gap_not_caught_by_design`, specifically
so a future change to the watchlist can't silently claim coverage the guard
doesn't actually have.

**The conclusion this earns:** a rule-based guard is a real, cheap,
worthwhile mitigation — 5x fewer false positives, essentially free to run —
but it is a patch for the failure modes this project happened to anticipate,
not a general solution. The general problem (mean-pooled vectors can't tell
"same shape, different entity" from a true paraphrase) is still only fixed
by a better embedder. Both things are true at once, and the case study
should say so rather than picking the more flattering one.

## 3. Traffic replay (`python -m eval.traffic_replay --n-requests 4000 --seed 7`)

4,000 simulated requests, Zipfian-skewed across 15 support-bot intents (4
deliberately confusable pairs: refund monthly/annual, upgrade/downgrade,
student/teacher discount, email/phone support), each intent expanded to
16-24 paraphrase variants via an openers x actions slot grid — not 3-4
hand-written sentences, which an earlier draft of this eval used and which
made exact-match-only look artificially sufficient (98.8% hit rate) purely
because the paraphrase pool was too narrow to need semantic matching at
all. Full numbers: `eval/results/traffic_replay_summary.txt`.

| config | hit rate | of which semantic | wrong-answer rate (of semantic hits) | $ saved vs no-cache |
|---|---|---|---|---|
| no cache | 0% | — | — | — (baseline: $0.0926 / 4000 req) |
| exact-match only | 93.8% | 0% | n/a | 94.0% |
| semantic-tuned (threshold 0.90 + conflict guard) | 99.1% | 3205 hits | **18.9%** (607/3205) | 99.2% |

**The honest headline: semantic caching bought 5.3 more points of hit rate
over exact-match alone, and one in five of those additional hits served the
wrong answer.** Cost/latency numbers alone (the brief's original framing)
would call this an unambiguous win — 99.2% cost reduction beats 94.0%. Once
correctness is measured too, it's a much closer call: exact-match-only is
strictly safe and already captures the large majority of the available
savings; the semantic tier's marginal contribution comes bundled with a
real, measured error rate, on a workload deliberately built to include
realistic confusable intents, not just favorable ones.

**What this changes about the recommendation:** ship exact-match caching
unconditionally — it has no correctness downside. Ship the semantic tier
only with the conflict guard in place (removing it was not evaluated here
because the adversarial audit already showed what that regresses to), and
only after either swapping in a real sentence encoder or accepting a
measured error budget on the intent set actually in production — not as a
default-on feature justified by cost savings alone.
