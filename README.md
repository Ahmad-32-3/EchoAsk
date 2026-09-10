# EchoAsk

A caching proxy for LLM APIs, plus a measurement of when semantic caching helps and when it hands back the wrong answer.

Support bots see a small set of questions, asked constantly, in slightly different words. Exact-string cache barely helps: "how do I reset my password" and "I forgot my password" never match. Cache on embedding similarity and you can skip redundant calls, but only if the similarity threshold is safe. That threshold is the hard part.

## What it does

| Path | Behavior |
|---|---|
| Identical prompt seen before | Exact-match hit. Instant, no extra model call. |
| A paraphrase of a prompt seen before | Semantic hit if cosine similarity clears the threshold and survives a conflict guard (differing numbers, currencies, roles, billing cadence, negation). See `app/conflict_guard.py`. |
| Prompt not seen before | Miss. Calls the origin model and caches the answer under both tiers. |
| Any request | Scoped to a required `namespace`. One tenant cannot see another's cached answer. |

## Findings

1. No similarity threshold is safe on its own. Sweeping 0.50-0.99 against 20,000 labeled Quora Question Pairs, precision peaks at 53.9%. At the best threshold tried, roughly half of predicted matches are wrong. A cosine plus lexical-overlap hybrid does not rescue it.
2. On this project's support-bot pairs it is worse. Ten hand-built pairs that look similar but mean something different (refund policy: monthly vs annual; order #4521 vs #7788; USD vs EUR) score 0.96-1.00 cosine similarity, as similar as true duplicates. Raw similarity: 10/10 false positives.
3. A cheap rule-based guard helps, but is not a general fix. Vetoing a hit on differing numbers, currency, role, billing cadence, or negation drops that false-positive rate from 100% to 20%. The two that remain are documented gaps: a scope change with no lexical signal, and a novel entity swap outside the watched vocabulary.
4. In mixed traffic the tradeoff is close. Simulating 4,000 Zipfian-distributed requests across 15 support intents: exact-match-only already gets 93.8% hit rate and 94.0% cost savings, with no correctness risk. Adding the tuned semantic tier pushes that to 99.1% hit rate and 99.2% savings, but 18.9% of the extra hits serve the wrong answer.

Ship exact-match caching. Ship semantic caching only with a conflict guard, and only after using a real trained sentence encoder or accepting a measured error budget on your traffic. Cost savings alone are not a reason to turn it on.

Full numbers: `loop/METRICS.md`.

## Why the embedder is spaCy

## Run

```bash
docker compose up --build
```

Starts Redis and the proxy on `:8000`. Origin defaults to a deterministic stub (`app/origin.py`).

```bash
curl -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"namespace":"demo","prompt":"how do I reset my password"}'
# cache_hit: false, cache_tier: null

curl -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"namespace":"demo","prompt":"I forgot my password, help"}'
# cache_hit: true, cache_tier: "semantic"
```

Real model on a miss:

```bash
ollama pull qwen2.5:3b
ORIGIN_BACKEND=ollama OLLAMA_MODEL=qwen2.5:3b docker compose up --build
```

## Reproduce the evals

```bash
pip install -r requirements.txt -r requirements-eval.txt
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"
python -m eval.download_qqp
python -m eval.threshold_sweep
python -m eval.adversarial_audit
python -m eval.traffic_replay
pytest tests/ -v
```

## Layout

- `app/` proxy: FastAPI, two-tier cache, conflict guard, embeddings, origin, cost model
- `eval/` three evals and results under `eval/results/`
- `tests/` tiering, namespace isolation, TTL, conflict guard
- `loop/METRICS.md` eval numbers
