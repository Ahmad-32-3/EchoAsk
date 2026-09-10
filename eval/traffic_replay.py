"""Simulated support-bot traffic through three real proxy configurations —
no-cache, exact-match-only, and the tuned semantic cache (threshold +
conflict guard) — to measure hit rate, latency, $ cost, and (since we now
know the embedder's real precision) the wrong-answer rate a semantic hit
actually introduces under realistic mixed traffic.

Only the origin LLM is faked (StubOrigin, by design — see app/origin.py).
Everything else — SemanticCache, the conflict guard, the cost model — is the
real system under test, exercised through its real interfaces.

Traffic model: a fixed set of support-bot intents, each with several
paraphrase variants, sampled with Zipfian skew (a handful of intents
dominate, as real repeat-question traffic does) — same request sequence
replayed identically across all three configs for a fair comparison.
Several intent pairs are deliberately confusable (refund monthly/annual,
upgrade/downgrade, student/teacher discount, email/phone support) so the
replay can actually observe the failure mode the adversarial audit found,
not just intents that were never going to collide.

Usage: python -m eval.traffic_replay [--n-requests 4000] [--seed 7]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.cache import SemanticCache  # noqa: E402
from app.cost_model import call_cost_usd  # noqa: E402
from app.origin import StubOrigin  # noqa: E402

# Each intent is a (openers x actions) slot grid rather than a handful of
# hand-written sentences. A real support bot doesn't see 3-4 phrasings of
# "reset my password" — it sees dozens, which is exactly the traffic
# pattern that makes exact-match caching alone insufficient and motivates
# semantic caching in the first place. An earlier version of this script
# used ~3 variants/intent and found exact-match-only already captured 98.8%
# of the hit rate — an artifact of too narrow a paraphrase pool, not a real
# finding, so this was widened until it stopped being one.
_INTENT_SLOTS: dict[str, dict[str, list[str]]] = {
    "password_reset": {
        "openers": ["How do I", "What's the way to", "Can you tell me how to", "I need help to", "Could you explain how to", "Is there a way to"],
        "actions": ["reset my password", "change my account password", "recover my forgotten password", "set a new password"],
    },
    "login_issue": {
        "openers": ["Why can't I", "I'm unable to", "Help, I can't", "Something's wrong, I can't", "It won't let me"],
        "actions": ["log in to my account", "sign in to the site", "access my account", "get past the login screen"],
    },
    "order_status": {
        "openers": ["Where is", "Can you check on", "Has", "I want an update on", "What's the status of"],
        "actions": ["my order", "my recent order", "my shipment", "my package"],
    },
    "refund_monthly": {
        "openers": ["What's", "Can you explain", "I have a question about", "Could you clarify"],
        "actions": ["the refund policy for monthly plans", "how refunds work on monthly billing", "monthly subscription refund rules"],
    },
    "refund_annual": {
        "openers": ["What's", "Can you explain", "I have a question about", "Could you clarify"],
        "actions": ["the refund policy for annual plans", "how refunds work on annual billing", "annual subscription refund rules"],
    },
    "cancel_subscription": {
        "openers": ["How do I", "What's the process to", "I'd like to", "Can you help me"],
        "actions": ["cancel my subscription", "cancel my plan", "close my account", "stop my subscription"],
    },
    "upgrade_plan": {
        "openers": ["How do I", "Can you help me", "What's involved to", "I'd like to"],
        "actions": ["upgrade to the Pro plan", "move to a higher tier", "upgrade my subscription", "get more features on a higher plan"],
    },
    "downgrade_plan": {
        "openers": ["How do I", "Can you help me", "What's involved to", "I'd like to"],
        "actions": ["downgrade from the Pro plan", "move to a lower tier", "downgrade my subscription", "switch to a cheaper plan"],
    },
    "billing_address_update": {
        "openers": ["How do I", "Can you help me", "I need to", "Where do I go to"],
        "actions": ["update my billing address", "change my billing address", "edit my payment address", "fix my billing details"],
    },
    "api_key_reset": {
        "openers": ["How do I", "Can I", "My key stopped working, how do I", "What's the process to"],
        "actions": ["regenerate my API key", "get a new API key", "reset my API key", "rotate my API credentials"],
    },
    "student_discount": {
        "openers": ["Is there", "Do you offer", "Can I get", "Are you running"],
        "actions": ["a discount for students", "student pricing", "an education discount for students", "a student rate"],
    },
    "teacher_discount": {
        "openers": ["Is there", "Do you offer", "Can I get", "Are you running"],
        "actions": ["a discount for teachers", "teacher pricing", "an education discount for teachers", "a teacher rate"],
    },
    "support_channel_email": {
        "openers": ["Does", "Is", "Can I get", "Do you offer"],
        "actions": ["support include email", "email support included", "support available over email", "help via email"],
    },
    "support_channel_phone": {
        "openers": ["Does", "Is", "Can I get", "Do you offer"],
        "actions": ["support include phone support", "phone support included", "support available over the phone", "help via phone"],
    },
    "account_balance": {
        "openers": ["What is", "Can you tell me", "How much is", "Could you check"],
        "actions": ["my account balance", "my current balance", "how much I currently owe", "my outstanding balance"],
    },
}


def _build_intents() -> dict[str, list[str]]:
    intents: dict[str, list[str]] = {}
    for intent_id, slots in _INTENT_SLOTS.items():
        variants = [
            f"{opener} {action}?" for opener in slots["openers"] for action in slots["actions"]
        ]
        intents[intent_id] = variants
    return intents


INTENTS: dict[str, list[str]] = _build_intents()


def zipf_weights(n: int, s: float = 1.1) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()


def generate_requests(n_requests: int, seed: int) -> list[tuple[str, str]]:
    """Returns [(intent_id, question_text), ...] — same sequence reused for
    every config so the comparison is apples-to-apples."""
    rng = np.random.default_rng(seed)
    intent_ids = list(INTENTS.keys())
    rng.shuffle(intent_ids)  # so "hot" intents aren't just alphabetically first
    weights = zipf_weights(len(intent_ids))

    requests = []
    for _ in range(n_requests):
        intent_id = rng.choice(intent_ids, p=weights)
        variants = INTENTS[intent_id]
        text = variants[rng.integers(0, len(variants))]
        requests.append((intent_id, text))
    return requests


@dataclass
class ConfigStats:
    name: str
    hits: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    wrong_answers: int = 0
    total_cost_usd: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def summary(self) -> str:
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        return (
            f"{self.name}:\n"
            f"  requests: {self.total}\n"
            f"  hit rate: {self.hits / self.total:.1%}  (exact={self.exact_hits}, semantic={self.semantic_hits})\n"
            f"  wrong-answer rate (of semantic hits): "
            f"{(self.wrong_answers / self.semantic_hits) if self.semantic_hits else 0:.1%} "
            f"({self.wrong_answers}/{self.semantic_hits})\n"
            f"  total cost: ${self.total_cost_usd:.4f}\n"
            f"  latency p50/p95: {np.percentile(lat, 50):.3f}s / {np.percentile(lat, 95):.3f}s\n"
        )


def run_config(
    name: str,
    requests: list[tuple[str, str]],
    read_threshold: float | None,
) -> ConfigStats:
    """read_threshold: None = use the real configured semantic threshold +
    conflict guard (the actual shipped behavior). A threshold > 1.0 makes
    the semantic tier structurally unable to match anything, simulating an
    exact-match-only cache without duplicating any cache logic."""
    cache = SemanticCache()
    namespace = f"bench-{name}"
    cache.flush_namespace(namespace)
    origin = StubOrigin()

    response_to_intent: dict[str, str] = {}
    stats = ConfigStats(name=name)

    for true_intent, text in requests:
        lookup_start = time.perf_counter()
        result = cache.get(namespace, text, threshold=read_threshold)
        lookup_latency = time.perf_counter() - lookup_start

        if result.hit:
            stats.hits += 1
            if result.tier == "exact":
                stats.exact_hits += 1
            else:
                stats.semantic_hits += 1
                served_intent = response_to_intent.get(result.response)
                if served_intent is not None and served_intent != true_intent:
                    stats.wrong_answers += 1
            stats.latencies.append(lookup_latency)
        else:
            stats.misses += 1
            origin_result = origin.generate(text)
            cache.set(namespace, text, origin_result.text)
            response_to_intent[origin_result.text] = true_intent
            cost = call_cost_usd(text, origin_result.text)
            stats.total_cost_usd += cost
            stats.latencies.append(lookup_latency + origin_result.latency_seconds)

    cache.flush_namespace(namespace)
    return stats


def run_no_cache(requests: list[tuple[str, str]]) -> ConfigStats:
    origin = StubOrigin()
    stats = ConfigStats(name="no_cache")
    for _, text in requests:
        stats.misses += 1
        origin_result = origin.generate(text)
        stats.total_cost_usd += call_cost_usd(text, origin_result.text)
        stats.latencies.append(origin_result.latency_seconds)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-requests", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    requests = generate_requests(args.n_requests, args.seed)

    no_cache = run_no_cache(requests)
    exact_only = run_config("exact_only", requests, read_threshold=1.01)
    semantic_tuned = run_config("semantic_tuned", requests, read_threshold=None)

    lines = [
        f"n_requests={args.n_requests} seed={args.seed} "
        f"n_intents={len(INTENTS)} (includes 4 deliberately confusable pairs)\n",
        no_cache.summary(),
        exact_only.summary(),
        semantic_tuned.summary(),
    ]

    for cfg in (exact_only, semantic_tuned):
        saved = no_cache.total_cost_usd - cfg.total_cost_usd
        pct = saved / no_cache.total_cost_usd if no_cache.total_cost_usd else 0
        lines.append(f"{cfg.name} saves ${saved:.4f} vs no-cache ({pct:.1%})\n")

    out = "\n".join(lines)
    print(out)
    (pathlib.Path(__file__).parent / "results" / "traffic_replay_summary.txt").write_text(out)


if __name__ == "__main__":
    main()
