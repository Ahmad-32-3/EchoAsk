"""Hand-built pairs, scoped to this project's support-bot scenario, that are
worded very similarly but mean something materially different: a numeric
swap (monthly/annual), a negation, an entity swap, a scope swap
(one item/order/plan vs another). QQP proves the general problem exists;
this proves it exists concretely inside the exact domain this project
pretends to serve, which is the more damning (and more useful) version of
the same finding.

Exercises the REAL cache (app.cache.SemanticCache), not raw embedding
similarity — so this measures what actually happens after the threshold AND
the conflict guard (app/conflict_guard.py), not just the embedder's raw
signal. Requires a reachable Redis (REDIS_URL).

Usage: python -m eval.adversarial_audit
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.cache import SemanticCache  # noqa: E402
from app.config import settings  # noqa: E402

# (question_a, question_b, why_different)
ADVERSARIAL_PAIRS = [
    ("What's the refund policy for monthly plans?",
     "What's the refund policy for annual plans?",
     "numeric/plan-type swap — different policy, same sentence shape"),
    ("Can I cancel my subscription?",
     "Can I cancel my subscription and still get a refund?",
     "scope addition — second question needs a different, longer answer"),
    ("Do you offer a discount for students?",
     "Do you offer a discount for teachers?",
     "entity swap — different eligibility, near-identical wording"),
    ("Is my order #4521 delayed?",
     "Is my order #7788 delayed?",
     "order-id swap — different customer's data, must never share an answer"),
    ("How do I upgrade to the Pro plan?",
     "How do I downgrade from the Pro plan?",
     "negation-adjacent action swap — opposite intent"),
    ("What is the price of the annual plan in USD?",
     "What is the price of the annual plan in EUR?",
     "currency swap — different numeric answer"),
    ("Why isn't my payment going through?",
     "Why is my payment going through twice?",
     "negation/scope — opposite failure modes"),
    ("Can I use the API without a paid plan?",
     "Can I use the API without a support contract?",
     "entity swap — different gating condition"),
    ("What happens if I miss this month's payment?",
     "What happens if I miss next month's payment?",
     "time-reference swap — should classify time-sensitive, not be pooled"),
    ("Does the free tier include email support?",
     "Does the free tier include phone support?",
     "entity swap — different feature, same sentence shape"),
]


def run(cache: SemanticCache, namespace_prefix: str = "adv") -> tuple[list[str], int]:
    lines = []
    flagged = 0
    for i, (a, b, why) in enumerate(ADVERSARIAL_PAIRS):
        namespace = f"{namespace_prefix}-{i}"
        cache.flush_namespace(namespace)
        cache.set(namespace, a, f"[answer for: {a}]")
        result = cache.get(namespace, b)

        is_false_positive = result.hit and result.tier == "semantic"
        flagged += is_false_positive
        marker = "FALSE POSITIVE (served A's answer for B)" if is_false_positive else "correctly rejected"
        sim_note = f"sim={result.similarity:.3f}" if result.similarity is not None else "no candidate above threshold"
        lines.append(f"[{marker}] {sim_note}  ({why})\n  A: {a}\n  B: {b}\n")
        cache.flush_namespace(namespace)
    return lines, flagged


def main() -> None:
    cache = SemanticCache()
    lines, flagged = run(cache)

    header = (
        f"adversarial pairs: {len(ADVERSARIAL_PAIRS)}\n"
        f"threshold in use: {settings.similarity_threshold}\n"
        f"conflict guard: enabled (app/conflict_guard.py)\n"
        f"false positives (would have served the wrong answer): {flagged} / {len(ADVERSARIAL_PAIRS)}\n"
        f"false-positive rate: {flagged / len(ADVERSARIAL_PAIRS):.1%}\n\n"
    )
    out = header + "\n".join(lines)
    print(out)

    out_path = pathlib.Path(__file__).parent / "results" / "adversarial_summary.txt"
    out_path.write_text(out)


if __name__ == "__main__":
    main()
