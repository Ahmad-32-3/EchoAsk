"""A cheap, explainable second check applied AFTER a semantic similarity
hit, not instead of one. Built directly in response to the adversarial
audit (see loop/METRICS.md): mean-pooled word vectors cannot tell "same
sentence shape, different entity/number" apart from a true paraphrase, at
any similarity threshold. This catches the anticipated ways that happens in
a support-bot domain — differing numbers (order IDs, amounts), differing
members of a few known conflict groups (billing cadence, currency, role,
direction, support channel, temporal reference), and differing negation.

Explicitly NOT a general solution: it only catches conflicts in the
categories it's been told to watch. A novel entity swap outside these
groups (see the "paid plan" vs "support contract" pair in
eval/adversarial_audit.py, which this does NOT catch) sails through
untouched. The real fix for the general problem is a better embedder — see
DESIGN.md. This guard is a targeted patch for the failure modes we actually
observed, not a replacement for that fix.
"""
from __future__ import annotations

import re
from typing import Iterable

_DIGIT_RE = re.compile(r"\d+")
_NEGATION_RE = re.compile(r"\b(not|isn't|aren't|wasn't|weren't|cannot|can't|couldn't|no|never|n't)\b", re.IGNORECASE)

# Each group is a set of mutually-exclusive terms in a support-bot domain.
# If prompt A contains a term from a group and prompt B contains a
# *different* term from the same group, that's a conflict — the two prompts
# are about different things even though the surrounding sentence is
# near-identical.
_CONFLICT_GROUPS: list[set[str]] = [
    {"monthly", "annual", "yearly", "weekly", "quarterly"},
    {"student", "students", "teacher", "teachers", "employee", "employees",
     "admin", "admins", "guest", "guests", "member", "members"},
    {"upgrade", "upgrading", "downgrade", "downgrading", "increase", "decrease",
     "enable", "disable"},
    {"usd", "eur", "gbp", "cad", "aud", "dollar", "dollars", "euro", "euros", "pound", "pounds"},
    {"email", "phone", "chat", "sms"},
    {"this", "next", "last", "previous", "current"},
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _digits(text: str) -> set[str]:
    return set(_DIGIT_RE.findall(text))


def _negation_present(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def has_conflict(prompt_a: str, prompt_b: str) -> bool:
    """True if the two prompts look similar but structurally disagree on
    something that should block treating them as the same cached answer."""
    digits_a, digits_b = _digits(prompt_a), _digits(prompt_b)
    if digits_a and digits_b and digits_a != digits_b:
        return True

    toks_a, toks_b = _tokens(prompt_a), _tokens(prompt_b)
    for group in _CONFLICT_GROUPS:
        hits_a = toks_a & group
        hits_b = toks_b & group
        if hits_a and hits_b and hits_a != hits_b:
            return True

    if _negation_present(prompt_a) != _negation_present(prompt_b):
        return True

    return False
