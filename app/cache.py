"""Two-tier cache: exact-match (cheap, zero false-positive risk) then
semantic (embedding similarity). Namespaced so one tenant/session can never
see another's cached answer, and TTL differentiated by query type.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis

from app.config import settings
from app.conflict_guard import has_conflict
from app.embeddings import embed_one

_TIME_SENSITIVE_PATTERNS = re.compile(
    r"\b(today|current(ly)?|now|latest|this (week|month|year)|"
    r"right now|as of|up[- ]to[- ]date)\b",
    re.IGNORECASE,
)

# Found by an independent review pass, not by anyone testing for it up front:
# `namespace` is used to build a Redis SCAN MATCH pattern (see
# _semantic_scan_pattern / flush_namespace). Redis glob syntax treats
# * ? [ ] \ specially, so a caller passing namespace="*" turned
# f"sc:{namespace}:*" into "sc:*:*" — matching every tenant's semantic-tier
# keys, not just their own. Verified live: a namespace of "*" could read (and
# flush_namespace could wipe) every other namespace's cached data. This
# directly contradicted this project's own headline correctness claim
# ("namespace isolation") until this validation was added. Restricting
# namespace to a safe charset closes it structurally rather than trying to
# escape glob metacharacters case by case.
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_\-]{1,200}$")


def _validate_namespace(namespace: str) -> None:
    if not _NAMESPACE_RE.match(namespace):
        raise ValueError(
            f"invalid namespace {namespace!r}: must match ^[A-Za-z0-9_-]{{1,200}}$ "
            "(this also blocks Redis glob metacharacters — *, ?, [, ], \\ — from "
            "being used to scan or flush across other namespaces)"
        )


def classify_query(prompt: str) -> str:
    """Heuristic classification used to pick a TTL. Not a design centerpiece —
    a real system would want a small trained classifier or an allowlist of
    known-volatile intents. This is intentionally simple and named as such.
    """
    return "time_sensitive" if _TIME_SENSITIVE_PATTERNS.search(prompt) else "default"


def _normalize(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def _exact_key(namespace: str, prompt: str) -> str:
    digest = hashlib.sha256(_normalize(prompt).encode("utf-8")).hexdigest()
    return f"exact:{namespace}:{digest}"


def _semantic_key(namespace: str, entry_id: str) -> str:
    return f"sc:{namespace}:{entry_id}"


def _semantic_scan_pattern(namespace: str) -> str:
    return f"sc:{namespace}:*"


def _ttl_for(query_class: str) -> int:
    return (
        settings.ttl_time_sensitive_seconds
        if query_class == "time_sensitive"
        else settings.ttl_default_seconds
    )


@dataclass
class CacheResult:
    hit: bool
    tier: Optional[str]  # "exact" | "semantic" | None
    response: Optional[str]
    similarity: Optional[float] = None


class SemanticCache:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.r = redis_client or redis.from_url(settings.redis_url, decode_responses=False)

    # ---- write path ----
    def set(self, namespace: str, prompt: str, response: str) -> None:
        _validate_namespace(namespace)
        query_class = classify_query(prompt)
        ttl = _ttl_for(query_class)

        # Exact tier
        self.r.setex(_exact_key(namespace, prompt), ttl, response)

        # Semantic tier
        vector = embed_one(prompt)
        entry_id = hashlib.sha256(
            f"{namespace}:{_normalize(prompt)}:{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:16]
        key = _semantic_key(namespace, entry_id)
        self.r.hset(
            key,
            mapping={
                "prompt": prompt,
                "response": response,
                "vector": vector.astype(np.float32).tobytes(),
                "query_class": query_class,
                "created_at": str(time.time()),
            },
        )
        self.r.expire(key, ttl)

    # ---- read path ----
    def get(self, namespace: str, prompt: str, threshold: Optional[float] = None) -> CacheResult:
        """threshold overrides the configured similarity threshold for this
        call only — used by eval/traffic_replay.py to simulate an
        exact-match-only cache (threshold > 1.0, structurally unmatchable)
        without duplicating any cache logic. Leave as None for real traffic.
        """
        _validate_namespace(namespace)
        exact_hit = self.r.get(_exact_key(namespace, prompt))
        if exact_hit is not None:
            return CacheResult(hit=True, tier="exact", response=exact_hit.decode("utf-8"), similarity=1.0)

        best = self._semantic_search(namespace, prompt, threshold=threshold)
        if best is not None:
            response, similarity = best
            return CacheResult(hit=True, tier="semantic", response=response, similarity=similarity)

        return CacheResult(hit=False, tier=None, response=None)

    def _semantic_search(self, namespace: str, prompt: str, threshold: Optional[float] = None):
        """Return the highest-similarity cached entry at/above threshold that
        also clears the conflict guard — not just the single highest-scoring
        entry overall. A near-1.0 similarity candidate that conflicts (see
        app/conflict_guard.py) is skipped in favor of the next-best candidate
        that doesn't, rather than causing a miss outright.
        """
        threshold = settings.similarity_threshold if threshold is None else threshold
        query_vec = embed_one(prompt)

        candidates: list[tuple[float, str, str]] = []  # (score, response, stored_prompt)
        cursor = 0
        pattern = _semantic_scan_pattern(namespace)
        while True:
            cursor, keys = self.r.scan(cursor=cursor, match=pattern, count=200)
            for key in keys:
                data = self.r.hgetall(key)
                if not data or b"vector" not in data:
                    continue
                vec = np.frombuffer(data[b"vector"], dtype=np.float32)
                score = float(np.dot(query_vec, vec))  # both are L2-normalized
                if score >= threshold:
                    candidates.append(
                        (score, data[b"response"].decode("utf-8"), data[b"prompt"].decode("utf-8"))
                    )
            if cursor == 0:
                break

        candidates.sort(key=lambda c: c[0], reverse=True)
        for score, response, stored_prompt in candidates:
            if not has_conflict(prompt, stored_prompt):
                return response, score
        return None

    # ---- test/introspection helpers ----
    def flush_namespace(self, namespace: str) -> None:
        _validate_namespace(namespace)
        for pattern in (f"exact:{namespace}:*", _semantic_scan_pattern(namespace)):
            cursor = 0
            while True:
                cursor, keys = self.r.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    self.r.delete(*keys)
                if cursor == 0:
                    break

    def count_semantic_entries(self, namespace: str) -> int:
        _validate_namespace(namespace)
        cursor = 0
        n = 0
        while True:
            cursor, keys = self.r.scan(cursor=cursor, match=_semantic_scan_pattern(namespace), count=200)
            n += len(keys)
            if cursor == 0:
                break
        return n
