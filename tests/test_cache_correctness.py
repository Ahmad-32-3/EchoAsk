"""Correctness tests for the cache tiering, namespace isolation, and TTL
policy. These are the tests a semantic cache absolutely cannot ship without:
if any of these fail, the cache is actively dangerous (serving one tenant's
answer to another) rather than merely suboptimal.

Requires a real Redis reachable at REDIS_URL (defaults to localhost:6379).
Each test flushes its own namespace so tests don't interfere with each other.
"""
import pytest

from app.cache import SemanticCache, classify_query, _ttl_for
from app.config import settings


@pytest.fixture
def cache():
    c = SemanticCache()
    yield c
    c.flush_namespace("test-a")
    c.flush_namespace("test-b")


def test_exact_hit_after_set(cache):
    cache.set("test-a", "How do I reset my password?", "Click forgot password.")
    result = cache.get("test-a", "How do I reset my password?")
    assert result.hit is True
    assert result.tier == "exact"
    assert result.response == "Click forgot password."
    assert result.similarity == 1.0


def test_semantic_hit_on_paraphrase(cache):
    cache.set("test-a", "How do I reset my password?", "Click forgot password.")
    # A differently-worded but same-intent question should hit the semantic
    # tier, not the exact tier (different literal string).
    result = cache.get("test-a", "I forgot my password, what do I do")
    assert result.hit is True
    assert result.tier == "semantic"
    assert result.similarity >= settings.similarity_threshold


def test_miss_on_unrelated_query(cache):
    cache.set("test-a", "How do I reset my password?", "Click forgot password.")
    result = cache.get("test-a", "What is the capital of France?")
    assert result.hit is False
    assert result.response is None


def test_exact_tier_checked_before_semantic(cache):
    """If both tiers would technically match, exact must win — it's cheaper
    and carries zero similarity-threshold risk. We can't directly observe
    which code path ran except via the reported tier, which is exactly the
    point of exposing `tier` on the result."""
    cache.set("test-a", "reset password", "answer A")
    result = cache.get("test-a", "reset password")
    assert result.tier == "exact"


def test_namespace_isolation_no_cross_tenant_leak(cache):
    """The single most important correctness property of this whole system:
    tenant A's cached answer must never be served to tenant B, even for the
    identical literal prompt."""
    cache.set("test-a", "what is my account balance", "Tenant A's private answer")
    result = cache.get("test-b", "what is my account balance")
    assert result.hit is False, "cross-namespace cache leak: tenant B saw tenant A's cached answer"


def test_namespace_wildcard_cannot_read_other_namespaces(cache):
    """Regression test for a real bug an independent review found: `namespace`
    is used to build a Redis SCAN MATCH pattern, and Redis glob syntax gives
    special meaning to * ? [ ] \\. A namespace of "*" used to expand
    f"sc:{namespace}:*" into "sc:*:*" — matching every tenant's data, not
    just the caller's own. This must be rejected outright, not merely
    'happen not to match' by luck of the data present."""
    cache.set("test-a", "what is my account balance", "Tenant A's private answer")
    with pytest.raises(ValueError):
        cache.get("*", "what is my account balance")


def test_namespace_rejects_glob_metacharacters(cache):
    for bad_namespace in ["*", "a*", "[ab]", "a?b", "a\\b", "sc:other:*", ""]:
        with pytest.raises(ValueError):
            cache.get(bad_namespace, "anything")


def test_flush_namespace_wildcard_cannot_wipe_other_namespaces(cache):
    cache.set("test-a", "q", "a")
    with pytest.raises(ValueError):
        cache.flush_namespace("*")
    # and, having been rejected, tenant A's data must still be there
    assert cache.get("test-a", "q").hit is True


def test_namespace_isolation_semantic_tier_too(cache):
    """Same property, but via the semantic tier with a paraphrased query —
    the leak risk is higher here since it's not a literal key match that
    could be scoped by accident-proof hashing alone."""
    cache.set("test-a", "what is my account balance", "Tenant A's private answer")
    result = cache.get("test-b", "can you tell me my account balance please")
    assert result.hit is False, "cross-namespace semantic leak"


def test_ttl_set_on_exact_key(cache):
    from app.cache import _exact_key

    cache.set("test-a", "some question", "some answer")
    ttl = cache.r.ttl(_exact_key("test-a", "some question"))
    assert ttl > 0, "exact cache entry should have a positive TTL, not persist forever"


def test_time_sensitive_query_gets_shorter_ttl():
    assert classify_query("what's the current status of my order") == "time_sensitive"
    assert classify_query("how do I reset my password") == "default"
    assert _ttl_for("time_sensitive") < _ttl_for("default")


def test_flush_namespace_removes_both_tiers(cache):
    cache.set("test-a", "q1", "a1")
    assert cache.count_semantic_entries("test-a") == 1
    cache.flush_namespace("test-a")
    assert cache.count_semantic_entries("test-a") == 0
    assert cache.get("test-a", "q1").hit is False
