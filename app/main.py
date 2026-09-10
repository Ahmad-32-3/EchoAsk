"""FastAPI proxy: a drop-in-shaped front for a chat-completions-style API.

POST /v1/chat/completions with {"namespace": "...", "messages": [...]} (or a
bare "prompt") checks the cache first (exact, then semantic) and only calls
the origin on a genuine miss. `namespace` scopes the cache — a tenant, a
session, whatever boundary must never leak across; requests without one are
rejected rather than silently pooled into a shared default, since a silent
default is exactly how cross-tenant leakage happens by accident.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.cache import SemanticCache
from app.cost_model import call_cost_usd
from app.origin import get_origin

app = FastAPI(title="semantic-cache")
cache = SemanticCache()
origin = get_origin()


class ChatRequest(BaseModel):
    # Restricted charset, not just min_length: an independent review found
    # that a namespace containing Redis glob metacharacters (notably "*")
    # could read or flush other tenants' cached data through the semantic
    # tier's SCAN MATCH pattern. This pattern is enforced again, defensively,
    # in app/cache.py itself (SemanticCache methods don't trust callers to
    # only ever go through this API), so the same protection holds for any
    # other caller (the eval scripts, a future second endpoint, etc).
    namespace: str = Field(
        ...,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Tenant/session scope. Required. Alphanumeric, '_' and '-' only.",
    )
    prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    cache_hit: bool
    cache_tier: str | None
    similarity: float | None
    latency_seconds: float
    cost_usd: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest):
    # Pydantic's pattern constraint on ChatRequest.namespace already rejects
    # a malformed namespace with a 422 before this runs; SemanticCache also
    # validates independently (see app/cache.py) so the same protection
    # holds for any caller that bypasses this Pydantic model. This try/except
    # is defense-in-depth for that second layer, mapped to a clean 400
    # instead of a raw 500 if it's ever the one that catches something.
    try:
        start = time.perf_counter()
        result = cache.get(req.namespace, req.prompt)

        if result.hit:
            latency = time.perf_counter() - start
            return ChatResponse(
                response=result.response,
                cache_hit=True,
                cache_tier=result.tier,
                similarity=result.similarity,
                latency_seconds=latency,
                cost_usd=0.0,
            )

        origin_result = origin.generate(req.prompt)
        cache.set(req.namespace, req.prompt, origin_result.text)
        latency = time.perf_counter() - start + origin_result.latency_seconds
        cost = call_cost_usd(req.prompt, origin_result.text)

        return ChatResponse(
            response=origin_result.text,
            cache_hit=False,
            cache_tier=None,
            similarity=None,
            latency_seconds=latency,
            cost_usd=cost,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
