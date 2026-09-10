"""The 'origin LLM' the proxy sits in front of.

Two backends, used for two different purposes — this split, and why it
exists, is itself a design decision worth stating rather than hiding:

- StubOrigin: deterministic, free, instant to run thousands of times. Used
  for every benchmark/eval number in this project (threshold sweep, traffic
  replay, adversarial audit). A real LLM would make those numbers slow to
  produce, non-reproducible (model output drifts run to run), and would cost
  real money every time a threshold changes and the eval re-runs.
- OllamaOrigin: a real local model, used exactly once, for a live
  end-to-end demo proving the proxy genuinely calls out on a cache miss.
  Not used for measurement.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import settings


@dataclass
class OriginResponse:
    text: str
    latency_seconds: float


class Origin(Protocol):
    def generate(self, prompt: str) -> OriginResponse: ...


class StubOrigin:
    """Deterministic canned responses with a realistic, reproducible fake
    latency. Seeded per-prompt so re-running the same traffic replay produces
    identical numbers, which is what makes the benchmark trustworthy to
    re-check.
    """

    _TEMPLATES = [
        "Here's how to handle that: {detail}. Let me know if you need the next step.",
        "Sure — {detail}. That should resolve it.",
        "To do that: {detail}. If it doesn't work, there may be a follow-up step.",
        "{detail} is the short answer. Happy to go deeper if needed.",
    ]

    def _rng_for(self, prompt: str) -> random.Random:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(seed)

    def generate(self, prompt: str) -> OriginResponse:
        rng = self._rng_for(prompt)
        template = self._TEMPLATES[rng.randrange(len(self._TEMPLATES))]
        detail_words = max(6, len(prompt.split()) * 2)
        detail = " ".join(prompt.split()[:1] or ["this"]) + " " + " ".join(
            rng.choice(["step", "setting", "option", "field", "menu", "link"])
            for _ in range(detail_words)
        )
        text = template.format(detail=detail)

        # Realistic hosted-LLM latency: lognormal, ~0.5-2.5s typical, longer tail.
        latency = min(6.0, rng.lognormvariate(mu=-0.35, sigma=0.5))
        return OriginResponse(text=text, latency_seconds=latency)


class OllamaOrigin:
    """Real local model via Ollama. Used only for the single live-demo call."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.ollama_url
        self.model = model or settings.ollama_model

    def generate(self, prompt: str) -> OriginResponse:
        start = time.perf_counter()
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
        latency = time.perf_counter() - start
        return OriginResponse(text=data.get("response", ""), latency_seconds=latency)


def get_origin() -> Origin:
    if settings.origin_backend == "ollama":
        return OllamaOrigin()
    return StubOrigin()
