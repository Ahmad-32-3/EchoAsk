"""A fake $ cost model for the benchmark/eval reporting, built on a real
published per-token rate (GPT-4o-mini pricing as of when this was written —
see config.py). No real billed call is ever made to produce these numbers;
this exists so the benchmark's "$ saved" figure means something concrete
instead of being an assertion.
"""
from __future__ import annotations

from app.config import settings

# Rough, deliberately simple token estimate (~4 chars/token for English).
# A real system would use the model's actual tokenizer; that precision isn't
# needed to make the cost-tradeoff point here.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def call_cost_usd(prompt: str, response: str) -> float:
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response)
    return (
        input_tokens / 1000 * settings.price_per_1k_input_tokens_usd
        + output_tokens / 1000 * settings.price_per_1k_output_tokens_usd
    )
