"""Central config. Everything overridable via env vars; sane local defaults."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # spaCy's en_core_web_md (300-dim GloVe-family vectors) — see
    # app/embeddings.py for why this replaced the originally-planned
    # fastembed/bge-small.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "en_core_web_md")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "300"))

    # Similarity threshold for a semantic cache hit. Set from
    # eval/threshold_sweep.py's real QQP results — see loop/METRICS.md.
    # IMPORTANT, read before reusing this default: no threshold in the sweep
    # (0.50-0.99) reaches a 97%-precision safety bar with this project's
    # embedding backend (spaCy word-vector averaging — see DESIGN.md for
    # why). 0.90 is a demo-illustrative operating point, not a
    # production-safe one: at this threshold, measured precision on QQP is
    # ~46%. Shipping this for real traffic needs a stronger sentence encoder
    # (the natural fix — swap the one embed() call in embeddings.py) and/or
    # a narrower query domain than open-domain QQP.
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

    # TTLs, in seconds, by query classification (see app/cache.py classify_query).
    ttl_default_seconds: int = int(os.getenv("TTL_DEFAULT_SECONDS", str(6 * 3600)))
    ttl_time_sensitive_seconds: int = int(os.getenv("TTL_TIME_SENSITIVE_SECONDS", "120"))

    # Origin backend: "stub" (deterministic, free, used for eval/benchmarks)
    # or "ollama" (real local model, used for the one live demo).
    origin_backend: str = os.getenv("ORIGIN_BACKEND", "stub")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    # Fake per-1K-token pricing used only by the cost model for benchmark
    # reporting, set from a real published rate (GPT-4o-mini as of 2025) so
    # the $ numbers in the case study mean something even though no real
    # billed call is made during the benchmark.
    price_per_1k_input_tokens_usd: float = float(os.getenv("PRICE_IN_PER_1K", "0.00015"))
    price_per_1k_output_tokens_usd: float = float(os.getenv("PRICE_OUT_PER_1K", "0.0006"))


settings = Settings()
