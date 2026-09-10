"""Local embedding backend. No network calls at runtime, no API key, no
per-call cost, fully offline once the model package is installed.

Originally planned as fastembed/bge-small (see DESIGN.md), but this sandbox
has no route to huggingface.co, which is where fastembed fetches model
weights from. Substituted spaCy's en_core_web_md: a pip-installable wheel
(fetched from a GitHub release, which this sandbox *can* reach) that bundles
300-dim GloVe-family word vectors directly — no runtime download at all,
better cold-start story than fastembed's lazy fetch-on-first-use. Sentence
embeddings here are the mean of token vectors, L2-normalized.

This is a real quality tradeoff, not a free substitution: averaged static
word vectors are weaker than a trained sentence encoder (like bge-small) at
distinguishing "similar wording" from "similar meaning" — see
loop/METRICS.md for the measured precision/recall this actually gets on
QQP, and DESIGN.md for the honest comparison.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    import spacy

    return spacy.load(settings.embedding_model, disable=["ner", "parser", "tagger", "lemmatizer"])


def embed(texts: Sequence[str]) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalized sentence embeddings
    (mean of token vectors)."""
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    nlp = _model()
    vectors = np.array([doc.vector for doc in nlp.pipe(texts)], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """a, b assumed already L2-normalized -> cosine similarity is a plain dot product."""
    return float(np.dot(a, b))
