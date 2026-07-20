"""Embedders provide the *relevance* term of the retrieval score.

The default :class:`HashingEmbedder` is dependency-free and deterministic: it
needs no API key and downloads no model, so the whole benchmark runs offline and
reproducibly. :class:`Embedder` is the seam for swapping in real sentence
embeddings (OpenAI, sentence-transformers, ...) without touching the store.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SparseVector:
    """A sparse embedding: nonzero buckets only, with a cached L2 norm.

    Texts here are short (a sentence), so the vectors are tiny; sparse dot
    products make retrieval over thousands of memories fast enough for a
    one-command benchmark, where a dense 1024-d cosine would not be.
    """

    data: dict[int, float] = field(default_factory=dict)
    norm: float = 0.0


class Embedder(ABC):
    """Maps text to a :class:`SparseVector`. Relevance = cosine similarity."""

    @abstractmethod
    def embed(self, text: str) -> SparseVector:
        ...


def cosine(a: SparseVector, b: SparseVector) -> float:
    """Cosine similarity of two sparse vectors. Returns 0.0 if either is empty."""
    if a.norm == 0.0 or b.norm == 0.0:
        return 0.0
    # Iterate the smaller vector for speed.
    if len(a.data) > len(b.data):
        a, b = b, a
    dot = sum(w * b.data.get(bucket, 0.0) for bucket, w in a.data.items())
    return dot / (a.norm * b.norm)


class HashingEmbedder(Embedder):
    """A deterministic hashed bag-of-words embedder with sublinear term weighting.

    Each token is hashed into one of ``dim`` buckets and accumulated with a
    ``1 + log(count)`` weight (the TF half of TF-IDF; no corpus IDF is needed
    because cosine already normalises for length). This is not semantic -- it is
    lexical overlap -- which is exactly right for a synthetic benchmark whose
    signal lives in shared entity/attribute words, and it keeps the demo free of
    heavyweight model dependencies.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> SparseVector:
        counts: dict[int, int] = {}
        for tok in tokenize(text):
            # Stable hash: Python's built-in hash() is salted per-process, so use
            # a fixed hashing scheme for reproducibility across runs.
            h = _stable_hash(tok) % self.dim
            counts[h] = counts.get(h, 0) + 1
        data = {bucket: 1.0 + math.log(c) for bucket, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in data.values()))
        return SparseVector(data=data, norm=norm)


def _stable_hash(token: str) -> int:
    """FNV-1a 32-bit hash -- deterministic across processes, unlike ``hash()``."""
    h = 0x811C9DC5
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h
