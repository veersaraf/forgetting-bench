"""Importance scoring -- the *importance* term of the retrieval score.

Generative-agents importance asks an LLM to rate an observation 1-10. We default
to a dependency-free heuristic so the benchmark runs offline, and expose
:class:`ImportanceScorer` as the seam for an LLM-backed scorer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .embedder import tokenize

# Words that tend to mark a durable, agent-relevant fact rather than idle chatter.
_SALIENT = frozenset(
    {
        "prefers",
        "favorite",
        "always",
        "never",
        "allergic",
        "birthday",
        "deadline",
        "password",
        "lives",
        "works",
        "married",
        "now",
        "changed",
        "moved",
        "switched",
        "important",
    }
)


class ImportanceScorer(ABC):
    """Rates how durable/salient an observation is, in [0, 1]."""

    @abstractmethod
    def score(self, text: str) -> float:
        ...


class HeuristicImportance(ImportanceScorer):
    """A transparent heuristic: salient-keyword hits plus a mild length prior.

    It deliberately does **not** look at the benchmark's ground-truth labels --
    it only reads the surface text, the same signal an LLM rater would get.
    """

    def __init__(self, base: float = 0.3) -> None:
        self.base = base

    def score(self, text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        hits = sum(1 for t in toks if t in _SALIENT)
        # Each salient marker adds weight; longer statements get a small bump.
        salience = min(1.0, self.base + 0.25 * hits)
        length_prior = min(0.2, 0.02 * len(toks))
        return min(1.0, salience + length_prior)
