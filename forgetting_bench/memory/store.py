"""The reference memory core.

A :class:`MemoryStore` is a stream of timestamped entries retrieved by the
generative-agents scoring trio -- **recency + importance + relevance** -- each
term bounded to [0, 1] and combined as a weighted sum, then multiplied by the
pluggable decay module's strength. Swapping the decay module (``NoDecay`` vs
``EbbinghausDecay``) is the whole A/B of the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

from .decay import DecayModule, NoDecay
from .embedder import Embedder, HashingEmbedder, cosine
from .entry import MemoryEntry
from .extractor import SlotExtractor
from .importance import HeuristicImportance, ImportanceScorer


@dataclass(frozen=True)
class RetrievalWeights:
    """Weights for the retrieval trio. Relevance leads so the right slot surfaces
    out of accumulated noise; recency breaks ties toward the current fact."""

    recency: float = 0.15
    importance: float = 0.15
    relevance: float = 0.7

    def total(self) -> float:
        return self.recency + self.importance + self.relevance


@dataclass
class ScoredEntry:
    entry: MemoryEntry
    score: float
    recency: float
    importance: float
    relevance: float
    strength: float


class MemoryStore:
    def __init__(
        self,
        embedder: Embedder | None = None,
        importance_scorer: ImportanceScorer | None = None,
        decay: DecayModule | None = None,
        weights: RetrievalWeights | None = None,
        recency_decay: float = 0.999,
        slot_extractor: SlotExtractor | None = None,
    ) -> None:
        if not 0.0 < recency_decay < 1.0:
            raise ValueError("recency_decay must be in (0, 1)")
        self.embedder = embedder or HashingEmbedder()
        self.importance_scorer = importance_scorer or HeuristicImportance()
        self.decay = decay or NoDecay()
        self.weights = weights or RetrievalWeights()
        self.recency_decay = recency_decay
        # Derives (entity, attribute) keys from text for supersession. Imperfect
        # by design (see extractor.py); may be None, in which case callers must
        # pass keys explicitly or slots go unlinked.
        self.slot_extractor = slot_extractor
        self._entries: list[MemoryEntry] = []
        self._next_id = 0

    # -- writing -----------------------------------------------------------

    def add(
        self,
        text: str,
        turn: int,
        *,
        entity: str | None = None,
        attribute: str | None = None,
        importance: float | None = None,
    ) -> MemoryEntry:
        """Append an observation and run decay bookkeeping (supersession, prune).

        If ``entity``/``attribute`` are not given and a ``slot_extractor`` is
        configured, the slot key is *extracted from the text* -- imperfectly, as
        a real system would. Whatever key results (possibly None) is what
        supersession keys on.
        """
        if entity is None and attribute is None and self.slot_extractor is not None:
            extracted = self.slot_extractor.extract(text)
            if extracted is not None:
                entity, attribute = extracted
        entry = MemoryEntry(
            id=self._next_id,
            turn=turn,
            text=text,
            importance=(
                importance
                if importance is not None
                else self.importance_scorer.score(text)
            ),
            entity=entity,
            attribute=attribute,
            embedding=self.embedder.embed(text),
            last_access=turn,
        )
        self._next_id += 1
        self.decay.on_add(entry, self._entries)
        self._entries.append(entry)
        self._prune(turn)
        return entry

    def _prune(self, now: int) -> None:
        self._entries = [
            e for e in self._entries if not self.decay.should_prune(e, now)
        ]

    # -- reading -----------------------------------------------------------

    def _recency(self, entry: MemoryEntry, now: int) -> float:
        return self.recency_decay ** max(0, now - entry.turn)

    def score_all(self, query: str, now: int) -> list[ScoredEntry]:
        qvec = self.embedder.embed(query)
        w = self.weights
        wt = w.total()
        scored: list[ScoredEntry] = []
        for e in self._entries:
            recency = self._recency(e, now)
            relevance = cosine(qvec, e.embedding)
            importance = e.importance
            base = (
                w.recency * recency
                + w.importance * importance
                + w.relevance * relevance
            ) / wt
            strength = self.decay.strength(e, now)
            scored.append(
                ScoredEntry(
                    entry=e,
                    score=base * strength,
                    recency=recency,
                    importance=importance,
                    relevance=relevance,
                    strength=strength,
                )
            )
        return scored

    def retrieve(self, query: str, now: int, k: int = 5) -> list[ScoredEntry]:
        """Return the top-``k`` entries by combined score (highest first).

        Ties break toward the *newer* entry (higher id), so when two versions of
        a slot score equally the current one wins -- deterministic, and the
        sensible default for a memory system.
        """
        scored = self.score_all(query, now)
        scored.sort(key=lambda s: (-s.score, -s.entry.id))
        return scored[:k]

    # -- introspection -----------------------------------------------------

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def size(self) -> int:
        """Number of live entries (post-pruning) -- the memory-bloat signal."""
        return len(self._entries)

    def token_count(self) -> int:
        """Total tokens held in the live stream -- the context-cost signal."""
        return sum(e.token_count() for e in self._entries)
