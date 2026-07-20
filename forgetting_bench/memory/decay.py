"""Pluggable decay / consolidation -- the axis incumbent memory libraries ignore.

A :class:`DecayModule` decides, for every entry, a *strength* in [0, 1] that
multiplies its retrieval score, whether a newer entry *supersedes* it, and
whether it should be *pruned* from the stream entirely (bounding memory growth).

Swap the module to change forgetting behaviour without touching the store. The
two shipped implementations are the A/B arms of the headline experiment:

* :class:`NoDecay`   -- the incumbent baseline: nothing is ever forgotten.
* :class:`EbbinghausDecay` -- time decay + importance-weighted consolidation +
  contradiction supersession + strength-threshold pruning.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .entry import MemoryEntry


class DecayModule(ABC):
    @abstractmethod
    def strength(self, entry: MemoryEntry, now: int) -> float:
        """Multiplicative *ranking* weight in [0, 1] applied to the retrieval score.

        This is how a memory is down-ranked while it still lives -- chiefly to
        suppress contradicted (superseded) facts. It is deliberately *not* the
        time-decay curve: recency is already a scoring term, and multiplying the
        score by time-decay too would let a fresh irrelevant memory outrank an
        old but still-correct one. Time decay governs *forgetting* (pruning),
        not ranking.
        """

    @abstractmethod
    def retention(self, entry: MemoryEntry, now: int) -> float:
        """Ebbinghaus retention in [0, 1] -- the probability the memory survives.

        Drives pruning: forgetting is binary (a memory is kept or dropped), and
        this curve decides when it drops.
        """

    @abstractmethod
    def on_add(self, new_entry: MemoryEntry, existing: Iterable[MemoryEntry]) -> None:
        """Hook fired when ``new_entry`` is added, for supersession bookkeeping."""

    @abstractmethod
    def should_prune(self, entry: MemoryEntry, now: int) -> bool:
        """Whether ``entry`` has decayed enough to drop from the stream."""


class NoDecay(DecayModule):
    """The incumbent baseline. Every memory is retained at full strength forever."""

    def strength(self, entry: MemoryEntry, now: int) -> float:
        return 1.0

    def retention(self, entry: MemoryEntry, now: int) -> float:
        return 1.0

    def on_add(self, new_entry: MemoryEntry, existing: Iterable[MemoryEntry]) -> None:
        return None

    def should_prune(self, entry: MemoryEntry, now: int) -> bool:
        return False


class LastWriteWins(DecayModule):
    """A dedup baseline modelling what mem0-style fact memory already does.

    On each add, a newer entry about the same *extracted* slot supersedes the
    older one, which is then pruned immediately -- so at most one entry per slot
    survives (last write wins). There is no time decay, so it perfectly handles
    contradictions it can *link*, but it (a) never forgets anything it can't slot
    (distractor noise accumulates forever -> unbounded bloat) and (b) misses
    exactly the paraphrased / implicit contradictions the extractor misses. This
    is the honest bar the Ebbinghaus module has to beat.
    """

    def strength(self, entry: MemoryEntry, now: int) -> float:
        return 1.0

    def retention(self, entry: MemoryEntry, now: int) -> float:
        return 0.0 if entry.superseded_by is not None else 1.0

    def on_add(self, new_entry: MemoryEntry, existing: Iterable[MemoryEntry]) -> None:
        slot = new_entry.slot
        if slot is None:
            return
        for e in existing:
            if e.id != new_entry.id and e.slot == slot and e.superseded_by is None:
                e.superseded_by = new_entry.id

    def should_prune(self, entry: MemoryEntry, now: int) -> bool:
        return entry.superseded_by is not None


class EbbinghausDecay(DecayModule):
    """Ebbinghaus forgetting curve with importance-weighted consolidation.

    Retention of an entry ``t`` turns after it was written is::

        R(t) = exp(-t / S)      where  S = tau * (1 + importance_gain * importance)

    ``S`` is the memory's *stability*: important memories consolidate (larger
    ``S``, slower forgetting), echoing spaced-repetition models. Two mechanisms
    sit on top of the curve:

    * **Supersession** (ranking). When a newer entry about the same
      (entity, attribute) slot arrives, older entries for that slot are flagged
      ``superseded_by`` and their *ranking strength* is multiplied by
      ``supersession_factor`` (near-zero). This is what pushes stale,
      contradicted facts out of retrieval before they are eventually pruned.
    * **Pruning** (forgetting). Once retention drops below ``prune_threshold`` the
      entry is dropped from the stream. Low-importance memories (noise) and
      superseded facts fall past the threshold soonest, so memory stays bounded
      instead of growing linearly -- at the cost of eventually forgetting old
      facts that were never refreshed. That cost is the tradeoff the benchmark
      measures; ``tau`` dials it.
    """

    def __init__(
        self,
        tau: float = 200.0,
        importance_gain: float = 4.0,
        supersession_factor: float = 0.05,
        prune_threshold: float = 0.02,
    ) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        if not 0.0 <= supersession_factor <= 1.0:
            raise ValueError("supersession_factor must be in [0, 1]")
        if not 0.0 <= prune_threshold < 1.0:
            raise ValueError("prune_threshold must be in [0, 1)")
        self.tau = tau
        self.importance_gain = importance_gain
        self.supersession_factor = supersession_factor
        self.prune_threshold = prune_threshold

    def stability(self, entry: MemoryEntry) -> float:
        return self.tau * (1.0 + self.importance_gain * entry.importance)

    def strength(self, entry: MemoryEntry, now: int) -> float:
        # Ranking weight: full unless the fact has been contradicted.
        return self.supersession_factor if entry.superseded_by is not None else 1.0

    def retention(self, entry: MemoryEntry, now: int) -> float:
        elapsed = max(0, now - entry.turn)
        r = math.exp(-elapsed / self.stability(entry))
        # A contradicted fact is far more forgettable than its age implies.
        if entry.superseded_by is not None:
            r *= self.supersession_factor
        return r

    def on_add(self, new_entry: MemoryEntry, existing: Iterable[MemoryEntry]) -> None:
        slot = new_entry.slot
        if slot is None:
            return
        for e in existing:
            if e.id != new_entry.id and e.slot == slot and e.superseded_by is None:
                e.superseded_by = new_entry.id

    def should_prune(self, entry: MemoryEntry, now: int) -> bool:
        return self.retention(entry, now) < self.prune_threshold
