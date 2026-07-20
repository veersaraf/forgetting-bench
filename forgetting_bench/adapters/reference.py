"""Adapter wrapping the reference memory core."""

from __future__ import annotations

from ..memory import DecayModule, MemoryStore, NoDecay, SlotExtractor
from ..workload.synthetic import Observation, Query, default_extractor
from .base import MemoryAdapter, RetrievedItem


class ReferenceAdapter(MemoryAdapter):
    """Runs Forgetting-Bench's own :class:`MemoryStore` through the harness.

    The store extracts its own slot keys from text via ``slot_extractor``
    (imperfect, by design) -- it is *not* handed ground-truth keys. The adapter
    separately remembers each observation's true (entity, attribute, value) and
    round-trips that to the harness for scoring, so the metric is measured
    against ground truth while the memory operates on what it could extract.
    """

    def __init__(
        self,
        decay: DecayModule | None = None,
        name: str | None = None,
        slot_extractor: SlotExtractor | None = None,
    ) -> None:
        self.store = MemoryStore(
            decay=decay or NoDecay(),
            slot_extractor=slot_extractor or default_extractor(),
        )
        self.name = name or type(self.store.decay).__name__
        # Ground-truth slot/value, kept out of the memory core entirely.
        self._truth: dict[int, tuple[str | None, str | None, str | None]] = {}

    def add(self, obs: Observation) -> None:
        entry = self.store.add(obs.text, obs.turn)  # store extracts its own keys
        self._truth[entry.id] = (obs.entity, obs.attribute, obs.value)

    def retrieve(self, query: Query, k: int) -> list[RetrievedItem]:
        items = []
        for s in self.store.retrieve(query.text, query.turn, k):
            entity, attribute, value = self._truth.get(s.entry.id, (None, None, None))
            items.append(RetrievedItem(s.entry.text, entity, attribute, value))
        return items

    def size(self) -> int:
        return self.store.size()

    def token_count(self) -> int:
        return self.store.token_count()
