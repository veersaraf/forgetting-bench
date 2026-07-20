"""Adapter wrapping the reference memory core."""

from __future__ import annotations

from ..memory import DecayModule, MemoryStore, NoDecay
from ..workload.synthetic import Observation, Query
from .base import MemoryAdapter, RetrievedItem


class ReferenceAdapter(MemoryAdapter):
    """Runs Forgetting-Bench's own :class:`MemoryStore` through the harness.

    The A/B experiment is just this adapter with two decay modules: ``NoDecay``
    (the incumbent-style baseline) and ``EbbinghausDecay`` (forgetting on).
    """

    def __init__(self, decay: DecayModule | None = None, name: str | None = None) -> None:
        self.store = MemoryStore(decay=decay or NoDecay())
        self.name = name or f"reference[{type(self.store.decay).__name__}]"
        # Benchmark ground-truth value, kept out of the memory core itself.
        self._value: dict[int, str | None] = {}

    def add(self, obs: Observation) -> None:
        entry = self.store.add(
            obs.text, obs.turn, entity=obs.entity, attribute=obs.attribute
        )
        self._value[entry.id] = obs.value

    def retrieve(self, query: Query, k: int) -> list[RetrievedItem]:
        return [
            RetrievedItem(
                text=s.entry.text,
                entity=s.entry.entity,
                attribute=s.entry.attribute,
                value=self._value.get(s.entry.id),
            )
            for s in self.store.retrieve(query.text, query.turn, k)
        ]

    def size(self) -> int:
        return self.store.size()

    def token_count(self) -> int:
        return self.store.token_count()
