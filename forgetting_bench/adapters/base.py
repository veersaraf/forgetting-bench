"""The comparison-adapter seam.

Every memory system under test -- our reference core, or an incumbent like mem0 /
Letta / Zep -- is driven through the same :class:`MemoryAdapter` interface, so
the harness scores them identically. An adapter's job is to (1) ingest an
observation into its backend and (2) answer a query with a ranked list of
:class:`RetrievedItem`, carrying back the benchmark metadata (entity / attribute
/ value) needed to score contradictions and recall.

Real backends won't hand you (entity, attribute, value) for free -- you store it
as memory metadata on write and read it back on retrieve. See
``adapters/mem0_stub.py`` for how that maps onto an incumbent API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..workload.synthetic import Observation, Query


@dataclass
class RetrievedItem:
    """One memory returned for a query, with round-tripped benchmark metadata."""

    text: str
    entity: str | None
    attribute: str | None
    value: str | None


class MemoryAdapter(ABC):
    #: Human-readable label used in results tables and plot legends.
    name: str = "adapter"

    @abstractmethod
    def add(self, obs: Observation) -> None:
        """Ingest one observation into the backend."""

    @abstractmethod
    def retrieve(self, query: Query, k: int) -> list[RetrievedItem]:
        """Return up to ``k`` memories, most relevant first."""

    @abstractmethod
    def size(self) -> int:
        """Number of live memories held -- the memory-bloat signal."""

    @abstractmethod
    def token_count(self) -> int:
        """Total tokens held across live memories -- the context-cost signal."""
