"""EXAMPLE STUB -- not a working mem0 integration.

This file exists to show *exactly* how an incumbent memory library plugs into the
harness. It is intentionally not wired to the real ``mem0`` package: the MVP
takes no hard dependency on it. To make it real, install ``mem0ai``, fill in the
three marked spots, and drop it into any harness run alongside ``ReferenceAdapter``.

The one non-obvious trick is metadata round-tripping: the benchmark needs each
retrieved memory's (entity, attribute, value) to score contradictions, so we
stash those on write as mem0 metadata and read them back on retrieve.
"""

from __future__ import annotations

from ..workload.synthetic import Observation, Query
from .base import MemoryAdapter, RetrievedItem


class Mem0Adapter(MemoryAdapter):
    name = "mem0"

    def __init__(self, user_id: str = "forgetting-bench") -> None:
        raise NotImplementedError(
            "Mem0Adapter is a documented stub. To enable it: `pip install mem0ai`, "
            "then implement the three TODOs below and remove this raise."
        )
        # from mem0 import Memory                      # noqa: E501  (TODO 1: import)
        # self.client = Memory()
        # self.user_id = user_id

    def add(self, obs: Observation) -> None:
        raise NotImplementedError
        # TODO 2: persist text + benchmark metadata so it round-trips on retrieve.
        # self.client.add(
        #     obs.text,
        #     user_id=self.user_id,
        #     metadata={"entity": obs.entity, "attribute": obs.attribute,
        #               "value": obs.value, "turn": obs.turn},
        # )

    def retrieve(self, query: Query, k: int) -> list[RetrievedItem]:
        raise NotImplementedError
        # TODO 3: map mem0 hits back to RetrievedItem via the stored metadata.
        # hits = self.client.search(query.text, user_id=self.user_id, limit=k)
        # return [
        #     RetrievedItem(h["memory"], h["metadata"].get("entity"),
        #                   h["metadata"].get("attribute"), h["metadata"].get("value"))
        #     for h in hits["results"]
        # ]

    def size(self) -> int:
        raise NotImplementedError

    def token_count(self) -> int:
        raise NotImplementedError
