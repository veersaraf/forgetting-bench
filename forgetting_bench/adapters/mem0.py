"""Live mem0 adapter -- calls the real ``mem0`` SDK, not a re-implementation.

Construction fails with :class:`AdapterUnavailable` unless a backend can
actually be opened:

* ``MEM0_API_KEY`` -> ``MemoryClient`` (Mem0 platform)
* else ``OPENAI_API_KEY`` + installed ``mem0ai`` -> OSS ``Memory()``

``make bench`` never reports a mem0 row unless this adapter was constructed
and completed a harness run. There is no fallback score.
"""

from __future__ import annotations

import os
from typing import Any

from ..workload.synthetic import Observation, Query
from .base import MemoryAdapter, RetrievedItem
from .errors import AdapterUnavailable
from .probe import mem0_probe
from ._compat import (
    result_rows,
    row_id,
    row_metadata,
    row_text,
    whitespace_tokens,
)


def _truth_from_obs(obs: Observation) -> dict[str, Any]:
    return {
        "entity": obs.entity,
        "attribute": obs.attribute,
        "value": obs.value,
        "turn": obs.turn,
    }


def _item_from_row(row: Any, sidecar: dict[str, tuple[str | None, str | None, str | None]]) -> RetrievedItem:
    meta = row_metadata(row)
    mid = row_id(row)
    if mid and mid in sidecar:
        entity, attribute, value = sidecar[mid]
    else:
        entity = meta.get("entity")
        attribute = meta.get("attribute")
        value = meta.get("value")
    return RetrievedItem(row_text(row), entity, attribute, value)


class Mem0Adapter(MemoryAdapter):
    name = "mem0"

    def __init__(
        self,
        user_id: str = "forgetting-bench",
        client: Any | None = None,
        infer: bool = True,
    ) -> None:
        self.user_id = user_id
        self.infer = infer
        # Ground-truth sidecar -- scoring only. Never fed back into mem0 search.
        self._truth: dict[str, tuple[str | None, str | None, str | None]] = {}
        self._mode = "injected"
        if client is not None:
            self.client = client
            return
        probe = mem0_probe()
        if not probe.runnable:
            raise AdapterUnavailable("mem0", probe.reason)
        self.client, self._mode = self._open_client()

    @staticmethod
    def _open_client() -> tuple[Any, str]:
        if os.environ.get("MEM0_API_KEY"):
            try:
                from mem0 import MemoryClient
            except ImportError as exc:
                raise AdapterUnavailable("mem0", "mem0ai is not installed") from exc
            return MemoryClient(api_key=os.environ["MEM0_API_KEY"]), "platform"
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise AdapterUnavailable("mem0", "mem0ai is not installed") from exc
        return Memory(), "oss"

    def add(self, obs: Observation) -> None:
        metadata = _truth_from_obs(obs)
        payload = self._call_add(obs.text, metadata)
        for row in result_rows(payload) or [payload]:
            mid = row_id(row)
            if mid:
                self._truth[mid] = (obs.entity, obs.attribute, obs.value)

    def _call_add(self, text: str, metadata: dict[str, Any]) -> Any:
        # Prefer the current SDK (text + user_id + metadata). Fall back to the
        # messages form some releases require.
        attempts = (
            dict(user_id=self.user_id, metadata=metadata, infer=self.infer),
            dict(user_id=self.user_id, metadata=metadata),
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return self.client.add(text, **kwargs)
            except TypeError as exc:
                last_error = exc
        messages = [{"role": "user", "content": text}]
        try:
            return self.client.add(messages, user_id=self.user_id, metadata=metadata, infer=self.infer)
        except TypeError:
            return self.client.add(messages, user_id=self.user_id, metadata=metadata)
        except Exception:
            if last_error is not None:
                raise last_error
            raise

    def retrieve(self, query: Query, k: int) -> list[RetrievedItem]:
        payload = self._call_search(query.text, k)
        return [_item_from_row(row, self._truth) for row in result_rows(payload)[:k]]

    def _call_search(self, text: str, k: int) -> Any:
        filters = {"user_id": self.user_id}
        attempts = (
            dict(filters=filters, top_k=k),
            dict(filters=filters, limit=k),
            dict(user_id=self.user_id, limit=k),
            dict(user_id=self.user_id, top_k=k),
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return self.client.search(text, **kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise AdapterUnavailable("mem0", "client.search() rejected every known signature")

    def _all_rows(self) -> list[Any]:
        filters = {"user_id": self.user_id}
        attempts = (
            dict(filters=filters, top_k=100_000),
            dict(filters=filters, limit=100_000),
            dict(user_id=self.user_id),
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return result_rows(self.client.get_all(**kwargs))
            except (TypeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return []

    def size(self) -> int:
        return len(self._all_rows())

    def token_count(self) -> int:
        return sum(whitespace_tokens(row_text(row)) for row in self._all_rows())
