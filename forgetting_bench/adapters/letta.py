"""Live Letta adapter -- archival memory via the real Letta SDK.

Drives ``agents.passages.insert`` / ``agents.passages.search`` (archival
memory), not a full conversational agent loop. Construction fails with
:class:`AdapterUnavailable` unless ``letta-client`` (or ``letta``) is
installed and ``LETTA_API_KEY`` or ``LETTA_BASE_URL`` is set.

``make bench`` never reports a Letta row unless this adapter was constructed
and completed a harness run.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from ..workload.synthetic import Observation, Query
from .base import MemoryAdapter, RetrievedItem
from .errors import AdapterUnavailable
from .probe import letta_probe
from ._compat import (
    result_rows,
    row_id,
    row_tags,
    row_text,
    whitespace_tokens,
)

_TAG_PREFIX = "fb:"


def _open_letta_client() -> Any:
    token = os.environ.get("LETTA_API_KEY")
    base_url = os.environ.get("LETTA_BASE_URL")
    last_error: Exception | None = None
    try:
        from letta_client import Letta

        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        if base_url:
            kwargs["base_url"] = base_url
        try:
            return Letta(**kwargs)
        except TypeError:
            # Older keyword: api_key
            if token:
                kwargs.pop("token", None)
                kwargs["api_key"] = token
            return Letta(**kwargs)
    except ImportError as exc:
        last_error = exc
    try:
        from letta import create_client

        if base_url:
            return create_client(base_url=base_url, token=token)
        return create_client(token=token)
    except ImportError as exc:
        last_error = exc
    except TypeError:
        from letta import create_client

        return create_client()
    raise AdapterUnavailable(
        "letta",
        f"could not construct a Letta client ({last_error})",
    )


class LettaAdapter(MemoryAdapter):
    name = "letta"

    def __init__(
        self,
        client: Any | None = None,
        agent_id: str | None = None,
        passages: Any | None = None,
    ) -> None:
        self._truth: dict[str, tuple[str | None, str | None, str | None]] = {}
        self._by_text: dict[str, tuple[str | None, str | None, str | None]] = {}
        self._texts: list[str] = []
        if client is not None and passages is not None:
            self.client = client
            self.agent_id = agent_id or "injected"
            self._passages = passages
            return
        probe = letta_probe()
        if not probe.runnable:
            raise AdapterUnavailable("letta", probe.reason)
        self.client = _open_letta_client()
        self.agent_id = agent_id or os.environ.get("LETTA_AGENT_ID") or self._create_agent()
        self._passages = self._resolve_passages()

    def _create_agent(self) -> str:
        model = os.environ.get("LETTA_MODEL", "openai/gpt-4o-mini")
        blocks = [
            {
                "label": "human",
                "value": "Forgetting-Bench live adapter user.",
                "limit": 2000,
            },
            {
                "label": "persona",
                "value": "Archival-memory store for a forgetting benchmark.",
                "limit": 2000,
            },
        ]
        try:
            agent = self.client.agents.create(
                memory_blocks=blocks,
                model=model,
            )
        except TypeError:
            agent = self.client.agents.create(name="forgetting-bench")
        agent_id = getattr(agent, "id", None) or (agent.get("id") if isinstance(agent, dict) else None)
        if not agent_id:
            raise AdapterUnavailable("letta", "agents.create() returned no id")
        return str(agent_id)

    def _resolve_passages(self) -> Any:
        agents = getattr(self.client, "agents", None)
        if agents is not None and hasattr(agents, "passages"):
            return agents.passages
        if hasattr(self.client, "passages"):
            return self.client.passages
        raise AdapterUnavailable(
            "letta",
            "installed SDK has no agents.passages / passages surface",
        )

    def add(self, obs: Observation) -> None:
        bench_id = uuid.uuid4().hex[:12]
        triple = (obs.entity, obs.attribute, obs.value)
        self._truth[bench_id] = triple
        self._by_text[obs.text] = triple
        self._texts.append(obs.text)
        tags = [f"{_TAG_PREFIX}{bench_id}"]
        self._insert(obs.text, tags)

    def _insert(self, content: str, tags: list[str]) -> Any:
        attempts = (
            dict(agent_id=self.agent_id, content=content, tags=tags),
            dict(agent_id=self.agent_id, text=content, tags=tags),
            dict(content=content, tags=tags),
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return self._passages.insert(**kwargs)
            except (TypeError, AttributeError) as exc:
                last_error = exc
        if hasattr(self._passages, "create"):
            try:
                return self._passages.create(agent_id=self.agent_id, content=content, tags=tags)
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise AdapterUnavailable("letta", "passages.insert/create rejected every known signature")

    def retrieve(self, query: Query, k: int) -> list[RetrievedItem]:
        payload = self._search(query.text, k)
        items: list[RetrievedItem] = []
        for row in result_rows(payload)[:k]:
            entity, attribute, value = self._truth_for_row(row)
            items.append(RetrievedItem(row_text(row), entity, attribute, value))
        return items

    def _search(self, text: str, k: int) -> Any:
        attempts = (
            dict(agent_id=self.agent_id, query=text, limit=k),
            dict(agent_id=self.agent_id, query=text),
            dict(query=text, limit=k),
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return self._passages.search(**kwargs)
            except (TypeError, AttributeError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise AdapterUnavailable("letta", "passages.search rejected every known signature")

    def _truth_for_row(self, row: Any) -> tuple[str | None, str | None, str | None]:
        for tag in row_tags(row):
            if tag.startswith(_TAG_PREFIX):
                key = tag[len(_TAG_PREFIX) :]
                if key in self._truth:
                    return self._truth[key]
        mid = row_id(row)
        if mid and mid in self._truth:
            return self._truth[mid]
        return self._by_text.get(row_text(row), (None, None, None))

    def _list_rows(self) -> list[Any]:
        attempts = (
            dict(agent_id=self.agent_id),
            dict(agent_id=self.agent_id, limit=100_000),
            {},
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                payload = self._passages.list(**kwargs)
                return result_rows(payload) or list(payload or [])
            except (TypeError, AttributeError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return []

    def size(self) -> int:
        rows = self._list_rows()
        return len(rows) if rows else len(self._texts)

    def token_count(self) -> int:
        rows = self._list_rows()
        if rows:
            return sum(whitespace_tokens(row_text(row)) for row in rows)
        return sum(whitespace_tokens(t) for t in self._texts)
