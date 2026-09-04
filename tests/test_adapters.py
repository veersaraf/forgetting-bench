"""Live adapters talk to a real SDK when configured; otherwise they refuse."""

import pytest

from forgetting_bench.adapters import (
    AdapterUnavailable,
    LettaAdapter,
    Mem0Adapter,
    incumbent_probes,
    letta_probe,
    mem0_probe,
)
from forgetting_bench.adapters._compat import result_rows, row_text
from forgetting_bench.workload.synthetic import Observation, Query


class _FakeMem0:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, messages, **kwargs):
        text = messages if isinstance(messages, str) else messages[0]["content"]
        mid = str(len(self.items))
        rec = {
            "id": mid,
            "memory": text,
            "metadata": kwargs.get("metadata") or {},
        }
        self.items.append(rec)
        return {"results": [{"id": mid, "memory": text, "event": "ADD"}]}

    def search(self, query, **kwargs):
        k = kwargs.get("top_k") or kwargs.get("limit") or 5
        return {"results": list(reversed(self.items))[:k]}

    def get_all(self, **kwargs):
        return {"results": self.items}


class _FakePassages:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def insert(self, **kwargs):
        content = kwargs.get("content") or kwargs.get("text") or ""
        rec = {
            "id": str(len(self.rows)),
            "content": content,
            "tags": kwargs.get("tags") or [],
        }
        self.rows.append(rec)
        return rec

    def search(self, **kwargs):
        k = kwargs.get("limit") or 5
        return {"results": list(reversed(self.rows))[:k]}

    def list(self, **kwargs):
        return {"results": self.rows}


def _obs(text="Alice's home city is boston.", turn=0, entity="alice",
         attribute="home_city", value="boston"):
    return Observation(turn, text, entity, attribute, value, is_noise=False)


def _query():
    return Query(1, "What is Alice's home city?", "alice", "home_city", "boston")


def test_probes_are_honest_when_unconfigured():
    mem = mem0_probe()
    let = letta_probe()
    if not mem.runnable:
        assert "not installed" in mem.reason or "API_KEY" in mem.reason
    if not let.runnable:
        assert "not installed" in let.reason or "LETTA" in let.reason
    assert len(incumbent_probes()) == 2


def test_mem0_refuses_to_construct_when_not_runnable():
    if mem0_probe().runnable:
        pytest.skip("a live mem0 backend is configured in this environment")
    with pytest.raises(AdapterUnavailable) as exc:
        Mem0Adapter()
    assert "mem0" in str(exc.value)


def test_letta_refuses_to_construct_when_not_runnable():
    if letta_probe().runnable:
        pytest.skip("a live Letta backend is configured in this environment")
    with pytest.raises(AdapterUnavailable) as exc:
        LettaAdapter()
    assert "letta" in str(exc.value)


def test_mem0_adapter_round_trips_metadata_through_injected_client():
    adapter = Mem0Adapter(client=_FakeMem0(), infer=False)
    adapter.add(_obs())
    adapter.add(_obs("Alice relocated to denver.", turn=1, value="denver", entity="alice",
                     attribute="home_city"))
    hits = adapter.retrieve(_query(), k=2)
    assert len(hits) == 2
    assert hits[0].value == "denver"
    assert hits[0].entity == "alice"
    assert adapter.size() == 2
    assert adapter.token_count() > 0


def test_letta_adapter_round_trips_tags_through_injected_client():
    passages = _FakePassages()
    adapter = LettaAdapter(client=object(), agent_id="a1", passages=passages)
    adapter.add(_obs())
    hits = adapter.retrieve(_query(), k=1)
    assert hits[0].entity == "alice"
    assert hits[0].attribute == "home_city"
    assert hits[0].value == "boston"
    assert adapter.size() == 1


def test_compat_unwraps_mem0_envelope():
    rows = result_rows({"results": [{"memory": "hello", "id": "1"}]})
    assert row_text(rows[0]) == "hello"
