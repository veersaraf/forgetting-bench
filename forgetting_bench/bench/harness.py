"""Drives a workload through an adapter and collects the metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.base import MemoryAdapter
from ..workload.synthetic import Observation, Query, Workload
from .metrics import (
    QueryEval,
    answer_accuracy,
    contradiction_rate,
    evaluate_query,
    mean_precision,
    recall_rate,
    stale_context_rate,
)


@dataclass
class BenchResult:
    name: str
    evals: list[QueryEval]
    size_trace: list[tuple[int, int]] = field(default_factory=list)
    token_trace: list[tuple[int, int]] = field(default_factory=list)

    @property
    def contradiction_rate(self) -> float:
        return contradiction_rate(self.evals)

    @property
    def recall_rate(self) -> float:
        return recall_rate(self.evals)

    @property
    def answer_accuracy(self) -> float:
        return answer_accuracy(self.evals)

    @property
    def mean_precision(self) -> float:
        return mean_precision(self.evals)

    @property
    def stale_context_rate(self) -> float:
        return stale_context_rate(self.evals)

    @property
    def final_size(self) -> int:
        return self.size_trace[-1][1] if self.size_trace else 0

    @property
    def peak_size(self) -> int:
        return max((s for _, s in self.size_trace), default=0)

    @property
    def final_tokens(self) -> int:
        return self.token_trace[-1][1] if self.token_trace else 0

    def summary(self) -> dict[str, float]:
        return {
            "contradiction_rate": self.contradiction_rate,
            "recall_rate": self.recall_rate,
            "answer_accuracy": self.answer_accuracy,
            "mean_precision": self.mean_precision,
            "stale_context_rate": self.stale_context_rate,
            "final_size": float(self.final_size),
            "peak_size": float(self.peak_size),
            "final_tokens": float(self.final_tokens),
            "n_queries": float(len(self.evals)),
        }


def run(
    adapter: MemoryAdapter,
    workload: Workload,
    k: int = 5,
    trace_every: int = 25,
) -> BenchResult:
    """Replay ``workload`` through ``adapter``, scoring every query in order.

    ``trace_every`` controls how often the live memory size/token count is
    sampled for the bloat curves.
    """
    evals: list[QueryEval] = []
    size_trace: list[tuple[int, int]] = []
    token_trace: list[tuple[int, int]] = []

    for event in workload.events:
        if isinstance(event, Observation):
            adapter.add(event)
        elif isinstance(event, Query):
            retrieved = adapter.retrieve(event, k)
            evals.append(evaluate_query(event, retrieved))
        else:  # pragma: no cover - defensive
            raise TypeError(f"unknown event type: {type(event)!r}")

        if event.turn % trace_every == 0:
            size_trace.append((event.turn, adapter.size()))
            token_trace.append((event.turn, adapter.token_count()))

    # Always capture the final state.
    last_turn = workload.events[-1].turn if workload.events else 0
    size_trace.append((last_turn, adapter.size()))
    token_trace.append((last_turn, adapter.token_count()))

    return BenchResult(
        name=adapter.name,
        evals=evals,
        size_trace=size_trace,
        token_trace=token_trace,
    )
