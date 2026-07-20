"""Metrics the recall-obsessed incumbents don't report.

Each aggregate is a pure function of per-query results or a size trace, so it can
be unit-tested in isolation. The headline metric is the **stale-fact
contradiction rate**: how often the memory hands the agent an outdated,
contradicted fact for a slot it was asked about.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters.base import RetrievedItem
from ..workload.synthetic import Query


@dataclass
class QueryEval:
    """Scored outcome of one query against ground truth."""

    turn: int
    entity: str
    attribute: str
    correct_value: str
    recall_hit: bool          # the current correct fact is somewhere in top-k
    contradiction_hit: bool   # a stale (outdated) fact for the slot is in top-k
    top1_correct: bool        # the single top-ranked item is the correct fact
    precision: float          # fraction of top-k that IS the current correct fact
    stale_fraction: float     # fraction of top-k that are stale, contradicted facts


def evaluate_query(query: Query, retrieved: list[RetrievedItem]) -> QueryEval:
    """Score one query's retrieved items against the known current value.

    * on-topic  -> item is about the queried (entity, attribute) slot
    * correct   -> on-topic and its value equals the current ground truth
    * stale     -> on-topic but its value is an older, contradicted one
    """
    n_correct = 0
    n_stale = 0
    for item in retrieved:
        if item.entity == query.entity and item.attribute == query.attribute:
            if item.value == query.correct_value:
                n_correct += 1
            elif item.value is not None:
                n_stale += 1

    top1_correct = bool(
        retrieved
        and retrieved[0].entity == query.entity
        and retrieved[0].attribute == query.attribute
        and retrieved[0].value == query.correct_value
    )
    k = len(retrieved)
    return QueryEval(
        turn=query.turn,
        entity=query.entity,
        attribute=query.attribute,
        correct_value=query.correct_value,
        recall_hit=n_correct > 0,
        contradiction_hit=n_stale > 0,
        top1_correct=top1_correct,
        precision=(n_correct / k) if k else 0.0,
        stale_fraction=(n_stale / k) if k else 0.0,
    )


def _rate(evals: list[QueryEval], attr: str) -> float:
    if not evals:
        return 0.0
    return sum(getattr(e, attr) for e in evals) / len(evals)


def contradiction_rate(evals: list[QueryEval]) -> float:
    """Fraction of queries whose top-k contained a stale, contradicted fact."""
    return _rate(evals, "contradiction_hit")


def recall_rate(evals: list[QueryEval]) -> float:
    """Fraction of queries whose top-k contained the current correct fact."""
    return _rate(evals, "recall_hit")


def answer_accuracy(evals: list[QueryEval]) -> float:
    """Fraction of queries whose single top-ranked item was the correct fact."""
    return _rate(evals, "top1_correct")


def mean_precision(evals: list[QueryEval]) -> float:
    """Mean fraction of retrieved items that were the current correct fact."""
    return _rate(evals, "precision")


def stale_context_rate(evals: list[QueryEval]) -> float:
    """Mean fraction of the retrieved context that was stale, contradicted facts.

    This is the *severity* companion to :func:`contradiction_rate`: not just
    whether a stale fact showed up, but how much of the context it polluted.
    """
    return _rate(evals, "stale_fraction")


def memory_growth(size_trace: list[tuple[int, int]]) -> float:
    """Final live-memory size -- the endpoint of the bloat curve."""
    return float(size_trace[-1][1]) if size_trace else 0.0
