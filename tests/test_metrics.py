from forgetting_bench.adapters.base import RetrievedItem
from forgetting_bench.bench.metrics import (
    QueryEval,
    answer_accuracy,
    contradiction_rate,
    evaluate_query,
    mean_precision,
    memory_growth,
    recall_rate,
    stale_context_rate,
)
from forgetting_bench.workload.synthetic import Query


def _query():
    return Query(turn=10, text="What is Alice's home city?",
                 entity="alice", attribute="home_city", correct_value="denver")


def _item(entity, attribute, value):
    return RetrievedItem(text="", entity=entity, attribute=attribute, value=value)


def test_evaluate_detects_correct_current_fact():
    retrieved = [
        _item("alice", "home_city", "denver"),   # correct current
        _item("bob", "home_city", "austin"),      # off-slot
    ]
    ev = evaluate_query(_query(), retrieved)
    assert ev.recall_hit is True
    assert ev.contradiction_hit is False
    assert ev.top1_correct is True
    assert ev.precision == 0.5   # 1 correct of 2 retrieved
    assert ev.stale_fraction == 0.0


def test_evaluate_detects_stale_contradiction():
    retrieved = [
        _item("alice", "home_city", "boston"),    # stale (was denver)
        _item("alice", "home_city", "denver"),    # correct current
    ]
    ev = evaluate_query(_query(), retrieved)
    assert ev.recall_hit is True
    assert ev.contradiction_hit is True
    assert ev.top1_correct is False               # top-1 is the stale one
    assert ev.stale_fraction == 0.5


def test_noise_only_retrieval_scores_zero():
    retrieved = [_item(None, None, None), _item("carol", "job_title", "chef")]
    ev = evaluate_query(_query(), retrieved)
    assert ev.recall_hit is False
    assert ev.contradiction_hit is False
    assert ev.precision == 0.0


def test_empty_retrieval_is_safe():
    ev = evaluate_query(_query(), [])
    assert ev.precision == 0.0 and ev.stale_fraction == 0.0
    assert ev.recall_hit is False


def _ev(recall, contra, top1, prec, stale):
    return QueryEval(0, "a", "b", "v", recall, contra, top1, prec, stale)


def test_aggregate_rates():
    evals = [
        _ev(True, True, False, 0.2, 0.4),
        _ev(True, False, True, 0.2, 0.0),
    ]
    assert recall_rate(evals) == 1.0
    assert contradiction_rate(evals) == 0.5
    assert answer_accuracy(evals) == 0.5
    assert mean_precision(evals) == 0.2
    assert stale_context_rate(evals) == 0.2


def test_aggregates_on_empty_are_zero():
    assert contradiction_rate([]) == 0.0
    assert recall_rate([]) == 0.0


def test_memory_growth_returns_final_size():
    assert memory_growth([(0, 10), (50, 30), (100, 42)]) == 42.0
    assert memory_growth([]) == 0.0
