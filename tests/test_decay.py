import math

import pytest

from forgetting_bench.memory.decay import EbbinghausDecay, NoDecay
from forgetting_bench.memory.entry import MemoryEntry


def _entry(turn=0, importance=0.5, superseded=None):
    e = MemoryEntry(id=1, turn=turn, text="x", importance=importance)
    e.superseded_by = superseded
    return e


def test_nodecay_never_forgets():
    d = NoDecay()
    e = _entry(turn=0)
    assert d.strength(e, 10_000) == 1.0
    assert d.retention(e, 10_000) == 1.0
    assert d.should_prune(e, 10_000) is False


def test_retention_is_one_at_creation_and_decreases_with_time():
    d = EbbinghausDecay(tau=100)
    e = _entry(turn=0, importance=0.0)
    assert d.retention(e, 0) == pytest.approx(1.0)
    assert d.retention(e, 100) < d.retention(e, 50) < 1.0


def test_higher_importance_slows_forgetting():
    d = EbbinghausDecay(tau=100, importance_gain=4.0)
    low = _entry(turn=0, importance=0.1)
    high = _entry(turn=0, importance=0.9)
    # Same elapsed time -> the important memory retains more.
    assert d.retention(high, 200) > d.retention(low, 200)


def test_retention_matches_ebbinghaus_formula():
    d = EbbinghausDecay(tau=100, importance_gain=4.0)
    e = _entry(turn=0, importance=0.5)
    stability = 100 * (1 + 4.0 * 0.5)
    assert d.retention(e, 150) == pytest.approx(math.exp(-150 / stability))


def test_supersession_lowers_ranking_strength_not_time():
    d = EbbinghausDecay(supersession_factor=0.05)
    live = _entry(turn=0, superseded=None)
    stale = _entry(turn=0, superseded=42)
    assert d.strength(live, 0) == 1.0
    assert d.strength(stale, 0) == 0.05


def test_should_prune_fires_below_threshold():
    d = EbbinghausDecay(tau=50, importance_gain=0.0, prune_threshold=0.02)
    e = _entry(turn=0, importance=0.0)
    # exp(-t/50) < 0.02  ->  t > 50 * ln(50) ~= 195
    assert d.should_prune(e, 100) is False
    assert d.should_prune(e, 300) is True


def test_superseded_facts_are_pruned_far_sooner():
    d = EbbinghausDecay(tau=100, importance_gain=0.0, prune_threshold=0.02)
    live = _entry(turn=0, importance=0.0, superseded=None)
    stale = _entry(turn=0, importance=0.0, superseded=7)
    now = 120
    assert d.should_prune(live, now) is False
    assert d.should_prune(stale, now) is True


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        EbbinghausDecay(tau=0)
    with pytest.raises(ValueError):
        EbbinghausDecay(supersession_factor=2.0)
