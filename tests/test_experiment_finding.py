"""Regression guard on the headline finding itself.

If a change silently breaks the decay module, these assertions -- run on a small
but real workload -- should fail. They encode the claim the README makes.
"""

from forgetting_bench.adapters.reference import ReferenceAdapter
from forgetting_bench.bench.harness import run
from forgetting_bench.memory import EbbinghausDecay, NoDecay
from forgetting_bench.workload.synthetic import generate_workload


def _ab(seed: int, n_turns: int = 1500):
    wl = generate_workload(seed=seed, n_turns=n_turns)
    off = run(ReferenceAdapter(NoDecay()), wl, k=5)
    on = run(ReferenceAdapter(EbbinghausDecay(tau=150)), wl, k=5)
    return off, on


def test_decay_eliminates_contradictions_baseline_suffers():
    off, on = _ab(seed=0)
    assert off.contradiction_rate > 0.4      # baseline surfaces stale facts a lot
    assert on.contradiction_rate == 0.0      # decay suppresses every one


def test_decay_bounds_memory_growth():
    off, on = _ab(seed=0)
    assert on.final_size < off.final_size
    assert on.final_tokens < off.final_tokens


def test_decay_preserves_recall():
    off, on = _ab(seed=0)
    # Forgetting stale/noise should not cost us the current facts.
    assert on.recall_rate >= off.recall_rate - 0.05
    assert on.recall_rate > 0.9


def test_finding_holds_across_seeds():
    for seed in (1, 2, 3):
        off, on = _ab(seed=seed)
        assert on.contradiction_rate < off.contradiction_rate
        assert on.final_size < off.final_size
