"""Regression guard on the headline finding itself.

These run the three arms on a real (small) workload and assert the *shape* of the
result the README claims -- magnitudes and orderings, not just ``a < b`` where
one side is trivially zero. If a change breaks the decay mechanism or the
realistic-contradiction setup, these should fail.
"""

from forgetting_bench.adapters.reference import ReferenceAdapter
from forgetting_bench.bench.harness import run
from forgetting_bench.memory import EbbinghausDecay, LastWriteWins, LearnedForget, NoDecay
from forgetting_bench.workload.synthetic import generate_workload


def _arms(seed: int, n_turns: int = 2000, tau: float = 150.0):
    wl = generate_workload(seed=seed, n_turns=n_turns)
    keep = run(ReferenceAdapter(NoDecay()), wl, k=5)
    lww = run(ReferenceAdapter(LastWriteWins()), wl, k=5)
    decay = run(ReferenceAdapter(EbbinghausDecay(tau=tau)), wl, k=5)
    return keep, lww, decay


def test_contradiction_is_nonzero_everywhere_not_a_tautology():
    keep, lww, decay = _arms(seed=0)
    # Extraction misses leave a real contradiction floor -- no arm reaches zero.
    assert keep.contradiction_rate > 0.6
    assert lww.contradiction_rate > 0.15
    assert decay.contradiction_rate > 0.15
    # ...but dedup / decay roughly halve keep-everything's contradictions.
    assert lww.contradiction_rate < keep.contradiction_rate - 0.3
    assert decay.contradiction_rate < keep.contradiction_rate - 0.3


def test_decay_beats_last_write_wins_on_memory_at_matched_quality():
    # Run at the documented horizon: decay's win is that it stays *bounded* while
    # LWW hoards noise, so the gap widens with turns.
    keep, lww, decay = _arms(seed=0, n_turns=3000)
    assert decay.final_size < lww.final_size * 0.75      # ~40% smaller here
    assert decay.final_tokens < lww.final_tokens * 0.75
    # ...at matched contradiction handling and recall.
    assert abs(decay.contradiction_rate - lww.contradiction_rate) < 0.05
    assert abs(decay.recall_rate - lww.recall_rate) < 0.05


def test_decay_memory_is_bounded_while_last_write_wins_grows():
    wl = generate_workload(seed=0, n_turns=3000)
    lww = run(ReferenceAdapter(LastWriteWins()), wl, k=5)
    decay = run(ReferenceAdapter(EbbinghausDecay(tau=150)), wl, k=5)

    def half_and_end(res):
        trace = res.size_trace
        mid = trace[len(trace) // 2][1]
        return mid, res.final_size

    lww_mid, lww_end = half_and_end(lww)
    decay_mid, decay_end = half_and_end(decay)
    lww_growth = lww_end - lww_mid
    decay_growth = decay_end - decay_mid
    # LWW grows ~linearly with noise; decay's growth is bounded (much slower).
    assert lww_growth > 3 * decay_growth
    assert decay_growth >= 0


def test_contradiction_is_tau_sensitive_a_real_frontier():
    wl = generate_workload(seed=0, n_turns=2000)
    aggressive = run(ReferenceAdapter(EbbinghausDecay(tau=30)), wl, k=5)
    lenient = run(ReferenceAdapter(EbbinghausDecay(tau=1000)), wl, k=5)
    # Forgetting harder cuts contradictions further -- but costs recall.
    assert aggressive.contradiction_rate < lenient.contradiction_rate - 0.1
    assert aggressive.recall_rate < lenient.recall_rate - 0.05
    assert aggressive.final_size < lenient.final_size


def test_learned_forget_matches_floor_and_bounds_harder_than_decay():
    """Harness-shaped guard: learned-forget sits on the same contradiction
    floor as Ebbinghaus, with a tighter bound on unslotted noise."""
    wl = generate_workload(seed=0, n_turns=3000)
    decay = run(ReferenceAdapter(EbbinghausDecay(tau=150)), wl, k=5)
    learned = run(ReferenceAdapter(LearnedForget.trained(seed=0)), wl, k=5)
    assert abs(learned.contradiction_rate - decay.contradiction_rate) < 0.05
    assert abs(learned.recall_rate - decay.recall_rate) < 0.05
    assert learned.final_size < decay.final_size
    assert learned.final_tokens < decay.final_tokens


def test_finding_holds_across_seeds():
    for seed in (1, 2, 3):
        keep, lww, decay = _arms(seed=seed)
        assert keep.contradiction_rate > 0.6            # baseline genuinely bad
        assert decay.contradiction_rate < keep.contradiction_rate - 0.3
        assert decay.final_size < lww.final_size        # memory win vs dedup
        assert decay.recall_rate > 0.55                 # recall not destroyed
