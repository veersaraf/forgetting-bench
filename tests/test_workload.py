from forgetting_bench.workload.synthetic import (
    Observation,
    Query,
    generate_workload,
)


def test_workload_is_deterministic_for_a_seed():
    a = generate_workload(seed=3, n_turns=500)
    b = generate_workload(seed=3, n_turns=500)
    assert [e.text for e in a.events] == [e.text for e in b.events]


def test_different_seeds_differ():
    a = generate_workload(seed=1, n_turns=500)
    b = generate_workload(seed=2, n_turns=500)
    assert [e.text for e in a.events] != [e.text for e in b.events]


def test_query_ground_truth_matches_latest_assertion():
    """The correct_value on every query is the most recent value asserted for its
    slot -- this is the ground truth the harness scores against."""
    wl = generate_workload(seed=0, n_turns=2000)
    current: dict[tuple[str, str], str] = {}
    for e in wl.events:
        if isinstance(e, Observation):
            if not e.is_noise:
                current[(e.entity, e.attribute)] = e.value
        elif isinstance(e, Query):
            assert current[(e.entity, e.attribute)] == e.correct_value


def test_workload_contains_updates_and_noise():
    wl = generate_workload(seed=0, n_turns=2000)
    obs = wl.observations
    assert any(o.is_noise for o in obs)            # distractors present
    assert any(not o.is_noise for o in obs)        # tracked facts present
    # At least one slot is updated more than once (a contradiction is created).
    assert any(len(v) > 1 for v in wl.history.values())


def test_updates_actually_change_the_value():
    wl = generate_workload(seed=0, n_turns=3000)
    for assertions in wl.history.values():
        values = [v for _, v in assertions]
        for a, b in zip(values, values[1:]):
            assert a != b   # consecutive assertions differ -> real contradiction
