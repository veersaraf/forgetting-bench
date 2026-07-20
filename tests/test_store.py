from forgetting_bench.memory import (
    EbbinghausDecay,
    LastWriteWins,
    MemoryStore,
    NoDecay,
)
from forgetting_bench.workload.synthetic import default_extractor


def test_retrieve_ranks_correct_slot_above_other_slots():
    store = MemoryStore(decay=NoDecay())
    store.add("Alice's home city is boston.", turn=0, entity="alice", attribute="home_city")
    store.add("Bob's home city is denver.", turn=1, entity="bob", attribute="home_city")
    store.add("Alice's job title is nurse.", turn=2, entity="alice", attribute="job_title")

    top = store.retrieve("What is Alice's home city?", now=3, k=1)
    assert top[0].entry.entity == "alice"
    assert top[0].entry.attribute == "home_city"


def test_within_slot_newer_fact_outranks_older():
    store = MemoryStore(decay=NoDecay())
    store.add("Alice's home city is boston.", turn=0, entity="alice", attribute="home_city")
    store.add("Alice's home city is now denver.", turn=100, entity="alice", attribute="home_city")

    top = store.retrieve("What is Alice's home city?", now=101, k=2)
    assert "denver" in top[0].entry.text  # the current value ranks first


def test_scores_are_bounded_unit_interval():
    store = MemoryStore(decay=NoDecay())
    store.add("Alice's home city is boston.", turn=0, entity="alice", attribute="home_city")
    for s in store.score_all("What is Alice's home city?", now=5):
        assert 0.0 <= s.score <= 1.0


def test_supersession_removes_stale_fact_from_top_results():
    """The core differentiator: a contradicted fact should drop out of retrieval."""
    on = MemoryStore(decay=EbbinghausDecay(tau=200))
    off = MemoryStore(decay=NoDecay())
    for store in (on, off):
        store.add("Alice's home city is boston.", turn=0, entity="alice", attribute="home_city")
        store.add("Alice's home city is now denver.", turn=50, entity="alice", attribute="home_city")

    q, now = "What is Alice's home city?", 60
    off_texts = [s.entry.text for s in off.retrieve(q, now, k=5)]
    on_texts = [s.entry.text for s in on.retrieve(q, now, k=5)]

    # Baseline keeps the stale fact retrievable; decay suppresses it.
    assert any("boston" in t for t in off_texts)
    on_top2 = on.retrieve(q, now, k=2)
    assert "denver" in on_top2[0].entry.text
    assert on_top2[0].strength == 1.0
    stale = next(s for s in on.score_all(q, now) if "boston" in s.entry.text)
    assert stale.strength < 0.1  # ranking strength crushed by supersession


def test_pruning_bounds_memory_growth():
    on = MemoryStore(decay=EbbinghausDecay(tau=20, prune_threshold=0.05))
    off = MemoryStore(decay=NoDecay())
    for turn in range(500):
        text = f"Noise observation number {turn} about nothing."
        on.add(text, turn=turn)
        off.add(text, turn=turn)
    assert off.size() == 500
    assert on.size() < off.size()


def test_last_write_wins_dedups_slots_but_hoards_noise():
    store = MemoryStore(decay=LastWriteWins())
    for turn, city in enumerate(["boston", "denver", "austin"]):
        store.add(f"Alice's home city is {city}.", turn=turn,
                  entity="alice", attribute="home_city")
    for turn in range(3, 20):
        store.add(f"A stranger said thing number {turn}.", turn=turn)  # no slot
    slot_entries = [e for e in store.entries if e.slot == ("alice", "home_city")]
    assert len(slot_entries) == 1                       # only the latest survives
    assert "austin" in slot_entries[0].text
    assert store.size() == 1 + 17                       # 1 slot + all 17 noise kept


def test_store_extracts_keys_and_misses_paraphrases():
    """Integration: with the real extractor, a canonical update supersedes but a
    paraphrased one does not, so the stale fact survives as a contradiction."""
    store = MemoryStore(decay=EbbinghausDecay(tau=500), slot_extractor=default_extractor())
    store.add("Alice's home city is boston.", turn=0)         # extracted -> slot
    store.add("Alice relocated to denver.", turn=10)          # paraphrase -> no slot

    entries = {e.text: e for e in store.entries}
    boston = entries["Alice's home city is boston."]
    denver = entries["Alice relocated to denver."]
    assert boston.slot == ("alice", "home_city")
    assert denver.slot is None                    # extraction missed it
    assert boston.superseded_by is None           # so the stale fact is NOT superseded

    # The stale canonical fact still ranks top for the query -> a live contradiction.
    top = store.retrieve("What is Alice's home city?", now=11, k=1)
    assert "boston" in top[0].entry.text


def test_recency_decay_config_validated():
    import pytest

    with pytest.raises(ValueError):
        MemoryStore(recency_decay=1.5)
