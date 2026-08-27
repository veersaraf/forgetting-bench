from forgetting_bench.memory.decay import LastWriteWins
from forgetting_bench.memory.entry import MemoryEntry
from forgetting_bench.memory.learned_decay import ForgetNet, LearnedForget, train_forget_net
from forgetting_bench.memory.store import MemoryStore


def _entry(turn=0, importance=0.5, superseded=None, entity=None, attribute=None):
    e = MemoryEntry(
        id=1,
        turn=turn,
        text="x",
        importance=importance,
        entity=entity,
        attribute=attribute,
    )
    e.superseded_by = superseded
    return e


def test_forget_net_is_a_torch_module():
    import torch
    import torch.nn as nn

    net = ForgetNet()
    assert isinstance(net, nn.Module)
    features = torch.zeros(3, 4)
    strength, retain_logit = net(features)
    assert strength.shape == (3,)
    assert retain_logit.shape == (3,)


def test_training_produces_finite_loss():
    _, loss = train_forget_net(seed=0, steps=20)
    assert loss < 2.0


def test_learned_forget_crushes_superseded_ranking():
    policy = LearnedForget.trained(seed=0)
    live = _entry(turn=0, entity="alice", attribute="home_city")
    stale = _entry(turn=0, entity="alice", attribute="home_city", superseded=9)
    assert policy.strength(live, 10) > 0.8
    assert policy.strength(stale, 10) < 0.2


def test_learned_forget_prunes_old_noise_and_keeps_fresh_facts():
    policy = LearnedForget.trained(seed=0)
    noise = _entry(turn=0, importance=0.2)  # unslotted
    fact = _entry(turn=0, importance=0.8, entity="alice", attribute="home_city")
    assert policy.should_prune(noise, 400) is True
    assert policy.should_prune(fact, 20) is False


def test_learned_forget_bounds_noise_unlike_last_write_wins():
    learned = MemoryStore(decay=LearnedForget.trained(seed=0))
    lww = MemoryStore(decay=LastWriteWins())
    for turn in range(400):
        text = f"A stranger said thing number {turn}."
        learned.add(text, turn=turn)
        lww.add(text, turn=turn)
    assert lww.size() == 400
    assert learned.size() < 200


def test_learned_forget_supersedes_on_extracted_slots():
    store = MemoryStore(decay=LearnedForget.trained(seed=0))
    store.add("Alice's home city is boston.", turn=0, entity="alice", attribute="home_city")
    store.add("Alice's home city is now denver.", turn=10, entity="alice", attribute="home_city")
    slotted = [e for e in store.entries if e.slot == ("alice", "home_city")]
    # The stale one is either crushed or already pruned.
    live = [e for e in slotted if e.superseded_by is None]
    assert len(live) == 1
    assert "denver" in live[0].text
