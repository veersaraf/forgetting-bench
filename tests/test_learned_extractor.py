"""The learned extractor is a real torch module, and it keeps the honest miss."""

import torch
import torch.nn as nn

from forgetting_bench.memory.extractor import KeywordSlotExtractor
from forgetting_bench.memory.learned_extractor import (
    LearnedSlotExtractor,
    SlotTagger,
    default_learned_extractor,
    train_slot_tagger,
)
from forgetting_bench.workload.synthetic import (
    ATTRIBUTES,
    ENTITIES,
    default_extractor,
    generate_workload,
    keyword_extractor,
)


def test_slot_tagger_is_a_torch_module():
    model = SlotTagger(vocab_size=32, n_entities=3, n_attributes=4)
    assert isinstance(model, nn.Module)
    ids = torch.randint(0, 32, (2, 8))
    ent, attr = model(ids)
    assert ent.shape == (2, 4)
    assert attr.shape == (2, 5)


def test_training_drives_loss_down():
    torch.manual_seed(0)
    _, _, _, _, first = train_slot_tagger(seed=0, epochs=1)
    _, _, _, _, later = train_slot_tagger(seed=0, epochs=12)
    assert later.final_loss < first.final_loss
    assert later.steps > first.steps


def test_learned_hits_canonical_and_misses_hard_updates():
    ex = default_learned_extractor()
    assert isinstance(ex, LearnedSlotExtractor)
    assert ex.extract("Alice's home city is boston.") == ("alice", "home_city")
    assert ex.extract("Alice's home city is now denver.") == ("alice", "home_city")
    assert ex.extract("Bob's salary is 100k.") == ("bob", "salary")
    assert ex.extract("Alice relocated to denver.") is None
    assert ex.extract("Bob got a raise to 110k.") is None
    assert ex.extract("Carol was promoted to pilot.") is None
    assert ex.extract("A coworker complained about the printer again.") is None


def test_learned_does_not_leak_value_to_attribute():
    """'denver' alone must not become home_city -- that would collapse the gap."""
    ex = default_learned_extractor()
    assert ex.extract("Someone mentioned denver.") is None
    assert ex.extract("A podcast discussed 110k.") is None


def test_learned_matches_keyword_teacher_on_workload_observations():
    teacher = keyword_extractor()
    student = default_extractor()
    assert isinstance(teacher, KeywordSlotExtractor)
    wl = generate_workload(seed=0, n_turns=800)
    mismatches = 0
    for obs in wl.observations:
        if teacher.extract(obs.text) != student.extract(obs.text):
            mismatches += 1
    assert mismatches == 0


def test_all_hard_templates_still_unslotted():
    ex = default_learned_extractor()
    for entity in ENTITIES:
        for spec in ATTRIBUTES.values():
            for value in spec.values:
                for tmpl in spec.hard_templates:
                    text = tmpl.format(E=entity.title(), v=value)
                    assert ex.extract(text) is None, text
