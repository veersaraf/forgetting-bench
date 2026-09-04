from forgetting_bench.workload.synthetic import default_extractor, keyword_extractor


def test_extracts_canonical_statement():
    ex = default_extractor()
    assert ex.extract("Alice's home city is boston.") == ("alice", "home_city")
    assert ex.extract("Alice's home city is now denver.") == ("alice", "home_city")


def test_keyword_teacher_still_available():
    ex = keyword_extractor()
    assert ex.extract("Alice's home city is boston.") == ("alice", "home_city")
    assert ex.extract("Alice relocated to denver.") is None


def test_extracts_canonical_salary():
    ex = default_extractor()
    assert ex.extract("Bob's salary is 100k.") == ("bob", "salary")


def test_misses_paraphrased_update():
    """The honest failure mode: a paraphrase that omits the attribute phrase is
    not linked to its slot, so supersession can't fire on it."""
    ex = default_extractor()
    assert ex.extract("Alice relocated to denver.") is None
    assert ex.extract("Bob got a raise to 110k.") is None
    assert ex.extract("Carol was promoted to pilot.") is None


def test_leaves_noise_unslotted():
    ex = default_extractor()
    assert ex.extract("A coworker complained about the printer again.") is None


def test_requires_both_entity_and_attribute():
    ex = default_extractor()
    # attribute phrase but no known entity
    assert ex.extract("the home city is boston") is None
