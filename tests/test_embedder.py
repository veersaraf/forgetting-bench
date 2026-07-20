from forgetting_bench.memory.embedder import (
    HashingEmbedder,
    _stable_hash,
    cosine,
    tokenize,
)


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert tokenize("Alice's home-city is Boston!") == [
        "alice", "s", "home", "city", "is", "boston",
    ]


def test_stable_hash_is_deterministic_across_calls():
    assert _stable_hash("boston") == _stable_hash("boston")
    assert _stable_hash("boston") != _stable_hash("denver")


def test_identical_text_has_cosine_one():
    emb = HashingEmbedder()
    v = emb.embed("alice home city boston")
    assert cosine(v, v) == 1.0


def test_disjoint_text_has_cosine_zero():
    emb = HashingEmbedder()
    a = emb.embed("alice home city boston")
    b = emb.embed("quarterly revenue projections spreadsheet")
    assert cosine(a, b) == 0.0


def test_relevant_text_scores_higher_than_partial_overlap():
    emb = HashingEmbedder()
    query = emb.embed("what is alice home city")
    on_slot = emb.embed("alice home city is boston")
    other_entity = emb.embed("bob home city is boston")
    assert cosine(query, on_slot) > cosine(query, other_entity)


def test_empty_text_embeds_to_zero_vector():
    emb = HashingEmbedder()
    v = emb.embed("")
    assert v.norm == 0.0
    assert cosine(v, emb.embed("anything")) == 0.0
