"""A reproducible, seeded long-horizon workload.

The stream mixes three things that stress *forgetting* rather than recall:

* **facts** -- the first assertion about an (entity, attribute) slot,
* **updates** -- a later assertion that changes a slot's value, making every
  earlier value for that slot *stale* (a contradiction if still retrieved),
* **noise** -- distractor observations about nothing the queries ever ask for.

Interleaved **queries** ask for the current value of a slot. Because we build the
stream ourselves we know the ground-truth current value at every turn, so the
harness can score contradictions, recall and precision exactly. Everything is
driven by a single seeded RNG, so a given seed always yields the same stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Tracked vocabulary. Attribute phrases are multi-word so queries and facts share
# lexical signal for the relevance term.
ENTITIES = [
    "alice", "bob", "carol", "dan", "erin",
    "frank", "grace", "heidi", "ivan", "judy",
]

ATTRIBUTES = {
    "home_city": ("home city", ["boston", "denver", "austin", "seattle", "miami", "portland"]),
    "job_title": ("job title", ["engineer", "teacher", "nurse", "pilot", "chef", "lawyer"]),
    "favorite_food": ("favorite food", ["ramen", "tacos", "sushi", "pizza", "curry", "falafel"]),
    "phone_model": ("phone model", ["pixel", "iphone", "galaxy", "nokia", "oneplus", "moto"]),
    "current_project": ("current project", ["atlas", "orbit", "delta", "nimbus", "vertex", "cobalt"]),
}

# Distractor material -- deliberately disjoint from tracked vocabulary.
_NOISE_SUBJECTS = ["the weather", "a coworker", "the team", "a stranger", "the news", "a podcast"]
_NOISE_PREDICATES = [
    "mentioned a delayed train this morning",
    "was surprisingly cheerful today",
    "talked about a movie over lunch",
    "spotted a hawk near the parking lot",
    "recommended a new coffee shop downtown",
    "complained about the printer again",
]


@dataclass
class Observation:
    turn: int
    text: str
    entity: str | None
    attribute: str | None
    value: str | None
    is_noise: bool


@dataclass
class Query:
    turn: int
    text: str
    entity: str
    attribute: str
    correct_value: str


@dataclass
class Workload:
    events: list[Observation | Query]
    seed: int
    n_turns: int
    # slot -> ordered list of (turn, value) assertions, for test/introspection.
    history: dict[tuple[str, str], list[tuple[int, str]]] = field(default_factory=dict)

    @property
    def observations(self) -> list[Observation]:
        return [e for e in self.events if isinstance(e, Observation)]

    @property
    def queries(self) -> list[Query]:
        return [e for e in self.events if isinstance(e, Query)]


def _attr_phrase(attribute: str) -> str:
    return ATTRIBUTES[attribute][0]


def generate_workload(
    seed: int = 0,
    n_turns: int = 3000,
    p_query: float = 0.12,
    p_update: float = 0.18,
) -> Workload:
    """Build a deterministic stream of ``n_turns`` events.

    ``p_query`` / ``p_update`` are the per-turn probabilities of a query and a
    slot update respectively; the remainder are noise. Every slot is seeded once
    before random updates begin, so queries always have a ground-truth answer.
    """
    rng = random.Random(seed)
    slots = [(e, a) for e in ENTITIES for a in ATTRIBUTES]

    current: dict[tuple[str, str], str] = {}
    history: dict[tuple[str, str], list[tuple[int, str]]] = {s: [] for s in slots}
    events: list[Observation | Query] = []

    # Seed every slot once at the front so ground truth exists for all queries.
    seed_order = list(slots)
    rng.shuffle(seed_order)

    def assign_value(slot: tuple[str, str], turn: int, *, is_first: bool) -> Observation:
        entity, attribute = slot
        phrase, values = ATTRIBUTES[attribute]
        # Pick a value different from the current one so updates truly contradict.
        choices = [v for v in values if v != current.get(slot)]
        value = rng.choice(choices)
        current[slot] = value
        history[slot].append((turn, value))
        # Phrase updates in the same frame as the initial fact so lexical
        # relevance tracks the *slot*, not the surface form. Otherwise a stale
        # initial fact would out-match the query on wording alone and the
        # experiment would measure phrasing, not forgetting. "now" marks the
        # update as salient for the importance heuristic.
        if is_first:
            text = f"{entity.title()}'s {phrase} is {value}."
        else:
            text = f"{entity.title()}'s {phrase} is now {value}."
        return Observation(turn, text, entity, attribute, value, is_noise=False)

    def make_noise(turn: int) -> Observation:
        text = f"{rng.choice(_NOISE_SUBJECTS).title()} {rng.choice(_NOISE_PREDICATES)}."
        return Observation(turn, text, None, None, None, is_noise=True)

    def make_query(turn: int) -> Query:
        slot = rng.choice([s for s in slots if current.get(s) is not None])
        entity, attribute = slot
        text = f"What is {entity.title()}'s {_attr_phrase(attribute)}?"
        return Query(turn, text, entity, attribute, current[slot])

    for turn in range(n_turns):
        if turn < len(seed_order):
            events.append(assign_value(seed_order[turn], turn, is_first=True))
            continue

        r = rng.random()
        if r < p_query:
            events.append(make_query(turn))
        elif r < p_query + p_update:
            slot = rng.choice(slots)
            events.append(assign_value(slot, turn, is_first=False))
        else:
            events.append(make_noise(turn))

    return Workload(events=events, seed=seed, n_turns=n_turns, history=history)
