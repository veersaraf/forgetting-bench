"""A reproducible, seeded long-horizon workload.

The stream mixes three things that stress *forgetting* rather than recall:

* **facts** -- the first assertion about an (entity, attribute) slot,
* **updates** -- a later assertion that changes a slot's value, making every
  earlier value for that slot *stale* (a contradiction if still retrieved),
* **noise** -- distractor observations about nothing the queries ever ask for.

Crucially, updates come in two flavours:

* **canonical** ("Alice's home city is now denver") -- restates the attribute
  phrase, so a keyword extractor links it to the slot and supersession fires,
* **hard** ("Alice relocated to denver", "Alice got a raise to 110k") -- a
  paraphrase or an implicit/numeric update that never restates the attribute, so
  extraction misses and the stale fact is left un-superseded.

That hard fraction is what makes the contradiction metric something a memory
system can genuinely fail at. Interleaved **queries** ask for the current value
of a slot; because we build the stream we know the ground truth at every turn.
Everything is driven by one seeded RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..memory.extractor import KeywordSlotExtractor, SlotExtractor


@dataclass(frozen=True)
class AttrSpec:
    key: str
    phrase: str                 # canonical phrase the extractor knows
    values: list[str]
    hard_templates: list[str]   # paraphrases that omit the phrase (extractor misses)


ENTITIES = [
    "alice", "bob", "carol", "dan", "erin",
    "frank", "grace", "heidi", "ivan", "judy",
]

ATTRIBUTES: dict[str, AttrSpec] = {
    "home_city": AttrSpec(
        "home_city", "home city",
        ["boston", "denver", "austin", "seattle", "miami", "portland"],
        ["{E} relocated to {v}.", "{E} settled down in {v}."],
    ),
    "job_title": AttrSpec(
        "job_title", "job title",
        ["engineer", "teacher", "nurse", "pilot", "chef", "lawyer"],
        ["{E} started working as a {v}.", "{E} was promoted to {v}."],
    ),
    "favorite_food": AttrSpec(
        "favorite_food", "favorite food",
        ["ramen", "tacos", "sushi", "pizza", "curry", "falafel"],
        ["{E} cannot stop eating {v} lately.", "{E} is obsessed with {v} now."],
    ),
    "phone_model": AttrSpec(
        "phone_model", "phone model",
        ["pixel", "iphone", "galaxy", "nokia", "oneplus", "moto"],
        ["{E} switched to a {v}.", "{E} upgraded to a {v}."],
    ),
    "current_project": AttrSpec(
        "current_project", "current project",
        ["atlas", "orbit", "delta", "nimbus", "vertex", "cobalt"],
        ["{E} moved onto {v}.", "{E} now leads {v}."],
    ),
    "salary": AttrSpec(
        "salary", "salary",
        ["90k", "100k", "110k", "120k", "130k", "140k"],
        ["{E} got a raise to {v}.", "{E} now earns {v}."],
    ),
}

# Distractor material -- disjoint from the tracked vocabulary and containing no
# entity names, so the extractor correctly leaves it unslotted.
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
    is_hard: bool = False   # True if this update is phrased to defeat extraction


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
    p_hard: float
    history: dict[tuple[str, str], list[tuple[int, str]]] = field(default_factory=dict)

    @property
    def observations(self) -> list[Observation]:
        return [e for e in self.events if isinstance(e, Observation)]

    @property
    def queries(self) -> list[Query]:
        return [e for e in self.events if isinstance(e, Query)]


def default_extractor() -> SlotExtractor:
    """The keyword extractor the memory core uses on this workload's vocabulary."""
    return KeywordSlotExtractor(
        entities=ENTITIES,
        attribute_phrases={spec.phrase: spec.key for spec in ATTRIBUTES.values()},
    )


def generate_workload(
    seed: int = 0,
    n_turns: int = 3000,
    p_query: float = 0.12,
    p_update: float = 0.18,
    p_hard: float = 0.4,
) -> Workload:
    """Build a deterministic stream of ``n_turns`` events.

    ``p_hard`` is the fraction of *updates* phrased to defeat keyword extraction
    (paraphrase / implicit / numeric). Every slot is seeded once with a canonical
    fact before random updates begin, so queries always have a ground truth.
    """
    rng = random.Random(seed)
    slots = [(e, a) for e in ENTITIES for a in ATTRIBUTES]

    current: dict[tuple[str, str], str] = {}
    history: dict[tuple[str, str], list[tuple[int, str]]] = {s: [] for s in slots}
    events: list[Observation | Query] = []

    seed_order = list(slots)
    rng.shuffle(seed_order)

    def assign_value(slot: tuple[str, str], turn: int, *, is_first: bool) -> Observation:
        entity, attribute = slot
        spec = ATTRIBUTES[attribute]
        value = rng.choice([v for v in spec.values if v != current.get(slot)])
        current[slot] = value
        history[slot].append((turn, value))
        title = entity.title()
        hard = False
        if is_first:
            text = f"{title}'s {spec.phrase} is {value}."
        elif rng.random() < p_hard:
            text = rng.choice(spec.hard_templates).format(E=title, v=value)
            hard = True
        else:
            text = f"{title}'s {spec.phrase} is now {value}."
        return Observation(turn, text, entity, attribute, value, is_noise=False, is_hard=hard)

    def make_noise(turn: int) -> Observation:
        text = f"{rng.choice(_NOISE_SUBJECTS).title()} {rng.choice(_NOISE_PREDICATES)}."
        return Observation(turn, text, None, None, None, is_noise=True)

    def make_query(turn: int) -> Query:
        slot = rng.choice([s for s in slots if current.get(s) is not None])
        entity, attribute = slot
        text = f"What is {entity.title()}'s {ATTRIBUTES[attribute].phrase}?"
        return Query(turn, text, entity, attribute, current[slot])

    for turn in range(n_turns):
        if turn < len(seed_order):
            events.append(assign_value(seed_order[turn], turn, is_first=True))
            continue
        r = rng.random()
        if r < p_query:
            events.append(make_query(turn))
        elif r < p_query + p_update:
            events.append(assign_value(rng.choice(slots), turn, is_first=False))
        else:
            events.append(make_noise(turn))

    return Workload(events=events, seed=seed, n_turns=n_turns, p_hard=p_hard, history=history)
