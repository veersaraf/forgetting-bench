"""Slot extraction -- deriving structured (entity, attribute) keys from text.

This is the honest crux of the benchmark. A real memory system does not receive
clean slot keys; it *extracts* them from natural language, and that extraction is
imperfect. Contradiction supersession can only fire when two statements about the
same real-world slot are extracted to the *same* key -- so whenever extraction
misses (a paraphrase, an implicit/numeric update that never restates the
attribute), the stale fact is never linked and survives as a live contradiction.

That imperfection is what makes the contradiction metric something the decay
mechanism can actually *fail* at, rather than a tautology. The default extractor
is a transparent keyword matcher; :class:`SlotExtractor` is the seam for an
LLM-backed extractor that would close some (never all) of the gap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .embedder import tokenize


class SlotExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> tuple[str, str] | None:
        """Return an (entity, attribute) key, or None if it can't be resolved."""


class KeywordSlotExtractor(SlotExtractor):
    """Matches known entity names and *canonical* attribute phrases in the text.

    It resolves the canonical statement form ("Alice's home city is denver") but
    has no idea that "Alice relocated to denver" or "Alice got a raise to 110k"
    are about the same slots -- exactly the blind spot a keyword/regex extractor
    has in production. Both an entity and an attribute must be found, or the
    statement is left unlinked (returns None).
    """

    def __init__(
        self,
        entities: list[str],
        attribute_phrases: dict[str, str],
    ) -> None:
        # entity name (lowercased) -> canonical entity id
        self.entities = {e.lower(): e for e in entities}
        # canonical phrase (lowercased) -> attribute id, longest first so that a
        # multi-word phrase wins over any substring of it.
        self.attribute_phrases = dict(
            sorted(attribute_phrases.items(), key=lambda kv: -len(kv[0]))
        )

    def extract(self, text: str) -> tuple[str, str] | None:
        low = text.lower()
        token_set = set(tokenize(low))

        entity = None
        for name, canonical in self.entities.items():
            if name in token_set:
                entity = canonical
                break
        if entity is None:
            return None

        for phrase, attribute in self.attribute_phrases.items():
            if phrase in low:
                return (entity, attribute)
        return None
