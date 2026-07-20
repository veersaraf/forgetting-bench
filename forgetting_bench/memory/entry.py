"""The atomic unit of the memory stream."""

from __future__ import annotations

from dataclasses import dataclass, field

from .embedder import SparseVector


@dataclass
class MemoryEntry:
    """A single timestamped observation in the memory stream.

    ``entity`` and ``attribute`` are optional structured keys describing *what
    slot* a statement is about (e.g. entity="alice", attribute="city"). When
    present they let the decay module detect contradictions: a newer entry about
    the same (entity, attribute) supersedes older ones. In a production system
    these keys come from an extraction step (see ``memory.importance`` for the
    analogous LLM seam); the benchmark supplies them directly so results are
    deterministic and the forgetting logic can be measured in isolation.
    """

    id: int
    turn: int
    text: str
    importance: float
    entity: str | None = None
    attribute: str | None = None

    # Populated by the store when the entry is added.
    embedding: SparseVector = field(default_factory=SparseVector, repr=False)
    last_access: int = 0

    # Managed by the decay module. ``superseded_by`` holds the id of the newer
    # entry that overrode this one's (entity, attribute) slot, or None.
    superseded_by: int | None = None

    @property
    def slot(self) -> tuple[str, str] | None:
        """The (entity, attribute) key this entry is about, if structured."""
        if self.entity is None or self.attribute is None:
            return None
        return (self.entity, self.attribute)

    def token_count(self) -> int:
        """Whitespace token count -- the unit used for the memory-bloat metric."""
        return len(self.text.split())
