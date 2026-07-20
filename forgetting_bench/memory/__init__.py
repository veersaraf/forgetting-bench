"""Reference memory core: a scored memory stream with pluggable forgetting."""

from .decay import DecayModule, EbbinghausDecay, LastWriteWins, NoDecay
from .embedder import Embedder, HashingEmbedder, SparseVector, cosine, tokenize
from .entry import MemoryEntry
from .extractor import KeywordSlotExtractor, SlotExtractor
from .importance import HeuristicImportance, ImportanceScorer
from .store import MemoryStore, RetrievalWeights, ScoredEntry

__all__ = [
    "DecayModule",
    "EbbinghausDecay",
    "LastWriteWins",
    "NoDecay",
    "KeywordSlotExtractor",
    "SlotExtractor",
    "Embedder",
    "HashingEmbedder",
    "SparseVector",
    "cosine",
    "tokenize",
    "MemoryEntry",
    "HeuristicImportance",
    "ImportanceScorer",
    "MemoryStore",
    "RetrievalWeights",
    "ScoredEntry",
]
