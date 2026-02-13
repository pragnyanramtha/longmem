"""Data models for the long-form memory system."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import uuid
import time


# -- Constants --

MEMORY_TYPES = Literal[
    "preference", "fact", "commitment", 
    "constraint", "entity", "instruction"
]

MEMORY_ACTIONS = Literal["add", "update", "keep", "expire"]

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "i", "me", "my",
    "can", "you", "your", "we", "they", "it", "its", "this", "that",
    "in", "on", "at", "to", "for", "of", "with", "and", "or", "but",
    "not", "no", "do", "does", "did", "has", "have", "had", "be",
    "been", "being", "will", "would", "could", "should", "may",
    "might", "shall", "so", "if", "then", "than", "too", "very",
    "just", "about", "up", "out", "how", "what", "when", "where",
    "who", "which", "there", "here", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "also", "into", "over", "after", "before", "between",
})


# -- Data Classes --

@dataclass
class Memory:
    """A single memory unit stored in the database."""
    id: str
    type: str               # one of MEMORY_TYPES
    category: str           # e.g. "language", "schedule", "dietary"
    key: str                # canonical key like "preferred_language"
    value: str              # the actual info like "Kannada"
    source_turn: int        # turn where this was first extracted
    confidence: float       # 0.0 to 1.0
    created_at: float       # unix timestamp
    updated_at: float       # unix timestamp
    is_active: bool = True
    last_used_turn: int = 0  # track when memory was last retrieved

    @staticmethod
    def generate_id() -> str:
        return f"mem_{uuid.uuid4().hex[:8]}"


@dataclass
class DistilledMemory:
    """Output from the distiller before being persisted."""
    action: str             # one of MEMORY_ACTIONS
    type: str
    category: str
    key: str
    value: str
    confidence: float
    reasoning: str


@dataclass 
class RetrievalResult:
    """A memory with its retrieval score."""
    memory: Memory
    score: float


@dataclass
class TurnRecord:
    """Minimal log of a conversation turn."""
    turn_id: int
    role: str               # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    memories_retrieved: list[str] = field(default_factory=list)  # memory IDs
