"""Memory consolidation: deduplication, staleness decay, and expiration.

Over long conversations, memories accumulate duplicates, go stale, and clutter
retrieval results.  MemoryConsolidator runs periodic maintenance to keep the
memory store lean and high-quality.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .models import Memory
from .store import MemoryStore

logger = logging.getLogger("atlas.consolidator")


# ---------------------------------------------------------------------------
# Data classes (local to this module -- not added to models.py)
# ---------------------------------------------------------------------------

@dataclass
class DuplicateGroup:
    """A cluster of memories that represent the same underlying fact."""
    canonical: Memory           # The "best" memory to keep
    duplicates: list[Memory]    # Memories to deactivate
    similarity: float           # How similar they are (0-1)


@dataclass
class ConsolidationReport:
    """Summary of a single consolidation run."""
    duplicates_found: int
    duplicates_merged: int
    memories_decayed: int
    memories_expired: int
    total_active_before: int
    total_active_after: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Periodic maintenance for the memory store.

    Responsibilities:
    1. **Duplicate detection** -- find memories that say the same thing.
    2. **Duplicate merging** -- keep the best version, deactivate the rest.
    3. **Staleness decay** -- reduce confidence of memories not used recently.
    4. **Expiration** -- deactivate memories whose confidence drops too low.
    """

    SIMILARITY_THRESHOLD = 0.85

    def __init__(
        self,
        store: MemoryStore,
        client: Any = None,
        model: str = "llama-3.3-70b-versatile",
        provider: str = "groq",
    ):
        self.store = store
        self.client = client
        self.model = model
        self.provider = provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_consolidation(self, current_turn: int) -> ConsolidationReport:
        """Run all consolidation steps and return a summary report."""
        total_before = self.store.active_count()
        logger.info(
            "Starting consolidation at turn %d (%d active memories)",
            current_turn,
            total_before,
        )

        # Step 1 & 2: duplicates
        groups = self.find_duplicates()
        duplicates_found = len(groups)
        duplicates_merged = self.merge_duplicates(groups)

        # Step 3: decay
        memories_decayed = self.decay_stale(current_turn)

        # Step 4: expire
        memories_expired = self.expire_low_confidence()

        total_after = self.store.active_count()

        report = ConsolidationReport(
            duplicates_found=duplicates_found,
            duplicates_merged=duplicates_merged,
            memories_decayed=memories_decayed,
            memories_expired=memories_expired,
            total_active_before=total_before,
            total_active_after=total_after,
        )
        logger.info(
            "Consolidation complete: %d duplicates merged, %d decayed, %d expired "
            "(%d -> %d active)",
            duplicates_merged,
            memories_decayed,
            memories_expired,
            total_before,
            total_after,
        )
        return report

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def find_duplicates(self) -> list[DuplicateGroup]:
        """Find groups of memories that are duplicates.

        Uses a two-pass approach for efficiency:
        1. **Key pass** -- exact key match (definite duplicates).
        2. **Vector pass** -- for each memory, search the store's vector index
           for semantically similar memories (similarity > threshold).
        """
        active = self.store.get_active_memories()
        if len(active) < 2:
            return []

        grouped_ids: set[str] = set()
        groups: list[DuplicateGroup] = []

        # --- Pass 1: exact key match ---
        by_key: dict[str, list[Memory]] = defaultdict(list)
        for mem in active:
            by_key[mem.key].append(mem)

        for key, mems in by_key.items():
            if len(mems) < 2:
                continue
            canonical, duplicates = self._pick_canonical(mems)
            groups.append(
                DuplicateGroup(
                    canonical=canonical,
                    duplicates=duplicates,
                    similarity=1.0,
                )
            )
            for m in mems:
                grouped_ids.add(m.id)
            logger.debug(
                "Key-match group: key=%s, canonical=%s, %d duplicates",
                key,
                canonical.id,
                len(duplicates),
            )

        # --- Pass 2: vector similarity ---
        for mem in active:
            if mem.id in grouped_ids:
                continue

            embed_text = f"{mem.key}: {mem.value}"
            try:
                results = self.store.search_vector(embed_text, top_k=10)
            except Exception:
                logger.debug("Vector search failed for %s; skipping", mem.id)
                continue

            similar: list[Memory] = []
            similarities: list[float] = []

            for hit_id, distance in results:
                if hit_id == mem.id or hit_id in grouped_ids:
                    continue
                hit_mem = self.store.get_memory_by_id(hit_id)
                if hit_mem is None:
                    continue

                embed_a = self.store.embed(embed_text)
                embed_b = self.store.embed(f"{hit_mem.key}: {hit_mem.value}")
                sim = _cosine_similarity(embed_a, embed_b)

                if sim >= self.SIMILARITY_THRESHOLD:
                    similar.append(hit_mem)
                    similarities.append(sim)

            if not similar:
                continue

            all_mems = [mem] + similar
            canonical, duplicates = self._pick_canonical(all_mems)
            avg_sim = sum(similarities) / len(similarities)
            groups.append(
                DuplicateGroup(
                    canonical=canonical,
                    duplicates=duplicates,
                    similarity=avg_sim,
                )
            )
            for m in all_mems:
                grouped_ids.add(m.id)
            logger.debug(
                "Vector-match group: canonical=%s (key=%s), %d duplicates, avg_sim=%.3f",
                canonical.id,
                canonical.key,
                len(duplicates),
                avg_sim,
            )

        return groups

    # ------------------------------------------------------------------
    # Duplicate merging
    # ------------------------------------------------------------------

    def merge_duplicates(self, groups: list[DuplicateGroup]) -> int:
        """Deactivate the duplicate memories in each group.

        The canonical memory is kept.  Returns the total number of memories
        deactivated across all groups.
        """
        total_merged = 0
        for group in groups:
            for dup in group.duplicates:
                self._deactivate_memory(dup)
                logger.debug(
                    "Deactivated duplicate %s (key=%s, value=%s) "
                    "in favour of canonical %s",
                    dup.id,
                    dup.key,
                    dup.value,
                    group.canonical.id,
                )
                total_merged += 1
        return total_merged

    # ------------------------------------------------------------------
    # Staleness decay
    # ------------------------------------------------------------------

    def decay_stale(
        self,
        current_turn: int,
        decay_threshold: int = 200,
        decay_factor: float = 0.9,
    ) -> int:
        """Reduce confidence of memories not used in *decay_threshold* turns.

        Only memories with ``last_used_turn > 0`` (i.e. that have been
        retrieved at least once) are eligible for decay.  A memory that was
        *never* retrieved is left alone -- it may simply not have been needed
        yet.

        Returns the number of memories whose confidence was reduced.
        """
        active = self.store.get_active_memories()
        decayed = 0

        for mem in active:
            if mem.last_used_turn <= 0:
                continue
            turns_since_use = current_turn - mem.last_used_turn
            if turns_since_use <= decay_threshold:
                continue

            new_conf = round(mem.confidence * decay_factor, 6)
            self.store.db.execute(
                "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
                (new_conf, time.time(), mem.id),
            )
            logger.debug(
                "Decayed memory %s (key=%s): %.3f -> %.3f "
                "(last used turn %d, current turn %d)",
                mem.id,
                mem.key,
                mem.confidence,
                new_conf,
                mem.last_used_turn,
                current_turn,
            )
            decayed += 1

        if decayed:
            self.store.db.commit()
        return decayed

    # ------------------------------------------------------------------
    # Expiration
    # ------------------------------------------------------------------

    def expire_low_confidence(self, min_confidence: float = 0.3) -> int:
        """Deactivate memories whose confidence has fallen below *min_confidence*.

        Returns the number of memories expired.
        """
        active = self.store.get_active_memories()
        expired = 0

        for mem in active:
            if mem.confidence >= min_confidence:
                continue
            self._deactivate_memory(mem)
            logger.debug(
                "Expired memory %s (key=%s, confidence=%.3f < %.3f)",
                mem.id,
                mem.key,
                mem.confidence,
                min_confidence,
            )
            expired += 1

        return expired

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_canonical(
        memories: list[Memory],
    ) -> tuple[Memory, list[Memory]]:
        """Choose the best memory as canonical.

        Selection criteria (in order):
        1. Highest confidence.
        2. Most recently updated (``updated_at``).
        3. Arbitrary (first in sorted order).
        """
        ranked = sorted(
            memories,
            key=lambda m: (m.confidence, m.updated_at),
            reverse=True,
        )
        canonical = ranked[0]
        duplicates = ranked[1:]
        return canonical, duplicates

    def _deactivate_memory(self, mem: Memory) -> None:
        """Deactivate a single memory by ID via direct SQL."""
        self.store.db.execute(
            "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?",
            (time.time(), mem.id),
        )
        self.store.db.commit()
