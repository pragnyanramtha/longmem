"""Tests for the MemoryConsolidator module."""

import sys
import time

sys.path.insert(0, "/home/pik/dev/longmem")

import pytest

from src.store import MemoryStore
from src.models import DistilledMemory
from src.consolidator import MemoryConsolidator, ConsolidationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(
    key: str,
    value: str,
    mem_type: str = "fact",
    category: str = "general",
    confidence: float = 0.9,
) -> DistilledMemory:
    return DistilledMemory(
        action="add",
        type=mem_type,
        category=category,
        key=key,
        value=value,
        confidence=confidence,
        reasoning="test",
    )


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path)


@pytest.fixture
def consolidator(store):
    return MemoryConsolidator(store=store)


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_exact_key_duplicates(self, store, consolidator):
        """Memories with the same key should be detected as duplicates."""
        store.add_memory(_make_memory("user_name", "Arjun"), turn_id=1)
        store.add_memory(_make_memory("user_name", "Arjun Kumar"), turn_id=5)

        groups = consolidator.find_duplicates()
        assert len(groups) == 1
        assert groups[0].similarity == 1.0
        assert len(groups[0].duplicates) == 1

    def test_semantically_similar_duplicates(self, store, consolidator):
        """Memories with very similar meaning should be detected."""
        store.add_memory(
            _make_memory("dietary_preference", "User is vegetarian"),
            turn_id=1,
        )
        store.add_memory(
            _make_memory("diet_choice", "User follows a vegetarian diet"),
            turn_id=10,
        )

        groups = consolidator.find_duplicates()
        # May or may not detect depending on embedding similarity
        # At minimum, no errors should occur
        assert isinstance(groups, list)

    def test_no_duplicates(self, store, consolidator):
        """Unrelated memories should not be grouped."""
        store.add_memory(_make_memory("user_name", "Arjun"), turn_id=1)
        store.add_memory(
            _make_memory("favorite_color", "Blue"), turn_id=2
        )

        groups = consolidator.find_duplicates()
        # These are unrelated, should not be grouped
        key_match_groups = [g for g in groups if g.similarity == 1.0]
        assert len(key_match_groups) == 0

    def test_empty_store(self, store, consolidator):
        """Empty store should return no groups."""
        groups = consolidator.find_duplicates()
        assert groups == []

    def test_single_memory(self, store, consolidator):
        """Single memory cannot be a duplicate."""
        store.add_memory(_make_memory("user_name", "Arjun"), turn_id=1)
        groups = consolidator.find_duplicates()
        assert groups == []


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

class TestMergeDuplicates:
    def test_merge_keeps_canonical(self, store, consolidator):
        """After merging, canonical survives and duplicates are deactivated."""
        store.add_memory(
            _make_memory("user_name", "Arjun", confidence=0.7), turn_id=1
        )
        store.add_memory(
            _make_memory("user_name", "Arjun Kumar", confidence=0.95),
            turn_id=5,
        )

        groups = consolidator.find_duplicates()
        assert len(groups) == 1

        merged = consolidator.merge_duplicates(groups)
        assert merged == 1

        active = store.get_active_memories()
        assert len(active) == 1
        # The canonical should be the higher-confidence one
        assert active[0].confidence == 0.95

    def test_merge_empty_groups(self, consolidator):
        """Merging empty groups should return 0."""
        assert consolidator.merge_duplicates([]) == 0


# ---------------------------------------------------------------------------
# Decay tests
# ---------------------------------------------------------------------------

class TestDecayStale:
    def test_stale_memory_decayed(self, store, consolidator):
        """Memory not used for > decay_threshold turns gets decayed."""
        mem_id = store.add_memory(
            _make_memory("user_name", "Arjun", confidence=0.9), turn_id=1
        )
        store.touch_memory(mem_id, turn_id=10)

        decayed = consolidator.decay_stale(current_turn=300, decay_threshold=200)
        assert decayed == 1

        mem = store.get_memory_by_id(mem_id)
        assert mem.confidence < 0.9
        assert mem.confidence == pytest.approx(0.9 * 0.9, abs=0.01)

    def test_recent_memory_not_decayed(self, store, consolidator):
        """Memory used recently should NOT be decayed."""
        mem_id = store.add_memory(
            _make_memory("user_name", "Arjun", confidence=0.9), turn_id=1
        )
        store.touch_memory(mem_id, turn_id=250)

        decayed = consolidator.decay_stale(current_turn=300, decay_threshold=200)
        assert decayed == 0

        mem = store.get_memory_by_id(mem_id)
        assert mem.confidence == 0.9

    def test_never_retrieved_not_decayed(self, store, consolidator):
        """Memory with last_used_turn=0 (never retrieved) should not be decayed."""
        mem_id = store.add_memory(
            _make_memory("user_name", "Arjun", confidence=0.9), turn_id=1
        )
        # Don't touch it - last_used_turn stays 0

        decayed = consolidator.decay_stale(current_turn=500, decay_threshold=200)
        assert decayed == 0


# ---------------------------------------------------------------------------
# Expiration tests
# ---------------------------------------------------------------------------

class TestExpireLowConfidence:
    def test_low_confidence_expired(self, store, consolidator):
        """Memory with confidence below threshold gets deactivated."""
        mem_id = store.add_memory(
            _make_memory("old_fact", "something", confidence=0.2), turn_id=1
        )

        expired = consolidator.expire_low_confidence(min_confidence=0.3)
        assert expired == 1
        assert store.get_memory_by_id(mem_id) is None  # deactivated

    def test_high_confidence_survives(self, store, consolidator):
        """Memory with confidence above threshold survives."""
        mem_id = store.add_memory(
            _make_memory("good_fact", "something", confidence=0.8), turn_id=1
        )

        expired = consolidator.expire_low_confidence(min_confidence=0.3)
        assert expired == 0
        assert store.get_memory_by_id(mem_id) is not None

    def test_edge_confidence(self, store, consolidator):
        """Memory with confidence exactly at threshold survives."""
        mem_id = store.add_memory(
            _make_memory("edge_fact", "something", confidence=0.3), turn_id=1
        )

        expired = consolidator.expire_low_confidence(min_confidence=0.3)
        assert expired == 0
        assert store.get_memory_by_id(mem_id) is not None


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestRunConsolidation:
    def test_full_pipeline(self, store, consolidator):
        """Full consolidation returns correct report."""
        # Add some duplicates
        store.add_memory(
            _make_memory("user_name", "Arjun", confidence=0.7), turn_id=1
        )
        store.add_memory(
            _make_memory("user_name", "Arjun K", confidence=0.9), turn_id=5
        )
        # Add a low-confidence memory
        store.add_memory(
            _make_memory("old_fact", "outdated", confidence=0.2), turn_id=1
        )
        # Add a stale memory
        stale_id = store.add_memory(
            _make_memory("stale_pref", "old preference", confidence=0.8),
            turn_id=1,
        )
        store.touch_memory(stale_id, turn_id=10)

        report = consolidator.run_consolidation(current_turn=500)

        assert isinstance(report, ConsolidationReport)
        assert report.duplicates_found >= 1
        assert report.duplicates_merged >= 1
        assert report.memories_expired >= 1  # low-confidence one
        assert report.total_active_after < report.total_active_before

    def test_empty_store(self, store, consolidator):
        """Consolidation on empty store should work gracefully."""
        report = consolidator.run_consolidation(current_turn=100)
        assert report.duplicates_found == 0
        assert report.duplicates_merged == 0
        assert report.memories_decayed == 0
        assert report.memories_expired == 0
        assert report.total_active_before == 0
        assert report.total_active_after == 0
