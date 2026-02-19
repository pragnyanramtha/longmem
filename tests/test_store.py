"""Tests for the MemoryStore module."""

import sys
import time
import os

sys.path.insert(0, "/home/pik/dev/longmem")

import pytest

from src.store import MemoryStore
from src.models import DistilledMemory, Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(
    key: str = "user_name",
    value: str = "Arjun",
    mem_type: str = "fact",
    category: str = "personal",
    confidence: float = 0.9,
    action: str = "add",
) -> DistilledMemory:
    return DistilledMemory(
        action=action,
        type=mem_type,
        category=category,
        key=key,
        value=value,
        confidence=confidence,
        reasoning="test",
    )


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_creates_database(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        store = MemoryStore(db_path)
        assert os.path.exists(db_path)

    def test_tables_created(self, store):
        tables = store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "memories" in table_names
        assert "profile" in table_names
        assert "turns" in table_names


# ---------------------------------------------------------------------------
# add_memory
# ---------------------------------------------------------------------------

class TestAddMemory:
    def test_add_memory_returns_id(self, store):
        mem = _make_memory()
        mem_id = store.add_memory(mem, turn_id=1)
        assert mem_id.startswith("mem_")

    def test_add_memory_appears_in_active(self, store):
        mem = _make_memory(key="user_name", value="Arjun")
        store.add_memory(mem, turn_id=1)

        active = store.get_active_memories()
        assert len(active) == 1
        assert active[0].key == "user_name"
        assert active[0].value == "Arjun"

    def test_add_memory_sets_timestamps(self, store):
        mem = _make_memory()
        mem_id = store.add_memory(mem, turn_id=1)

        stored = store.get_memory_by_id(mem_id)
        assert stored.created_at > 0
        assert stored.updated_at > 0
        assert stored.source_turn == 1

    def test_add_preference_updates_profile(self, store):
        mem = _make_memory(
            key="fav_color", value="blue", mem_type="preference"
        )
        store.add_memory(mem, turn_id=1)

        profile = store.get_profile()
        assert "fav_color" in profile
        assert profile["fav_color"] == "blue"

    def test_add_non_preference_no_profile(self, store):
        mem = _make_memory(
            key="meeting_tue", value="3 PM", mem_type="commitment"
        )
        store.add_memory(mem, turn_id=1)

        profile = store.get_profile()
        assert "meeting_tue" not in profile


# ---------------------------------------------------------------------------
# deactivate_by_key
# ---------------------------------------------------------------------------

class TestDeactivate:
    def test_deactivate_removes_from_active(self, store):
        mem = _make_memory(key="user_name", value="Arjun")
        store.add_memory(mem, turn_id=1)
        assert store.active_count() == 1

        store.deactivate_by_key("user_name")
        assert store.active_count() == 0

    def test_deactivate_nonexistent_key(self, store):
        # Should not raise
        store.deactivate_by_key("nonexistent")
        assert store.active_count() == 0


# ---------------------------------------------------------------------------
# touch_memory
# ---------------------------------------------------------------------------

class TestTouchMemory:
    def test_touch_updates_last_used_turn(self, store):
        mem = _make_memory()
        mem_id = store.add_memory(mem, turn_id=1)

        stored = store.get_memory_by_id(mem_id)
        assert stored.last_used_turn == 0

        store.touch_memory(mem_id, turn_id=42)
        stored = store.get_memory_by_id(mem_id)
        assert stored.last_used_turn == 42


# ---------------------------------------------------------------------------
# find_by_key
# ---------------------------------------------------------------------------

class TestFindByKey:
    def test_find_existing_key(self, store):
        mem = _make_memory(key="user_name", value="Arjun")
        store.add_memory(mem, turn_id=1)

        found = store.find_by_key("user_name")
        assert found is not None
        assert found.value == "Arjun"

    def test_find_missing_key(self, store):
        found = store.find_by_key("nonexistent")
        assert found is None

    def test_find_deactivated_key(self, store):
        mem = _make_memory(key="old_key", value="old_value")
        store.add_memory(mem, turn_id=1)
        store.deactivate_by_key("old_key")

        found = store.find_by_key("old_key")
        assert found is None


# ---------------------------------------------------------------------------
# Search: vector
# ---------------------------------------------------------------------------

class TestSearchVector:
    def test_vector_search_returns_results(self, store):
        store.add_memory(
            _make_memory(key="dietary_preference", value="vegetarian"),
            turn_id=1,
        )
        store.add_memory(
            _make_memory(key="user_name", value="Arjun"),
            turn_id=2,
        )

        results = store.search_vector("food diet", top_k=5)
        assert len(results) > 0
        # Results are (memory_id, distance) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_vector_search_empty_store(self, store):
        results = store.search_vector("anything", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Search: FTS
# ---------------------------------------------------------------------------

class TestSearchFTS:
    def test_fts_search_returns_results(self, store):
        store.add_memory(
            _make_memory(key="dietary_preference", value="vegetarian"),
            turn_id=1,
        )

        results = store.search_fts("vegetarian", top_k=5)
        assert len(results) > 0

    def test_fts_search_no_match(self, store):
        store.add_memory(
            _make_memory(key="user_name", value="Arjun"),
            turn_id=1,
        )

        results = store.search_fts("xylophone", top_k=5)
        assert results == []

    def test_fts_stopwords_only(self, store):
        """Query with only stopwords should return empty."""
        store.add_memory(
            _make_memory(key="user_name", value="Arjun"),
            turn_id=1,
        )

        results = store.search_fts("the is a", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# rowid_to_memory_id
# ---------------------------------------------------------------------------

class TestRowidMapping:
    def test_valid_rowid(self, store):
        mem = _make_memory()
        mem_id = store.add_memory(mem, turn_id=1)

        # Get the rowid
        row = store.db.execute(
            "SELECT rowid FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        rowid = row[0]

        mapped_id = store.rowid_to_memory_id(rowid)
        assert mapped_id == mem_id

    def test_invalid_rowid(self, store):
        mapped_id = store.rowid_to_memory_id(99999)
        assert mapped_id is None


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_profile_from_preferences(self, store):
        store.add_memory(
            _make_memory(
                key="fav_color", value="blue", mem_type="preference"
            ),
            turn_id=1,
        )
        store.add_memory(
            _make_memory(
                key="location", value="Mumbai", mem_type="fact"
            ),
            turn_id=2,
        )

        profile = store.get_profile()
        assert profile["fav_color"] == "blue"
        assert profile["location"] == "Mumbai"

    def test_empty_profile(self, store):
        profile = store.get_profile()
        assert profile == {}


# ---------------------------------------------------------------------------
# Turn logging
# ---------------------------------------------------------------------------

class TestTurnLogging:
    def test_log_and_retrieve_turn(self, store):
        store.log_turn(1, "user", "Hello there")
        store.log_turn(2, "assistant", "Hi! How can I help?")

        last_id = store.get_last_turn_id()
        assert last_id == 2

    def test_empty_turns(self, store):
        last_id = store.get_last_turn_id()
        assert last_id == 0

    def test_log_with_memories(self, store):
        store.log_turn(
            1, "user", "Hello",
            memories_retrieved=["mem_abc", "mem_def"],
        )

        last_id = store.get_last_turn_id()
        assert last_id == 1


# ---------------------------------------------------------------------------
# active_count
# ---------------------------------------------------------------------------

class TestActiveCount:
    def test_count_after_adds(self, store):
        assert store.active_count() == 0

        store.add_memory(_make_memory(key="a", value="1"), turn_id=1)
        assert store.active_count() == 1

        store.add_memory(_make_memory(key="b", value="2"), turn_id=2)
        assert store.active_count() == 2

    def test_count_after_deactivation(self, store):
        store.add_memory(_make_memory(key="a", value="1"), turn_id=1)
        store.add_memory(_make_memory(key="b", value="2"), turn_id=2)
        assert store.active_count() == 2

        store.deactivate_by_key("a")
        assert store.active_count() == 1


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_write_snapshot_creates_file(self, store, tmp_path):
        store.add_memory(
            _make_memory(key="user_name", value="Arjun"), turn_id=1
        )

        snapshot_dir = str(tmp_path / "snapshots")
        store.write_snapshot(1, snapshot_dir=snapshot_dir)

        expected_file = os.path.join(snapshot_dir, "turn_00001.md")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            content = f.read()
        assert "Arjun" in content
