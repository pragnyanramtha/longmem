"""Tests for the relevance-scored memory retrieval pipeline."""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/pik/dev/longmem")

import pytest

from src.models import DistilledMemory, RetrievalResult
from src.store import MemoryStore
from src.retriever import MemoryRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path=db_path)


@pytest.fixture()
def retriever(store):
    return MemoryRetriever(store)


def _make_memory(
    key: str,
    value: str,
    category: str = "general",
    mem_type: str = "fact",
    confidence: float = 0.9,
) -> DistilledMemory:
    return DistilledMemory(
        action="add",
        type=mem_type,
        category=category,
        key=key,
        value=value,
        confidence=confidence,
        reasoning="test seed",
    )


def _seed_standard_memories(store: MemoryStore) -> list[str]:
    """Insert a curated set of memories and return their IDs."""
    memories = [
        _make_memory(
            key="preferred_language",
            value="The user prefers to communicate in Kannada",
            category="language",
            mem_type="preference",
            confidence=0.95,
        ),
        _make_memory(
            key="favorite_food",
            value="The user loves spicy South Indian food especially dosa and sambar",
            category="dietary",
            mem_type="preference",
            confidence=0.90,
        ),
        _make_memory(
            key="pet_name",
            value="The user has a golden retriever dog named Bruno",
            category="personal",
            mem_type="fact",
            confidence=0.85,
        ),
        _make_memory(
            key="work_schedule",
            value="The user works Monday through Friday from 9am to 5pm",
            category="schedule",
            mem_type="fact",
            confidence=0.80,
        ),
        _make_memory(
            key="allergies",
            value="The user is allergic to peanuts and shellfish",
            category="health",
            mem_type="constraint",
            confidence=0.99,
        ),
        _make_memory(
            key="hobby_painting",
            value="The user enjoys watercolor painting on weekends",
            category="hobby",
            mem_type="fact",
            confidence=0.70,
        ),
        _make_memory(
            key="timezone",
            value="The user lives in IST timezone India",
            category="location",
            mem_type="fact",
            confidence=0.88,
        ),
    ]
    ids = []
    for i, mem in enumerate(memories):
        mem_id = store.add_memory(mem, turn_id=i + 1)
        ids.append(mem_id)
    return ids


# ---------------------------------------------------------------------------
# 1. Basic retrieval
# ---------------------------------------------------------------------------

class TestBasicRetrieval:
    def test_returns_results_for_matching_query(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve("What language does the user prefer?")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_positive_scores(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve("Tell me about the user's dog")
        assert all(r.score > 0 for r in results)

    def test_results_sorted_descending(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve("food preferences")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 2. Relevant memories score higher
# ---------------------------------------------------------------------------

class TestRelevanceRanking:
    def test_language_query_ranks_language_memory_first(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve("What language does the user speak?")
        assert len(results) > 0
        top_key = results[0].memory.key
        assert top_key == "preferred_language"

    def test_allergy_query_ranks_allergy_memory_highly(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve(
            "Does the user have any allergies or food restrictions?"
        )
        assert len(results) > 0
        top_keys = [r.memory.key for r in results[:2]]
        assert "allergies" in top_keys


# ---------------------------------------------------------------------------
# 3. Minimum threshold filtering
# ---------------------------------------------------------------------------

class TestMinimumThreshold:
    def test_high_threshold_filters_all(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve(
            "What language does the user speak?",
            min_score=999.0,
        )
        assert len(results) == 0

    def test_zero_threshold_returns_more(self, store, retriever):
        _seed_standard_memories(store)
        results_strict = retriever.retrieve(
            "What is the user's schedule?",
            min_score=0.5,
        )
        results_loose = retriever.retrieve(
            "What is the user's schedule?",
            min_score=0.0,
        )
        assert len(results_loose) >= len(results_strict)


# ---------------------------------------------------------------------------
# 4. Recency boost
# ---------------------------------------------------------------------------

class TestRecencyBoost:
    def test_recently_used_memory_scores_higher(self, store, retriever):
        mem_a = _make_memory(
            key="color_pref_old",
            value="The user likes the color blue",
            confidence=0.85,
        )
        mem_b = _make_memory(
            key="color_pref_new",
            value="The user likes the color blue very much",
            confidence=0.85,
        )
        id_a = store.add_memory(mem_a, turn_id=1)
        id_b = store.add_memory(mem_b, turn_id=2)

        store.touch_memory(id_b, turn_id=90)

        results = retriever.retrieve(
            "What color does the user like?",
            current_turn=100,
            top_k=5,
        )
        assert len(results) >= 2

        score_map = {r.memory.id: r.score for r in results}
        assert score_map[id_b] > score_map[id_a]

    def test_no_recency_without_current_turn(self, store, retriever):
        mem = _make_memory(key="recency_test", value="User likes jazz music")
        mem_id = store.add_memory(mem, turn_id=1)
        store.touch_memory(mem_id, turn_id=50)

        results_no = retriever.retrieve("jazz music", current_turn=None)
        results_yes = retriever.retrieve("jazz music", current_turn=100)

        assert len(results_no) > 0
        assert len(results_yes) > 0
        assert results_yes[0].score > results_no[0].score


# ---------------------------------------------------------------------------
# 5. Confidence weighting
# ---------------------------------------------------------------------------

class TestConfidenceWeighting:
    def test_high_confidence_scores_higher(self, store, retriever):
        mem_high = _make_memory(
            key="meeting_high",
            value="Team meeting is every Wednesday afternoon",
            confidence=0.99,
        )
        mem_low = _make_memory(
            key="meeting_low",
            value="Team meeting is every Wednesday afternoon",
            confidence=0.10,
        )
        id_high = store.add_memory(mem_high, turn_id=1)
        id_low = store.add_memory(mem_low, turn_id=2)

        results = retriever.retrieve("When is the team meeting?", top_k=5)
        assert len(results) >= 2

        score_map = {r.memory.id: r.score for r in results}
        assert score_map[id_high] > score_map[id_low]


# ---------------------------------------------------------------------------
# 6. Empty store
# ---------------------------------------------------------------------------

class TestEmptyStore:
    def test_empty_store_returns_empty(self, store, retriever):
        assert retriever.retrieve("anything at all") == []

    def test_empty_store_with_params(self, store, retriever):
        results = retriever.retrieve(
            "anything", top_k=10, min_score=0.0, current_turn=100,
        )
        assert results == []


# ---------------------------------------------------------------------------
# 7. Dynamic top_k
# ---------------------------------------------------------------------------

class TestDynamicTopK:
    def test_top_k_caps_results(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve(
            "Tell me everything about the user",
            top_k=2,
            min_score=0.0,
        )
        assert len(results) <= 2

    def test_very_high_threshold_returns_zero(self, store, retriever):
        _seed_standard_memories(store)
        results = retriever.retrieve(
            "What is the user's work schedule?",
            top_k=10,
            min_score=100.0,
        )
        assert len(results) == 0
