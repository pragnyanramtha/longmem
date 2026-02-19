"""Tests for the two-pass memory distillation pipeline."""

import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/pik/dev/longmem")

import pytest

from src.distiller import MemoryDistiller
from src.models import Memory, DistilledMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(pass1_response: dict, pass2_response: dict | None = None):
    """Create a mock LLM client that returns preset JSON responses.

    If pass2_response is None, defaults to accepting all candidates.
    """
    client = MagicMock()

    responses = []

    # Pass 1 response
    mock_resp1 = MagicMock()
    mock_resp1.choices = [MagicMock()]
    mock_resp1.choices[0].message.content = json.dumps(pass1_response)
    mock_resp1.usage = None
    responses.append(mock_resp1)

    # Pass 2 response
    if pass2_response is not None:
        mock_resp2 = MagicMock()
        mock_resp2.choices = [MagicMock()]
        mock_resp2.choices[0].message.content = json.dumps(pass2_response)
        mock_resp2.usage = None
        responses.append(mock_resp2)
    else:
        # Default: accept everything from pass 1
        candidates = pass1_response.get("memories", [])
        validations = [
            {"key": m.get("key", ""), "verdict": "accept", "reason": "test"}
            for m in candidates
            if m.get("action", "add") not in ("keep", "expire")
        ]
        mock_resp2 = MagicMock()
        mock_resp2.choices = [MagicMock()]
        mock_resp2.choices[0].message.content = json.dumps({"validations": validations})
        mock_resp2.usage = None
        responses.append(mock_resp2)

    client.chat.completions.create = MagicMock(side_effect=responses)
    return client


def _make_existing_memory(
    key: str = "user_name",
    value: str = "Arjun",
    mem_type: str = "fact",
) -> Memory:
    return Memory(
        id="mem_test0001",
        type=mem_type,
        category="personal",
        key=key,
        value=value,
        source_turn=1,
        confidence=0.9,
        created_at=1000.0,
        updated_at=1000.0,
        is_active=True,
        last_used_turn=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTwoPassPipeline:
    def test_world_facts_rejected_in_pass2(self):
        """World facts extracted in pass 1 should be rejected by pass 2."""
        pass1 = {"memories": [
            {
                "action": "add",
                "type": "fact",
                "category": "general",
                "key": "wifi_explanation",
                "value": "WiFi uses radio waves to transmit data",
                "confidence": 0.5,
                "reasoning": "User asked about WiFi",
            }
        ]}
        pass2 = {"validations": [
            {"key": "wifi_explanation", "verdict": "reject", "reason": "World fact, not user-specific"}
        ]}

        client = _make_mock_client(pass1, pass2)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: How does WiFi work?\n\nASSISTANT: WiFi uses radio waves...",
            existing_memories=[],
            start_turn=1,
            end_turn=2,
        )

        assert len(results) == 0

    def test_meta_noise_rejected(self):
        """Conversation meta-noise should be rejected in pass 2."""
        pass1 = {"memories": [
            {
                "action": "add",
                "type": "fact",
                "category": "general",
                "key": "topic_discussed_chess",
                "value": "User asked about chess strategies",
                "confidence": 0.5,
                "reasoning": "User showed interest in chess",
            }
        ]}
        pass2 = {"validations": [
            {"key": "topic_discussed_chess", "verdict": "reject", "reason": "Meta-noise, not a personal fact"}
        ]}

        client = _make_mock_client(pass1, pass2)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: Tell me about chess strategies\n\nASSISTANT: Here are some opening strategies...",
            existing_memories=[],
            start_turn=1,
            end_turn=2,
        )

        assert len(results) == 0

    def test_transient_details_rejected(self):
        """Transient conversational details should be rejected."""
        pass1 = {"memories": [
            {
                "action": "add",
                "type": "fact",
                "category": "technical",
                "key": "current_debugging",
                "value": "User is debugging a TypeError in their Python code",
                "confidence": 0.6,
                "reasoning": "User mentioned debugging",
            }
        ]}
        pass2 = {"validations": [
            {"key": "current_debugging", "verdict": "reject", "reason": "Transient detail, not durable"}
        ]}

        client = _make_mock_client(pass1, pass2)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: I'm getting a TypeError on line 42, can you help?\n\nASSISTANT: Sure...",
            existing_memories=[],
            start_turn=1,
            end_turn=2,
        )

        assert len(results) == 0

    def test_real_user_facts_pass_through(self):
        """Genuine user facts should pass both passes."""
        pass1 = {"memories": [
            {
                "action": "add",
                "type": "preference",
                "category": "dietary",
                "key": "dietary_preference",
                "value": "vegetarian",
                "confidence": 0.95,
                "reasoning": "User explicitly stated they are vegetarian",
            },
            {
                "action": "add",
                "type": "fact",
                "category": "personal",
                "key": "user_name",
                "value": "Arjun",
                "confidence": 0.95,
                "reasoning": "User stated their name directly",
            }
        ]}
        # Default pass2: accept all

        client = _make_mock_client(pass1)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: I'm Arjun and I'm vegetarian\n\nASSISTANT: Nice to meet you...",
            existing_memories=[],
            start_turn=1,
            end_turn=2,
        )

        assert len(results) == 2
        keys = {dm.key for dm in results}
        assert "dietary_preference" in keys
        assert "user_name" in keys

    def test_updates_to_existing_memories(self):
        """Updates to existing memories should work correctly."""
        existing = _make_existing_memory(key="user_location", value="Mumbai")

        pass1 = {"memories": [
            {
                "action": "update",
                "type": "fact",
                "category": "personal",
                "key": "user_location",
                "value": "Berlin",
                "confidence": 0.95,
                "reasoning": "User said they moved to Berlin",
            }
        ]}

        client = _make_mock_client(pass1)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: Actually I moved to Berlin last month\n\nASSISTANT: That's exciting!",
            existing_memories=[existing],
            start_turn=10,
            end_turn=11,
        )

        assert len(results) == 1
        assert results[0].action == "update"
        assert results[0].value == "Berlin"

    def test_empty_conversation_returns_empty(self):
        """Empty conversation should return no memories."""
        client = MagicMock()
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "",
            existing_memories=[],
            start_turn=1,
            end_turn=1,
        )

        assert results == []
        client.chat.completions.create.assert_not_called()

    def test_two_pass_flow_called(self):
        """Verify that two LLM calls are made (extract + validate)."""
        pass1 = {"memories": [
            {
                "action": "add",
                "type": "fact",
                "category": "personal",
                "key": "user_name",
                "value": "Test",
                "confidence": 0.9,
                "reasoning": "test",
            }
        ]}

        client = _make_mock_client(pass1)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        distiller.distill(
            "USER: My name is Test\n\nASSISTANT: Hello Test!",
            existing_memories=[],
            start_turn=1,
            end_turn=2,
        )

        # Should be called twice: once for extraction, once for validation
        assert client.chat.completions.create.call_count == 2

    def test_keep_actions_bypass_validation(self):
        """Keep actions should bypass pass 2 validation."""
        existing = _make_existing_memory(key="user_name", value="Arjun")

        pass1 = {"memories": [
            {
                "action": "keep",
                "type": "fact",
                "category": "personal",
                "key": "user_name",
                "value": "Arjun",
                "confidence": 0.9,
                "reasoning": "Still valid",
            }
        ]}
        # Pass 2 should not need to validate keep actions
        pass2 = {"validations": []}  # Empty - nothing to validate

        client = _make_mock_client(pass1, pass2)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: Hello again\n\nASSISTANT: Welcome back!",
            existing_memories=[existing],
            start_turn=5,
            end_turn=6,
        )

        assert len(results) == 1
        assert results[0].action == "keep"
        assert results[0].key == "user_name"

    def test_mixed_accept_reject(self):
        """Some candidates accepted, others rejected."""
        pass1 = {"memories": [
            {
                "action": "add",
                "type": "fact",
                "category": "personal",
                "key": "user_name",
                "value": "Arjun",
                "confidence": 0.95,
                "reasoning": "User stated name",
            },
            {
                "action": "add",
                "type": "fact",
                "category": "general",
                "key": "capital_france",
                "value": "Paris is the capital of France",
                "confidence": 0.5,
                "reasoning": "Mentioned in conversation",
            }
        ]}
        pass2 = {"validations": [
            {"key": "user_name", "verdict": "accept", "reason": "Personal fact"},
            {"key": "capital_france", "verdict": "reject", "reason": "World fact"},
        ]}

        client = _make_mock_client(pass1, pass2)
        distiller = MemoryDistiller(client, model="test", provider="groq")

        results = distiller.distill(
            "USER: I'm Arjun. What's the capital of France?\n\nASSISTANT: Paris!",
            existing_memories=[],
            start_turn=1,
            end_turn=2,
        )

        assert len(results) == 1
        assert results[0].key == "user_name"
