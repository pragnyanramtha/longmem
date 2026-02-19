"""Tests for the ContextManager module."""

import sys

sys.path.insert(0, "/home/pik/dev/longmem")

import pytest

from src.context import ContextManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return ContextManager(model_context_limit=8192, flush_threshold=0.70)


@pytest.fixture
def small_ctx():
    """A context manager with a very small limit for testing thresholds."""
    return ContextManager(model_context_limit=100, flush_threshold=0.70)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_returns_positive_int(self, ctx):
        count = ctx.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_empty_string(self, ctx):
        count = ctx.count_tokens("")
        assert count == 0

    def test_longer_text_more_tokens(self, ctx):
        short = ctx.count_tokens("hi")
        long = ctx.count_tokens("This is a much longer sentence with many more words.")
        assert long > short


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSetSystemPrompt:
    def test_set_system_prompt(self, ctx):
        ctx.set_system_prompt("You are a helpful assistant.")
        assert ctx.system_prompt == "You are a helpful assistant."
        assert ctx.total_tokens() > 0

    def test_replace_system_prompt(self, ctx):
        ctx.set_system_prompt("Short prompt.")
        tokens_1 = ctx.total_tokens()

        ctx.set_system_prompt("A much longer system prompt with more details about behavior.")
        tokens_2 = ctx.total_tokens()

        assert tokens_2 > tokens_1


# ---------------------------------------------------------------------------
# Add message
# ---------------------------------------------------------------------------

class TestAddMessage:
    def test_add_message(self, ctx):
        ctx.add_message("user", "Hello!")
        assert ctx.message_count() == 1

    def test_add_multiple_messages(self, ctx):
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "Hi there!")
        ctx.add_message("user", "How are you?")
        assert ctx.message_count() == 3

    def test_add_message_increases_tokens(self, ctx):
        tokens_before = ctx.total_tokens()
        ctx.add_message("user", "This is a test message.")
        tokens_after = ctx.total_tokens()
        assert tokens_after > tokens_before


# ---------------------------------------------------------------------------
# Total tokens
# ---------------------------------------------------------------------------

class TestTotalTokens:
    def test_system_plus_messages(self, ctx):
        ctx.set_system_prompt("System prompt here.")
        sys_tokens = ctx.total_tokens()

        ctx.add_message("user", "Hello!")
        total = ctx.total_tokens()

        assert total > sys_tokens

    def test_empty_context(self, ctx):
        assert ctx.total_tokens() == 0


# ---------------------------------------------------------------------------
# Utilization
# ---------------------------------------------------------------------------

class TestUtilization:
    def test_empty_utilization(self, ctx):
        assert ctx.utilization() == 0.0

    def test_utilization_increases(self, ctx):
        ctx.set_system_prompt("A system prompt.")
        util_1 = ctx.utilization()

        ctx.add_message("user", "A user message that adds tokens.")
        util_2 = ctx.utilization()

        assert util_2 > util_1
        assert 0.0 < util_2 < 1.0

    def test_utilization_is_fraction(self, small_ctx):
        small_ctx.set_system_prompt("Hello")
        util = small_ctx.utilization()
        assert 0.0 < util <= 1.0


# ---------------------------------------------------------------------------
# Needs flush
# ---------------------------------------------------------------------------

class TestNeedsFlush:
    def test_no_flush_when_empty(self, ctx):
        assert ctx.needs_flush(incoming_tokens=10) is False

    def test_flush_when_threshold_exceeded(self, small_ctx):
        """With limit=100 and threshold=0.70, flush at 70 tokens."""
        small_ctx.set_system_prompt("A" * 200)  # way over limit
        assert small_ctx.needs_flush(incoming_tokens=10) is True

    def test_no_flush_below_threshold(self, ctx):
        ctx.set_system_prompt("Short.")
        assert ctx.needs_flush(incoming_tokens=10) is False


# ---------------------------------------------------------------------------
# Tokens remaining
# ---------------------------------------------------------------------------

class TestTokensRemaining:
    def test_full_remaining_when_empty(self, ctx):
        assert ctx.tokens_remaining() == 8192

    def test_remaining_decreases(self, ctx):
        remaining_before = ctx.tokens_remaining()
        ctx.add_message("user", "Some text here.")
        remaining_after = ctx.tokens_remaining()
        assert remaining_after < remaining_before


# ---------------------------------------------------------------------------
# get_messages_for_api
# ---------------------------------------------------------------------------

class TestGetMessagesForAPI:
    def test_standard_format(self, ctx):
        ctx.set_system_prompt("You are helpful.")
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "Hi!")

        messages = ctx.get_messages_for_api()
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_gemini_format(self, ctx):
        ctx.set_system_prompt("You are helpful.")
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "Hi!")

        messages = ctx.get_messages_for_api(provider="gemini")
        # Gemini merges system into first user message
        assert messages[0]["role"] == "user"
        assert "You are helpful." in messages[0]["content"]
        assert "Hello!" in messages[0]["content"]
        assert len(messages) == 2  # merged user + assistant

    def test_no_system_prompt(self, ctx):
        ctx.add_message("user", "Hello!")
        messages = ctx.get_messages_for_api()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# get_conversation_text
# ---------------------------------------------------------------------------

class TestGetConversationText:
    def test_formats_correctly(self, ctx):
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "Hi there!")

        text = ctx.get_conversation_text()
        assert "USER: Hello!" in text
        assert "ASSISTANT: Hi there!" in text

    def test_truncates_long_assistant(self, ctx):
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "A" * 1000)

        text = ctx.get_conversation_text()
        assert "[truncated]" in text

    def test_empty_context(self, ctx):
        text = ctx.get_conversation_text()
        assert text == ""


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_keeps_last_n(self, ctx):
        ctx.set_system_prompt("System prompt.")
        ctx.add_message("user", "msg 1")
        ctx.add_message("assistant", "reply 1")
        ctx.add_message("user", "msg 2")
        ctx.add_message("assistant", "reply 2")
        ctx.add_message("user", "msg 3")
        ctx.add_message("assistant", "reply 3")

        assert ctx.message_count() == 6

        ctx.reset("New system prompt.")
        # keep_last_turns=4 by default
        assert ctx.message_count() == 4
        assert ctx.system_prompt == "New system prompt."

    def test_reset_recalculates_tokens(self, ctx):
        ctx.set_system_prompt("Old prompt.")
        ctx.add_message("user", "A" * 500)
        ctx.add_message("assistant", "B" * 500)

        tokens_before = ctx.total_tokens()
        ctx.reset("Short.")
        tokens_after = ctx.total_tokens()

        # Should be less since system prompt is shorter and some messages dropped
        assert tokens_after <= tokens_before

    def test_reset_empty_context(self, ctx):
        ctx.reset("New prompt.")
        assert ctx.message_count() == 0
        assert ctx.system_prompt == "New prompt."


# ---------------------------------------------------------------------------
# Message count
# ---------------------------------------------------------------------------

class TestMessageCount:
    def test_starts_at_zero(self, ctx):
        assert ctx.message_count() == 0

    def test_increments(self, ctx):
        ctx.add_message("user", "Hello!")
        assert ctx.message_count() == 1
        ctx.add_message("assistant", "Hi!")
        assert ctx.message_count() == 2
