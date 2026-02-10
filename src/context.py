"""Token-aware context window manager."""

from __future__ import annotations

import tiktoken


class ContextManager:
    """
    Manages the conversation context window with token counting.
    
    Triggers a memory distillation flush when token usage hits the 
    configured threshold (default 70%), then resets the context 
    keeping only the last few turns for continuity.
    """

    def __init__(
        self,
        model_context_limit: int = 8192,
        flush_threshold: float = 0.70,
        keep_last_turns: int = 4,  # number of messages to keep (2 exchanges)
    ):
        self.model_context_limit = model_context_limit
        self.flush_threshold = flush_threshold
        self.keep_last_turns = keep_last_turns
        
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        self.system_prompt: str = ""
        self.messages: list[dict[str, str]] = []
        self._system_tokens: int = 0
        self._message_tokens: int = 0

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.encoder.encode(text))

    def total_tokens(self) -> int:
        """Total tokens currently in context (system + messages)."""
        return self._system_tokens + self._message_tokens

    def utilization(self) -> float:
        """Fraction of context window used (0.0 to 1.0+)."""
        if self.model_context_limit == 0:
            return 0.0
        return self.total_tokens() / self.model_context_limit

    def tokens_remaining(self) -> int:
        """How many tokens left before hitting the limit."""
        return self.model_context_limit - self.total_tokens()

    def needs_flush(self, incoming_tokens: int = 0) -> bool:
        """
        Check if adding incoming_tokens would push past the flush threshold.
        Call this BEFORE adding the new user message.
        """
        projected = self.total_tokens() + incoming_tokens
        return projected >= (self.model_context_limit * self.flush_threshold)

    def set_system_prompt(self, prompt: str):
        """Set or replace the system prompt. Recalculates token count."""
        self.system_prompt = prompt
        self._system_tokens = self.count_tokens(prompt) + 4  # role overhead

    def add_message(self, role: str, content: str):
        """Append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        self._message_tokens += self.count_tokens(content) + 4  # role overhead

    def get_messages_for_api(self) -> list[dict[str, str]]:
        """
        Return the full message list for the LLM API call.
        Format: [system, ...messages]
        """
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(self.messages)
        return result

    def get_conversation_text(self) -> str:
        """
        Render all messages as plain text for the distiller.
        Format: 'USER: ...\nASSISTANT: ...\n'
        """
        lines = []
        for msg in self.messages:
            lines.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n\n".join(lines)

    def message_count(self) -> int:
        """Number of messages in current context."""
        return len(self.messages)

    def reset(self, new_system_prompt: str):
        """
        Reset context: keep only the last N messages for continuity,
        replace system prompt, recalculate all token counts.
        """
        carryover = self.messages[-self.keep_last_turns:] if self.messages else []
        self.messages = carryover
        self.system_prompt = new_system_prompt
        
        # Recalculate tokens from scratch
        self._system_tokens = self.count_tokens(new_system_prompt) + 4
        self._message_tokens = sum(
            self.count_tokens(m["content"]) + 4 for m in self.messages
        )
