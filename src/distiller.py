"""LLM-based memory distillation using Groq API."""

from __future__ import annotations

import json
import re

from typing import Any
from groq import Groq

from .models import Memory, DistilledMemory
from .prompts import DISTILL_PROMPT


class MemoryDistiller:
    """
    Extracts structured memories from a conversation segment
    by calling the Groq LLM with existing memories as context.
    """

    def __init__(self, client: Any, model: str = "llama-3.3-70b-versatile"):
        self.client = client
        self.model = model

    def distill(
        self,
        conversation_text: str,
        existing_memories: list[Memory],
        start_turn: int,
        end_turn: int,
    ) -> list[DistilledMemory]:
        """
        Analyze a conversation segment and return memory operations.
        
        Args:
            conversation_text: Plain text of the conversation segment
            existing_memories: All currently active memories
            start_turn: First turn number in this segment
            end_turn: Last turn number in this segment
            
        Returns:
            List of DistilledMemory objects with actions (add/update/keep/expire)
        """
        if not conversation_text.strip():
            return []

        # Format existing memories
        if existing_memories:
            mem_lines = []
            for m in existing_memories:
                mem_lines.append(
                    f"- [{m.type}] {m.key}: {m.value} "
                    f"(confidence: {m.confidence:.2f}, from turn {m.source_turn})"
                )
            existing_text = "\\n".join(mem_lines)
        else:
            existing_text = "(none yet — this is the start of the conversation)"

        prompt = DISTILL_PROMPT.format(
            existing_memories=existing_text,
            conversation=conversation_text,
            start_turn=start_turn,
            end_turn=end_turn,
        )

        # Build kwargs — skip response_format for providers that don't support it
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        
        # Only add response_format for Groq (has native JSON mode)
        # Gemini and Ollama don't reliably support it via OpenAI compat
        if hasattr(self.client, '_client'):  # Groq client
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        
        raw = response.choices[0].message.content
        parsed = self._parse_response(raw)
        return parsed

    def _parse_response(self, raw: str) -> list[DistilledMemory]:
        """Parse LLM JSON response into DistilledMemory objects."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        
        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to recover from truncated JSON
            data = self._recover_truncated_json(cleaned)
        
        if data is None:
            return []

        memories = data.get("memories", [])
        results = []

        for item in memories:
            try:
                val = item.get("value", "")
                if isinstance(val, (list, dict)):
                    val = json.dumps(val)
                
                dm = DistilledMemory(
                    action=item.get("action", "add"),
                    type=item.get("type", "fact"),
                    category=item.get("category", "general"),
                    key=item.get("key", "unknown"),
                    value=str(val),
                    confidence=float(item.get("confidence", 0.8)),
                )
                # Validate
                if dm.key and dm.value and dm.action in ("add", "update", "keep", "expire"):
                    results.append(dm)
            except (KeyError, ValueError, TypeError) as e:
                print(f"[distiller] Skipping malformed memory item: {e}")
                continue

        return results

    def _recover_truncated_json(self, text: str) -> dict | None:
        """Attempt to recover valid JSON from a truncated LLM response."""
        # Strategy 1: Try closing open brackets/braces
        # Strip trailing incomplete tokens (partial strings, trailing commas, etc.)
        attempt = text.rstrip()
        # Remove trailing partial content after last complete value
        # e.g., remove trailing comma, partial key, etc.
        attempt = re.sub(r',\s*"[^"]*$', '', attempt)  # trailing partial key
        attempt = re.sub(r',\s*$', '', attempt)  # trailing comma
        attempt = re.sub(r'\.{2,}$', '', attempt)  # trailing dots
        
        # Count unclosed brackets
        open_braces = attempt.count('{') - attempt.count('}')
        open_brackets = attempt.count('[') - attempt.count(']')
        
        # Try closing with the right number of brackets
        suffix = ']' * max(0, open_brackets) + '}' * max(0, open_braces)
        
        # Try multiple closing strategies
        strategies = [
            attempt + suffix,
            attempt + '}' + suffix,
            attempt + '"' + suffix,
            attempt + '"}' + suffix,
            attempt + '"}'  + ']' * max(0, open_brackets) + '}' * max(0, open_braces - 1),
        ]
        
        for s in strategies:
            try:
                data = json.loads(s)
                print(f"[distiller] Warning: recovered truncated JSON via bracket-closing")
                return data
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Extract individual memory objects via regex
        pattern = r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*"key"\s*:\s*"[^"]+"[^{}]*"value"\s*:\s*"[^"]+"[^{}]*\}'
        matches = re.findall(pattern, text)
        if matches:
            recovered_memories = []
            for m in matches:
                try:
                    obj = json.loads(m)
                    recovered_memories.append(obj)
                except json.JSONDecodeError:
                    continue
            if recovered_memories:
                print(f"[distiller] Warning: recovered {len(recovered_memories)} memories via regex extraction")
                return {"memories": recovered_memories}
        
        print(f"[distiller] Failed to parse JSON: {text[:200]}...")
        return None
