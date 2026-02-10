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

        # No try/except here - let it fail so the agent can retry or handle it
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        
        raw = response.choices[0].message.content
        parsed = self._parse_response(raw)
        return parsed

    def _parse_response(self, raw: str) -> list[DistilledMemory]:
        """Parse LLM JSON response into DistilledMemory objects."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[distiller] Failed to parse JSON: {cleaned[:200]}...")
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
                    reasoning=item.get("reasoning", ""),
                )
                # Validate
                if dm.key and dm.value and dm.action in ("add", "update", "keep", "expire"):
                    results.append(dm)
            except (KeyError, ValueError, TypeError) as e:
                print(f"[distiller] Skipping malformed memory item: {e}")
                continue

        return results
