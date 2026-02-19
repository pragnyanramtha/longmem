"""Two-pass LLM-based memory distillation.

Pass 1 (extraction): Liberal extraction of candidate memories.
Pass 2 (validation): Strict validation to reject world-facts, transient details,
and conversation meta-noise.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import Memory, DistilledMemory
from .prompts import EXTRACTION_PROMPT, VALIDATION_PROMPT

logger = logging.getLogger("atlas.distiller")

# Valid memory types (anything else gets rejected)
VALID_TYPES = frozenset({
    "preference", "fact", "commitment",
    "constraint", "entity", "instruction",
})

# Valid actions
VALID_ACTIONS = frozenset({"add", "update", "keep", "expire"})


class MemoryDistiller:
    """
    Extracts structured memories from a conversation segment
    using a two-pass LLM pipeline:

    1. Extract candidates (liberal)
    2. Validate candidates (strict)
    """

    def __init__(self, client: Any, model: str = "llama-3.3-70b-versatile", provider: str = "groq", verbose: bool = False):
        self.client = client
        self.model = model
        self.provider = provider
        self.verbose = verbose

    def distill(
        self,
        conversation_text: str,
        existing_memories: list[Memory],
        start_turn: int,
        end_turn: int,
    ) -> list[DistilledMemory]:
        """
        Analyze a conversation segment and return memory operations.

        Two-pass pipeline:
        1. _extract_candidates() -- liberal extraction
        2. _validate_candidates() -- strict validation
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
            existing_text = "\n".join(mem_lines)
        else:
            existing_text = "(none yet — this is the start of the conversation)"

        logger.info(
            "Distilling turns %d-%d (%d chars, %d existing memories)",
            start_turn, end_turn, len(conversation_text), len(existing_memories),
        )

        # Pass 1: Extract candidates
        candidates = self._extract_candidates(
            conversation_text, existing_text, start_turn, end_turn
        )

        if not candidates:
            logger.info("Pass 1 returned 0 candidates")
            return []

        logger.info("Pass 1 extracted %d candidates", len(candidates))

        # Pass 2: Validate candidates
        validated = self._validate_candidates(candidates, conversation_text)

        logger.info(
            "Pass 2: %d/%d candidates accepted",
            len(validated), len(candidates),
        )

        # Post-process: correct "keep" actions for genuinely new memories
        existing_keys = {m.key for m in existing_memories}
        for dm in validated:
            if dm.action == "keep" and dm.key not in existing_keys:
                logger.debug("Correcting %s: keep -> add (new memory)", dm.key)
                dm.action = "add"

        return validated

    def _extract_candidates(
        self,
        conversation_text: str,
        existing_text: str,
        start_turn: int,
        end_turn: int,
    ) -> list[DistilledMemory]:
        """Pass 1: Liberal extraction of candidate memories."""
        prompt = EXTRACTION_PROMPT.format(
            existing_memories=existing_text,
            conversation=conversation_text,
            start_turn=start_turn,
            end_turn=end_turn,
        )

        if self.verbose:
            print(f"\n[DISTILLER] Pass 1: Extracting from turns {start_turn}-{end_turn}")

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000,
        }

        if self.provider in ("groq", "gemini"):
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content

        if self.verbose and hasattr(response, 'usage') and response.usage:
            print(f"  Pass 1 tokens: prompt={response.usage.prompt_tokens}, "
                  f"completion={response.usage.completion_tokens}")

        return self._parse_response(raw)

    def _validate_candidates(
        self,
        candidates: list[DistilledMemory],
        conversation_text: str,
    ) -> list[DistilledMemory]:
        """Pass 2: Strict validation of candidate memories."""
        # Keep/expire actions on existing memories bypass validation
        keep_expire = [dm for dm in candidates if dm.action in ("keep", "expire")]
        to_validate = [dm for dm in candidates if dm.action not in ("keep", "expire")]

        if not to_validate:
            logger.debug("No candidates to validate (all keep/expire)")
            return keep_expire

        # Format candidates as JSON for the validation prompt
        candidates_json = json.dumps([
            {
                "action": dm.action,
                "type": dm.type,
                "key": dm.key,
                "value": dm.value,
                "confidence": dm.confidence,
                "reasoning": dm.reasoning,
            }
            for dm in to_validate
        ], indent=2)

        prompt = VALIDATION_PROMPT.format(
            candidates_json=candidates_json,
            conversation=conversation_text,
        )

        if self.verbose:
            print(f"  Pass 2: Validating {len(to_validate)} candidates")

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        if self.provider in ("groq", "gemini"):
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content

        if self.verbose and hasattr(response, 'usage') and response.usage:
            print(f"  Pass 2 tokens: prompt={response.usage.prompt_tokens}, "
                  f"completion={response.usage.completion_tokens}")

        # Parse validation results
        verdicts = self._parse_validation(raw)

        # Build accepted set
        accepted_keys = set()
        for v in verdicts:
            key = v.get("key", "")
            verdict = v.get("verdict", "reject")
            reason = v.get("reason", "")

            if verdict == "accept":
                accepted_keys.add(key)
                logger.debug("ACCEPTED: %s (%s)", key, reason)
            else:
                logger.debug("REJECTED: %s (%s)", key, reason)
                if self.verbose:
                    print(f"  REJECTED: {key} -- {reason}")

        # Filter candidates
        validated = [dm for dm in to_validate if dm.key in accepted_keys]

        return keep_expire + validated

    def _parse_validation(self, raw: str) -> list[dict]:
        """Parse the validation response JSON."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = self._recover_truncated_json(cleaned)

        if data is None:
            logger.warning("Failed to parse validation response")
            return []

        return data.get("validations", [])

    def _parse_response(self, raw: str) -> list[DistilledMemory]:
        """Parse LLM JSON response into DistilledMemory objects."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
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

                action = item.get("action", "add")
                mem_type = item.get("type", "fact")
                key = item.get("key", "unknown")
                value = str(val)

                # Basic structural validation
                if action not in VALID_ACTIONS:
                    logger.debug("Rejected (bad action): %s | %s", action, key)
                    continue

                if mem_type not in VALID_TYPES:
                    logger.debug("Rejected (bad type): %s | %s", mem_type, key)
                    continue

                if not key or not value or key == "unknown":
                    continue

                dm = DistilledMemory(
                    action=action,
                    type=mem_type,
                    category=item.get("category", "general"),
                    key=key,
                    value=value,
                    confidence=float(item.get("confidence", 0.8)),
                    reasoning=item.get("reasoning", ""),
                )
                results.append(dm)

            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Skipping malformed memory item: %s", e)
                continue

        return results

    def _recover_truncated_json(self, text: str) -> dict | None:
        """Attempt to recover valid JSON from a truncated LLM response."""
        attempt = text.rstrip()
        attempt = re.sub(r',\s*"[^"]*$', '', attempt)
        attempt = re.sub(r',\s*$', '', attempt)
        attempt = re.sub(r'\.{2,}$', '', attempt)

        open_braces = attempt.count('{') - attempt.count('}')
        open_brackets = attempt.count('[') - attempt.count(']')

        suffix = ']' * max(0, open_brackets) + '}' * max(0, open_braces)

        strategies = [
            attempt + suffix,
            attempt + '}' + suffix,
            attempt + '"' + suffix,
            attempt + '"}' + suffix,
            attempt + '"}' + ']' * max(0, open_brackets) + '}' * max(0, open_braces - 1),
        ]

        for s in strategies:
            try:
                data = json.loads(s)
                logger.warning("Recovered truncated JSON via bracket-closing")
                return data
            except json.JSONDecodeError:
                continue

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
                logger.warning("Recovered %d memories via regex extraction", len(recovered_memories))
                return {"memories": recovered_memories}

        logger.error("Failed to parse JSON: %s...", text[:200])
        return None
