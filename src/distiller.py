"""LLM-based memory distillation using Groq API."""

from __future__ import annotations

import json
import re

from typing import Any
from groq import Groq

from .models import Memory, DistilledMemory
from .prompts import DISTILL_PROMPT


# ─── Blocklists for post-parse filtering ───
# Keys that almost always indicate world-facts, not user-facts
BLOCKED_KEY_FRAGMENTS = frozenset({
    "wifi", "gps", "http", "https", "photosynthesis", "vaccine",
    "recipe", "chess", "history", "capital", "population",
    "weather", "temperature", "definition", "explanation",
    "how_to", "tutorial", "steps", "process", "mechanism",
    "science", "physics", "biology", "chemistry", "quantum",
    "solar_system", "airplane", "engine", "bridge", "magnet",
    "octopus", "animal", "planet", "ocean", "earthquake",
    "renaissance", "ancient", "civilization", "theory",
    "machine_learning", "algorithm", "cryptocurrency",
    "electric_car", "space_exploration", "rainforest",
    "bacteria", "virus_definition", "immune", "dna",
    "fun_fact", "interesting_fact", "trivia",
    "movie_recommendation", "book_recommendation",
    "game_recommendation", "song_recommendation",
    "cooking_tip", "exercise_tip", "sleep_tip",
    "topic_discussed", "question_asked", "conversation_topic",
    "user_interest_in", "interest_in", "curiosity_about",
    "asked_about", "discussed_topic", "mentioned_topic",
})

# Values that look like definitions/explanations rather than user facts
EXPLANATION_SIGNALS = [
    "is a ", "is an ", "is the ", "are the ", "was the ", "were the ",
    "works by ", "refers to ", "is defined as", "is when ",
    "involves ", "is the process", "can be described as",
    "is used to ", "is made of ", "consists of ",
    "was invented ", "was discovered ", "was founded ",
    "is located in ",  # unless about user's location
    "is known for ", "is famous for ",
    "originated in ", "dates back to ",
    "is caused by ", "occurs when ",
    "is measured in ", "is calculated by ",
]

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
    by calling the Groq LLM with existing memories as context.
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

        prompt = DISTILL_PROMPT.format(
            existing_memories=existing_text,
            conversation=conversation_text,
            start_turn=start_turn,
            end_turn=end_turn,
        )

        if self.verbose:
            print(f"\n[DISTILLER] Turns {start_turn}-{end_turn}")
            print(f"  Existing memories: {len(existing_memories)}")
            print(f"  Conversation length: {len(conversation_text)} chars")
            print(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

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
        
        if self.verbose:
            print(f"  Response length: {len(raw)} chars")
            if hasattr(response, 'usage') and response.usage:
                print(f"  Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}")
        
        parsed = self._parse_response(raw)
        
        # Post-process: correct "keep" actions for genuinely new memories
        existing_keys = {m.key for m in existing_memories}
        for dm in parsed:
            if dm.action == "keep" and dm.key not in existing_keys:
                if self.verbose:
                    print(f"  CORRECTING | {dm.key}: keep → add (new memory)")
                dm.action = "add"

        if self.verbose:
            print(f"  Memories extracted: {len(parsed)}")
            for dm in parsed:
                print(f"    {dm.action:6} | {dm.type:12} | {dm.key}: {dm.value[:60]}")
        
        return parsed

    def _parse_response(self, raw: str) -> list[DistilledMemory]:
        """Parse LLM JSON response into DistilledMemory objects, then filter."""
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
                
                # ── Basic validation ──
                if action not in VALID_ACTIONS:
                    if self.verbose:
                        print(f"  REJECTED (bad action): {action} | {key}")
                    continue
                
                if mem_type not in VALID_TYPES:
                    if self.verbose:
                        print(f"  REJECTED (bad type): {mem_type} | {key}")
                    continue
                
                if not key or not value or key == "unknown":
                    continue
                
                try:
                    dm = DistilledMemory(
                        action=action,
                        type=mem_type,
                        category=item.get("category", "general"),
                        key=key,
                        value=value,
                        confidence=float(item.get("confidence", 0.8)),
                        reasoning=item.get("reasoning", ""),
                    )
                except TypeError:
                    # Fallback for old class definition without reasoning field
                    dm = DistilledMemory(
                        action=action,
                        type=mem_type,
                        category=item.get("category", "general"),
                        key=key,
                        value=value,
                        confidence=float(item.get("confidence", 0.8)),
                    )
                results.append(dm)
                
            except (KeyError, ValueError, TypeError) as e:
                print(f"[distiller] Skipping malformed memory item: {e}")
                continue

        # ── Apply world-fact filter ──
        filtered = self._filter_world_facts(results)
        
        return filtered

    def _filter_world_facts(self, memories: list[DistilledMemory]) -> list[DistilledMemory]:
        """
        Second layer of defense: reject memories that look like
        world-facts, definitions, topics-discussed, or assistant-generated content.
        
        "keep" and "expire" actions on existing memories are always allowed through
        since they refer to previously validated memories.
        """
        filtered = []
        
        for dm in memories:
            # Always allow keep/expire — these reference existing validated memories
            if dm.action in ("keep", "expire"):
                filtered.append(dm)
                continue
            
            key_lower = dm.key.lower()
            value_lower = dm.value.lower()
            
            # ── Check 1: Blocked key fragments ──
            if any(blocked in key_lower for blocked in BLOCKED_KEY_FRAGMENTS):
                if self.verbose:
                    print(f"  FILTERED (blocked key): {dm.key} = {dm.value[:50]}")
                continue
            
            # ── Check 2: Value too long (explanations, not facts) ──
            if len(dm.value) > 200:
                if self.verbose:
                    print(f"  FILTERED (too long): {dm.key} = {dm.value[:50]}...")
                continue
            
            # ── Check 3: Value looks like a definition/explanation ──
            is_explanation = False
            for signal in EXPLANATION_SIGNALS:
                if signal in value_lower:
                    # Exception: value starts with "user" or a likely user-name pattern
                    # e.g., "User is a software engineer" should pass
                    # e.g., "Arjun is a developer" should pass
                    first_word = value_lower.split()[0] if value_lower.split() else ""
                    if first_word in ("user", "i", "they", "he", "she"):
                        break  # allow it
                    # Check if it's about the user (key suggests it)
                    user_key_signals = (
                        "user_", "my_", "preferred_", "favorite_",
                        "dietary_", "allergy", "name", "age", "location",
                        "email", "phone", "address", "occupation", "job",
                        "daughter", "son", "wife", "husband", "partner",
                        "pet_", "dog_", "cat_",
                    )
                    if any(key_lower.startswith(uks) for uks in user_key_signals):
                        break  # allow it
                    is_explanation = True
                    break
            
            if is_explanation:
                if self.verbose:
                    print(f"  FILTERED (explanation): {dm.key} = {dm.value[:50]}")
                continue
            
            # ── Check 4: Key suggests "topic discussed" rather than user fact ──
            topic_patterns = [
                r"^topic_",
                r"^discussed_",
                r"^asked_about_",
                r"^mentioned_",
                r"^conversation_about_",
                r"^info_about_",
                r"^knowledge_of_",
                r"^learned_about_",
                r"_topic$",
                r"_question$",
                r"_discussed$",
            ]
            if any(re.search(pat, key_lower) for pat in topic_patterns):
                if self.verbose:
                    print(f"  FILTERED (topic key): {dm.key} = {dm.value[:50]}")
                continue
            
            # ── Check 5: Value is a generic statement not about the user ──
            generic_value_starts = [
                "the user asked about ",
                "the user was curious about ",
                "the user wanted to know ",
                "the assistant explained ",
                "the assistant provided ",
                "the assistant suggested ",
                "they discussed ",
                "the conversation covered ",
            ]
            if any(value_lower.startswith(gvs) for gvs in generic_value_starts):
                if self.verbose:
                    print(f"  FILTERED (generic value): {dm.key} = {dm.value[:50]}")
                continue
            
            # ── Passed all checks ──
            filtered.append(dm)
        
        if self.verbose and len(memories) != len(filtered):
            print(f"  FILTER: {len(memories)} → {len(filtered)} memories ({len(memories) - len(filtered)} removed)")
        
        return filtered

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
                print(f"[distiller] Warning: recovered truncated JSON via bracket-closing")
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
                print(f"[distiller] Warning: recovered {len(recovered_memories)} memories via regex extraction")
                return {"memories": recovered_memories}
        
        print(f"[distiller] Failed to parse JSON: {text[:200]}...")
        return None