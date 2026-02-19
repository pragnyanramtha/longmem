"""Main agent: conversation loop with context-aware memory distillation."""

from __future__ import annotations

import time
from typing import Any
from groq import Groq
import openai

from .models import Memory, DistilledMemory
from .store import MemoryStore
from .context import ContextManager
from .distiller import MemoryDistiller
from .retriever import MemoryRetriever
from .consolidator import MemoryConsolidator
from .prompts import SYSTEM_PROMPT_TEMPLATE, PROFILE_SECTION, MEMORIES_SECTION


class LongMemAgent:
    """
    A conversational agent with long-form memory.
    
    Lifecycle per turn:
    1. Check if context needs flushing (70% full)
    2. If yes: distill memories from current segment, reset context
    3. Retrieve relevant memories for the incoming query
    4. Build system prompt with profile + retrieved memories
    5. Run LLM inference via Groq
    6. Return response with metadata
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        db_path: str = "memory.db",
        context_limit: int = 8192,
        flush_threshold: float = 0.70,
        verbose: bool = False,
    ):
        # Initialize LLM client
        self.provider = provider
        if provider == "groq" and not base_url:
            self.client = Groq(api_key=api_key)
        else:
            # Normalize defaults for Ollama
            if provider == "ollama" and not base_url:
                base_url = "http://localhost:11434/v1"
                if not api_key:
                    api_key = "ollama"
            
            # Normalize defaults for Gemini (OpenAI-compatible endpoint)
            if provider == "gemini" and not base_url:
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                if not api_key:
                    import os
                    api_key = os.environ.get("GEMINI_API_KEY", "")
            
            # If using a custom base_url (like local server) without a key, use a dummy key
            if base_url and not api_key:
                api_key = "dummy"
                
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.model = model
        self.verbose = verbose

        # Components
        self.store = MemoryStore(db_path)
        self.distiller = MemoryDistiller(self.client, model=model, provider=provider, verbose=verbose)
        self.retriever = MemoryRetriever(self.store)
        self.consolidator = MemoryConsolidator(self.store, client=self.client, model=model, provider=provider)
        self.ctx = ContextManager(
            model_context_limit=context_limit,
            flush_threshold=flush_threshold,
            keep_last_turns=4,
        )

        # State - load from database to continue previous conversations
        self.turn_id: int = self.store.get_last_turn_id()
        self.segment_start_turn: int = max(1, self.turn_id + 1)
        self.total_flushes: int = 0

        # Initialize system prompt
        self._rebuild_system_prompt()

    def chat(self, user_message: str) -> dict:
        """
        Process a user message and return the assistant response with metadata.
        
        Returns dict with keys:
            response: str
            turn_id: int
            context_utilization: str
            active_memories: list[dict]
            flush_triggered: bool
            total_memories: int
        """
        self.turn_id += 1
        flush_triggered = False
        retrieval_ms = 0.0
        total_start = time.time()

        # ── STEP 1: Check if context needs flushing ──
        incoming_estimate = self.ctx.count_tokens(user_message) + 300  # response estimate
        if self.ctx.needs_flush(incoming_estimate) and self.ctx.message_count() > 0:
            self._flush()
            flush_triggered = True

        # ── STEP 2: Retrieve relevant memories ──
        retrieval_start = time.time()
        results = self.retriever.retrieve(user_message, top_k=5, current_turn=self.turn_id)
        retrieved_memories = [r.memory for r in results]
        retrieval_ms = (time.time() - retrieval_start) * 1000
        
        # Update last_used_turn for retrieved memories
        for memory in retrieved_memories:
            self.store.touch_memory(memory.id, self.turn_id)

        # ── STEP 3: Rebuild system prompt with retrieved memories ──
        self._rebuild_system_prompt(query_memories=retrieved_memories)

        # ── STEP 4: Add user message to context ──
        self.ctx.add_message("user", user_message)

        # ── STEP 5: LLM inference ──
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.ctx.get_messages_for_api(provider=self.provider),
            temperature=0.7,
            max_tokens=1024,
        )
        assistant_msg = response.choices[0].message.content

        # ── STEP 6: Add assistant response to context ──
        self.ctx.add_message("assistant", assistant_msg)

        # ── STEP 7: Log turn ──
        self.store.log_turn(
            self.turn_id, "user", user_message,
            memories_retrieved=[m.id for m in retrieved_memories],
        )

        total_ms = (time.time() - total_start) * 1000

        # ── STEP 8: Return with metadata ──
        return {
            "response": assistant_msg,
            "turn_id": self.turn_id,
            "context_utilization": f"{self.ctx.utilization():.0%}",
            "context_tokens": self.ctx.total_tokens(),
            "retrieval_ms": round(retrieval_ms, 1),
            "total_ms": round(total_ms, 1),
            "flush_triggered": flush_triggered,
            "total_flushes": self.total_flushes,
            "active_memories": [
                {
                    "memory_id": m.id,
                    "content": f"{m.key}: {m.value}",
                    "origin_turn": m.source_turn,
                    "last_used_turn": self.turn_id,
                    "type": m.type,
                    "confidence": m.confidence,
                }
                for m in retrieved_memories
            ],
            "total_memories": self.store.active_count(),
        }

    def manual_distill(self) -> dict:
        """
        Manually trigger memory distillation without waiting for context threshold.
        Returns dict with distillation results.
        """
        if self.ctx.message_count() == 0:
            return {
                "success": False,
                "message": "No messages to distill yet.",
                "memories_added": 0,
            }
        
        initial_count = self.store.active_count()
        self._flush()
        final_count = self.store.active_count()
        
        return {
            "success": True,
            "message": f"Distillation complete. {final_count - initial_count} memories extracted.",
            "memories_added": final_count - initial_count,
            "total_memories": final_count,
            "snapshot_saved": f"snapshots/turn_{self.turn_id:05d}.md",
        }

    def _flush(self):
        """Distill memories from current conversation segment and reset context."""
        conversation_text = self.ctx.get_conversation_text()
        existing_memories = self.store.get_active_memories()

        # Run distillation
        distilled = self.distiller.distill(
            conversation_text=conversation_text,
            existing_memories=existing_memories,
            start_turn=self.segment_start_turn,
            end_turn=self.turn_id,
        )

        # Apply memory operations
        self._apply_distilled(distilled)

        # Write snapshot for debugging
        self.store.write_snapshot(self.turn_id)

        # Reset context
        self._rebuild_system_prompt()
        self.ctx.reset(self.ctx.system_prompt)
        self.segment_start_turn = self.turn_id
        self.total_flushes += 1

        # Run consolidation periodically (every 5 flushes)
        if self.total_flushes % 5 == 0:
            report = self.consolidator.run_consolidation(self.turn_id)
            if self.verbose:
                print(f"  [CONSOLIDATION] merged={report.duplicates_merged}, "
                      f"decayed={report.memories_decayed}, expired={report.memories_expired}")

    def _apply_distilled(self, distilled: list[DistilledMemory]):
        """Apply distilled memory operations to the store."""
        if self.verbose:
            print(f"\n[AGENT] Applying {len(distilled)} memory operations")
        
        for dm in distilled:
            if dm.action == "add":
                # Dedup: if key already exists with same value, skip
                existing = self.store.find_by_key(dm.key)
                if existing:
                    if existing.value.strip().lower() == dm.value.strip().lower():
                        if self.verbose:
                            print(f"  SKIP (duplicate) | {dm.key}")
                        continue  # exact duplicate — skip
                    else:
                        # Key exists but value changed — treat as update
                        if self.verbose:
                            print(f"  AUTO-UPDATE | {dm.key}: {existing.value[:40]}... → {dm.value[:40]}...")
                        self.store.deactivate_by_key(dm.key)
                else:
                    if self.verbose:
                        print(f"  ADD | {dm.type:12} | {dm.key}: {dm.value[:50]}...")
                self.store.add_memory(dm, self.turn_id)
            
            elif dm.action == "update":
                if self.verbose:
                    print(f"  UPDATE | {dm.key}")
                self.store.deactivate_by_key(dm.key)
                self.store.add_memory(dm, self.turn_id)
            
            elif dm.action == "expire":
                if self.verbose:
                    print(f"  EXPIRE | {dm.key}")
                self.store.deactivate_by_key(dm.key)
            
            # "keep" → no-op
            elif dm.action == "keep" and self.verbose:
                print(f"  KEEP | {dm.key}")

    def _rebuild_system_prompt(self, query_memories: list[Memory] | None = None):
        """Construct system prompt from profile + optional retrieved memories."""
        profile = self.store.get_profile()

        # Profile section
        if profile:
            profile_yaml = "\n".join(f"- {k}: {v}" for k, v in profile.items())
            profile_section = PROFILE_SECTION.format(profile_yaml=profile_yaml)
        else:
            profile_section = ""

        # Memories section
        if query_memories:
            profile_keys = set(profile.keys()) if profile else set()
            mem_lines = []
            for m in query_memories:
                # Deduplicate: don't show here if already in profile section
                if m.key not in profile_keys:
                    mem_lines.append(f"- [{m.type}] {m.key}: {m.value}")
            
            if mem_lines:
                memories_section = MEMORIES_SECTION.format(
                    memories_list="\n".join(mem_lines)
                )
            else:
                memories_section = ""
        else:
            memories_section = ""

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            profile_section=profile_section,
            memories_section=memories_section,
        )

        self.ctx.set_system_prompt(prompt)

    def get_all_memories(self) -> list[dict]:
        """Return all active memories as dicts (for debugging/eval)."""
        return [
            {
                "id": m.id,
                "type": m.type,
                "category": m.category,
                "key": m.key,
                "value": m.value,
                "source_turn": m.source_turn,
                "confidence": m.confidence,
            }
            for m in self.store.get_active_memories()
        ]
