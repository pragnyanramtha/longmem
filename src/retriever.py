"""Hybrid memory retrieval: vector similarity + BM25 keyword search with RRF merging."""

from __future__ import annotations

from .models import Memory, RetrievalResult
from .store import MemoryStore


class MemoryRetriever:
    """
    Retrieves relevant memories for a given query using hybrid search.
    
    Combines:
    1. Vector similarity search (semantic meaning)
    2. FTS5 keyword search (exact term matching)
    3. Reciprocal Rank Fusion to merge both ranked lists
    """

    RRF_K = 60  # RRF constant — standard value

    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Find the top_k most relevant active memories for the query.
        
        Always returns at most top_k results, sorted by relevance.
        Returns empty list if no memories exist.
        """
        if self.store.active_count() == 0:
            return []

        # 1. Vector search
        vec_results = self.store.search_vector(query, top_k=top_k * 3)
        
        # 2. FTS keyword search  
        fts_results = self.store.search_fts(query, top_k=top_k * 3)

        # 3. RRF merge
        scores: dict[str, float] = {}
        
        for rank, (mem_id, distance) in enumerate(vec_results):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        
        for rank, (rowid, fts_rank) in enumerate(fts_results):
            mem_id = self.store.rowid_to_memory_id(rowid)
            if mem_id:
                scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        # 4. Sort by combined score, fetch full memory objects
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for mem_id, score in ranked:
            memory = self.store.get_memory_by_id(mem_id)
            if memory and memory.is_active:
                results.append(RetrievalResult(memory=memory, score=score))
            if len(results) >= top_k:
                break

        return results
