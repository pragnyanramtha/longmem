"""Hybrid memory retrieval: vector similarity + BM25 keyword search with RRF merging
and relevance-scored filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import Memory, RetrievalResult
from .store import MemoryStore

logger = logging.getLogger("atlas.retriever")


@dataclass
class _ScoredCandidate:
    """Internal bookkeeping for a candidate memory during scoring."""
    memory_id: str
    rrf_score: float
    vector_distance: float | None  # L2 distance from vector search, None if not in vec results
    memory: Memory | None = None   # populated after fetch


class MemoryRetriever:
    """
    Retrieves relevant memories for a given query using hybrid search
    with relevance scoring and dynamic filtering.

    Pipeline:
    1. Vector similarity search (semantic meaning)
    2. FTS5 keyword search (exact term matching)
    3. Reciprocal Rank Fusion to merge both ranked lists
    4. Relevance scoring: weighted combination of RRF, semantic similarity,
       recency boost, and confidence
    5. Minimum threshold filtering: only return memories above a score floor
    6. Dynamic top_k: return up to top_k, but fewer if not enough qualify
    """

    RRF_K = 60  # RRF constant -- standard value

    # Default weights for the combined score
    WEIGHT_RRF = 0.40
    WEIGHT_SEMANTIC = 0.30
    WEIGHT_RECENCY = 0.15
    WEIGHT_CONFIDENCE = 0.15

    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
        current_turn: int | None = None,
        recency_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Find the most relevant active memories for *query*.

        Parameters
        ----------
        query : str
            Natural-language query to match against stored memories.
        top_k : int
            Maximum number of results to return.
        min_score : float
            Minimum combined relevance score a memory must reach to be included.
        current_turn : int | None
            The current conversation turn number. Used to compute the recency
            boost. When ``None`` the recency component is effectively zero for
            all candidates.
        recency_weight : float | None
            Override the default recency weight (0.15).

        Returns
        -------
        list[RetrievalResult]
            Up to *top_k* results sorted by descending relevance score.
            May return fewer than *top_k* -- or an empty list -- when not
            enough memories pass *min_score*.
        """
        if self.store.active_count() == 0:
            logger.info("retrieve: store is empty, returning []")
            return []

        w_recency = recency_weight if recency_weight is not None else self.WEIGHT_RECENCY

        # 1. Vector search
        vec_results = self.store.search_vector(query, top_k=top_k * 3)
        logger.debug(
            "vector search returned %d candidates for query=%r",
            len(vec_results), query,
        )

        # 2. FTS keyword search
        fts_results = self.store.search_fts(query, top_k=top_k * 3)
        logger.debug(
            "FTS search returned %d candidates for query=%r",
            len(fts_results), query,
        )

        # 3. RRF merge
        rrf_scores: dict[str, float] = {}
        vec_distances: dict[str, float] = {}

        for rank, (mem_id, distance) in enumerate(vec_results):
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            vec_distances[mem_id] = distance

        for rank, (rowid, _fts_rank) in enumerate(fts_results):
            mem_id = self.store.rowid_to_memory_id(rowid)
            if mem_id:
                rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        if not rrf_scores:
            logger.info("retrieve: no candidates after RRF merge")
            return []

        # Build candidate objects
        candidates: dict[str, _ScoredCandidate] = {}
        for mem_id, rrf in rrf_scores.items():
            candidates[mem_id] = _ScoredCandidate(
                memory_id=mem_id,
                rrf_score=rrf,
                vector_distance=vec_distances.get(mem_id),
            )

        # Normalize RRF scores to 0-1 range
        max_rrf = max(c.rrf_score for c in candidates.values())
        if max_rrf == 0:
            max_rrf = 1.0

        # 4. Fetch Memory objects and compute final scores
        scored_results: list[tuple[RetrievalResult, dict]] = []

        for cand in candidates.values():
            memory = self.store.get_memory_by_id(cand.memory_id)
            if memory is None or not memory.is_active:
                logger.debug("skipping inactive/missing memory %s", cand.memory_id)
                continue
            cand.memory = memory

            # a) RRF normalized
            rrf_normalized = cand.rrf_score / max_rrf

            # b) Semantic similarity (from L2 distance)
            if cand.vector_distance is not None:
                semantic_sim = 1.0 / (1.0 + cand.vector_distance)
            else:
                semantic_sim = 0.0

            # c) Recency boost
            if current_turn is not None and current_turn > 0 and memory.last_used_turn > 0:
                recency = min(1.0, memory.last_used_turn / current_turn)
            else:
                recency = 0.0

            # d) Confidence weight
            confidence = memory.confidence

            # Combined score
            final_score = (
                self.WEIGHT_RRF * rrf_normalized
                + self.WEIGHT_SEMANTIC * semantic_sim
                + w_recency * recency
                + self.WEIGHT_CONFIDENCE * confidence
            )

            score_details = {
                "memory_id": cand.memory_id,
                "key": memory.key,
                "rrf_raw": cand.rrf_score,
                "rrf_normalized": round(rrf_normalized, 4),
                "semantic_sim": round(semantic_sim, 4),
                "recency": round(recency, 4),
                "confidence": round(confidence, 4),
                "final_score": round(final_score, 4),
            }
            logger.debug("candidate score: %s", score_details)

            scored_results.append((
                RetrievalResult(memory=memory, score=final_score),
                score_details,
            ))

        # 5. Sort by final_score descending
        scored_results.sort(key=lambda pair: pair[0].score, reverse=True)

        # 6. Apply minimum threshold and dynamic top_k
        filtered: list[RetrievalResult] = []
        dropped_details: list[dict] = []

        for result, details in scored_results:
            if result.score >= min_score and len(filtered) < top_k:
                filtered.append(result)
            else:
                dropped_details.append(details)

        logger.info(
            "retrieve: query=%r | candidates=%d | kept=%d | dropped=%d | "
            "min_score=%.4f | top_k=%d",
            query,
            len(scored_results),
            len(filtered),
            len(dropped_details),
            min_score,
            top_k,
        )
        if dropped_details:
            logger.debug("dropped candidates: %s", dropped_details)

        return filtered
