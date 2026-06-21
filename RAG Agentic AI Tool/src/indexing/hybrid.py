"""
Hybrid Retrieval — Reciprocal Rank Fusion and Weighted Fusion.

Combines results from dense and sparse retrievers into a unified ranked list.
"""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import get_settings
from src.core.models import RetrievalResult

logger = structlog.get_logger(__name__)


class HybridFusion:
    """
    Fuses results from multiple retrievers using RRF or weighted fusion.
    """

    def __init__(
        self,
        method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        self._method = method
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    @classmethod
    def from_config(cls) -> "HybridFusion":
        """Create from retrieval config."""
        settings = get_settings()
        if settings.retrieval_config:
            fc = settings.retrieval_config.fusion
            return cls(
                method=fc.method,
                rrf_k=fc.rrf_k,
                dense_weight=fc.dense_weight,
                sparse_weight=fc.sparse_weight,
            )
        return cls()

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Fuse dense and sparse results.

        Args:
            dense_results: Results from dense retriever (sorted by score desc).
            sparse_results: Results from sparse retriever (sorted by score desc).
            top_k: Number of results to return.

        Returns:
            Fused and re-scored results.
        """
        if self._method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        elif self._method == "weighted":
            return self._weighted_fusion(dense_results, sparse_results, top_k)
        else:
            logger.warning(f"Unknown fusion method: {self._method}, using RRF")
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF).

        Score = sum(1 / (k + rank)) for each retriever where the document appears.
        """
        # Build chunk_id → (best_result, rrf_score) mapping
        fused: dict[str, tuple[RetrievalResult, float]] = {}

        # Score from dense
        for rank, result in enumerate(dense_results):
            cid = result.chunk.chunk_id
            rrf_score = 1.0 / (self._rrf_k + rank + 1)
            if cid in fused:
                existing_result, existing_score = fused[cid]
                existing_result.dense_score = result.score
                fused[cid] = (existing_result, existing_score + rrf_score)
            else:
                result.dense_score = result.score
                fused[cid] = (result, rrf_score)

        # Score from sparse
        for rank, result in enumerate(sparse_results):
            cid = result.chunk.chunk_id
            rrf_score = 1.0 / (self._rrf_k + rank + 1)
            if cid in fused:
                existing_result, existing_score = fused[cid]
                existing_result.sparse_score = result.score
                fused[cid] = (existing_result, existing_score + rrf_score)
            else:
                result.sparse_score = result.score
                fused[cid] = (result, rrf_score)

        # Sort by RRF score descending
        sorted_results = sorted(fused.values(), key=lambda x: x[1], reverse=True)

        final = []
        for result, rrf_score in sorted_results[:top_k]:
            result.score = rrf_score
            result.retriever_name = "hybrid_rrf"
            final.append(result)

        logger.debug(
            "rrf_fusion_complete",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(final),
        )
        return final

    def _weighted_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Weighted score fusion.

        Normalizes scores from each retriever and combines with weights.
        """
        # Normalize scores to [0, 1]
        def normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
            if not results:
                return results
            max_score = max(r.score for r in results) or 1.0
            min_score = min(r.score for r in results)
            score_range = max_score - min_score or 1.0
            for r in results:
                r.score = (r.score - min_score) / score_range
            return results

        dense_normalized = normalize_scores(list(dense_results))
        sparse_normalized = normalize_scores(list(sparse_results))

        fused: dict[str, RetrievalResult] = {}

        for result in dense_normalized:
            cid = result.chunk.chunk_id
            result.dense_score = result.score
            result.score = result.score * self._dense_weight
            result.retriever_name = "hybrid_weighted"
            fused[cid] = result

        for result in sparse_normalized:
            cid = result.chunk.chunk_id
            weighted_sparse = result.score * self._sparse_weight
            if cid in fused:
                fused[cid].score += weighted_sparse
                fused[cid].sparse_score = result.score
            else:
                result.sparse_score = result.score
                result.score = weighted_sparse
                result.retriever_name = "hybrid_weighted"
                fused[cid] = result

        sorted_results = sorted(fused.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]
