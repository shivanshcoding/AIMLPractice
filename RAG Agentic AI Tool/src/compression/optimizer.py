"""
Long-Context Optimization.

Implements:
  1. Lost-in-the-Middle Mitigation
  2. Context Reordering
  3. Context Packing
"""

from __future__ import annotations

import structlog

from src.core.models import RetrievalResult

logger = structlog.get_logger(__name__)


class ContextOptimizer:
    """Optimizes context ordering and packing for LLM consumption."""

    @staticmethod
    def lost_in_middle_reorder(documents: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        Lost-in-the-Middle Mitigation.

        LLMs attend more to the beginning and end of the context window.
        Places most relevant documents at the start and end,
        less relevant in the middle.

        Original order (by score): [1, 2, 3, 4, 5]
        Reordered:                 [1, 3, 5, 4, 2]
        """
        if len(documents) <= 2:
            return documents

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        n = len(sorted_docs)

        reordered = []
        left = []
        right = []

        for i, doc in enumerate(sorted_docs):
            if i % 2 == 0:
                left.append(doc)
            else:
                right.append(doc)

        reordered = left + list(reversed(right))

        logger.debug("lost_in_middle_reordered", count=len(reordered))
        return reordered

    @staticmethod
    def context_packing(
        documents: list[RetrievalResult],
        max_tokens: int = 4096,
    ) -> list[RetrievalResult]:
        """
        Context Packing.

        Greedily selects documents to maximize context window utilization
        without exceeding the token budget.
        """
        packed: list[RetrievalResult] = []
        current_tokens = 0

        # Sort by score (highest first) to prioritize quality
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        for doc in sorted_docs:
            if current_tokens + doc.chunk.token_count <= max_tokens:
                packed.append(doc)
                current_tokens += doc.chunk.token_count

        logger.debug(
            "context_packed",
            total_docs=len(documents),
            packed_docs=len(packed),
            tokens_used=current_tokens,
            max_tokens=max_tokens,
        )
        return packed

    @staticmethod
    def optimize(
        documents: list[RetrievalResult],
        max_tokens: int = 4096,
        apply_lost_in_middle: bool = True,
        apply_packing: bool = True,
    ) -> list[RetrievalResult]:
        """Apply all optimizations in sequence."""
        result = documents

        if apply_packing:
            result = ContextOptimizer.context_packing(result, max_tokens)

        if apply_lost_in_middle:
            result = ContextOptimizer.lost_in_middle_reorder(result)

        return result
