"""
Retrieval Strategy Selector.

Maps query types to retrieval strategies and provides
a unified interface for strategy-based retrieval.
"""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import get_settings
from src.core.interfaces import BaseRetriever
from src.core.models import QueryType, RetrieverType, RetrievalResult

logger = structlog.get_logger(__name__)


class StrategySelector:
    """Selects and executes the optimal retrieval strategy based on query type."""

    def __init__(
        self,
        retrievers: dict[str, BaseRetriever],
    ) -> None:
        self._retrievers = retrievers

        # Load strategy mapping from config
        settings = get_settings()
        self._strategy_map: dict[str, str] = {}
        if settings.retrieval_config and settings.retrieval_config.retrieval.strategy_map:
            for qt, config in settings.retrieval_config.retrieval.strategy_map.items():
                self._strategy_map[qt] = config.retriever

    def select_retriever(self, query_type: QueryType) -> BaseRetriever:
        """Select the appropriate retriever for a query type."""
        # Look up in config-driven strategy map
        retriever_name = self._strategy_map.get(query_type.value, "hybrid")

        if retriever_name in self._retrievers:
            return self._retrievers[retriever_name]

        # Fallback to hybrid if available, then dense, then first available
        for fallback in ["hybrid", "dense", "sparse"]:
            if fallback in self._retrievers:
                logger.warning(
                    "retriever_fallback",
                    requested=retriever_name,
                    using=fallback,
                )
                return self._retrievers[fallback]

        raise ValueError(f"No retrievers available. Registered: {list(self._retrievers.keys())}")

    def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.FACTUAL,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Select strategy and execute retrieval."""
        retriever = self.select_retriever(query_type)
        logger.info(
            "strategy_selected",
            query_type=query_type.value,
            retriever=retriever.retriever_name,
        )
        return retriever.retrieve(query=query, top_k=top_k, filters=filters)
