"""
Hybrid Retriever.

Combines dense and sparse retrieval with RRF or weighted fusion.
"""

from __future__ import annotations

import time
from typing import Any

from src.core.interfaces import BaseEmbedder, BaseRetriever
from src.core.models import RetrievalResult
from src.indexing.bm25_store import BM25Store
from src.indexing.hybrid import HybridFusion
from src.indexing.qdrant_store import QdrantStore


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense + sparse with fusion."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        qdrant_store: QdrantStore,
        bm25_store: BM25Store,
        fusion: HybridFusion | None = None,
    ) -> None:
        self._embedder = embedder
        self._qdrant = qdrant_store
        self._bm25 = bm25_store
        self._fusion = fusion or HybridFusion.from_config()

    @property
    def retriever_name(self) -> str:
        return "hybrid"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        start = time.perf_counter()

        # Dense retrieval
        query_vector = self._embedder.embed_query(query)
        dense_results = self._qdrant.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        # Sparse retrieval
        sparse_results = self._bm25.search(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        # Fuse results
        fused = self._fusion.fuse(dense_results, sparse_results, top_k)

        elapsed_ms = (time.perf_counter() - start) * 1000
        for r in fused:
            r.latency_ms = elapsed_ms

        return fused
