"""
Dense Vector Retriever.

Performs semantic search using dense embeddings in Qdrant.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import BaseEmbedder, BaseRetriever
from src.core.models import RetrievalResult
from src.indexing.qdrant_store import QdrantStore


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using Qdrant."""

    def __init__(self, embedder: BaseEmbedder, qdrant_store: QdrantStore) -> None:
        self._embedder = embedder
        self._qdrant = qdrant_store

    @property
    def retriever_name(self) -> str:
        return "dense"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_vector = self._embedder.embed_query(query)
        results = self._qdrant.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
        for r in results:
            r.dense_score = r.score
        return results
