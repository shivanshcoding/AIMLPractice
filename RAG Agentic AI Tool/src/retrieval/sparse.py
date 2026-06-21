"""
BM25 Sparse Retriever.

Performs keyword-based retrieval using BM25 scoring.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import BaseRetriever
from src.core.models import RetrievalResult
from src.indexing.bm25_store import BM25Store


class SparseRetriever(BaseRetriever):
    """BM25 sparse retriever."""

    def __init__(self, bm25_store: BM25Store) -> None:
        self._bm25 = bm25_store

    @property
    def retriever_name(self) -> str:
        return "sparse"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        return self._bm25.search(query=query, top_k=top_k, filters=filters)
