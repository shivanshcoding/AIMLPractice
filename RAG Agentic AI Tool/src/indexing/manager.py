"""
Index Lifecycle Manager.

Orchestrates the full indexing pipeline:
  Document → Chunk → Embed → Upsert to Qdrant + BM25
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from src.core.interfaces import BaseEmbedder
from src.core.models import Chunk, Document
from src.embeddings.bge_m3 import BGEM3Embedder
from src.indexing.qdrant_store import QdrantStore
from src.indexing.bm25_store import BM25Store

logger = structlog.get_logger(__name__)


class IndexManager:
    """
    Manages the full indexing lifecycle.

    Coordinates between embedder, Qdrant store, and BM25 store.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        qdrant_store: QdrantStore,
        bm25_store: BM25Store,
    ) -> None:
        self._embedder = embedder
        self._qdrant = qdrant_store
        self._bm25 = bm25_store

    def initialize(self, recreate: bool = False) -> None:
        """Initialize stores (create collections, etc.)."""
        self._qdrant.create_collection(
            dense_dim=self._embedder.dense_dim,
            recreate=recreate,
        )
        logger.info("index_manager_initialized")

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """
        Index chunks into both Qdrant and BM25.

        Returns stats about the indexing operation.
        """
        start = time.perf_counter()

        # Extract text content
        texts = [c.content for c in chunks]

        # Generate embeddings (dense + sparse if available)
        embed_start = time.perf_counter()
        if isinstance(self._embedder, BGEM3Embedder):
            dense_vectors, sparse_vectors = self._embedder.embed_dense_and_sparse(texts)
        else:
            dense_vectors = self._embedder.embed(texts)
            sparse_vectors = None
        embed_time = time.perf_counter() - embed_start

        # Upsert to Qdrant
        qdrant_start = time.perf_counter()
        num_upserted = self._qdrant.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors if sparse_vectors else None,
            batch_size=batch_size,
        )
        qdrant_time = time.perf_counter() - qdrant_start

        # Build BM25 index
        bm25_start = time.perf_counter()
        self._bm25.add_chunks(chunks)
        bm25_time = time.perf_counter() - bm25_start

        total_time = time.perf_counter() - start

        stats = {
            "chunks_indexed": num_upserted,
            "embedding_time_s": round(embed_time, 3),
            "qdrant_upsert_time_s": round(qdrant_time, 3),
            "bm25_index_time_s": round(bm25_time, 3),
            "total_time_s": round(total_time, 3),
            "has_sparse_vectors": sparse_vectors is not None and len(sparse_vectors) > 0,
        }

        logger.info("indexing_complete", **stats)
        return stats

    def get_stats(self) -> dict[str, Any]:
        """Get stats from both stores."""
        return {
            "qdrant": self._qdrant.get_collection_info(),
            "bm25": {"size": self._bm25.size},
        }

    def save_bm25(self) -> None:
        """Persist BM25 index."""
        self._bm25.save()

    def load_bm25(self) -> bool:
        """Load BM25 index from disk."""
        return self._bm25.load()
