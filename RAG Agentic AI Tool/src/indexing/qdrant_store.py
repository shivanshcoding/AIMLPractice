"""
Qdrant Vector Store Wrapper.

Manages collections, upserts, and searches in Qdrant.
Supports both dense and sparse named vectors in a single collection.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector as QdrantSparseVector,
    SparseVectorParams,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    PayloadSchemaType,
    models,
)

from config.settings import QdrantConfig, get_settings
from src.core.exceptions import IndexingError
from src.core.models import Chunk, RetrievalResult, SparseVector

logger = structlog.get_logger(__name__)


class QdrantStore:
    """
    Qdrant vector store with support for dense + sparse vectors.

    Each document is stored as a point with:
      - Named vector "dense": 1024-dim float vector
      - Named sparse vector "sparse": learned sparse vector (BGE-M3)
      - Payload: chunk metadata for filtering
    """

    def __init__(
        self,
        config: QdrantConfig | None = None,
    ) -> None:
        if config is None:
            settings = get_settings()
            config = settings.retrieval_config.qdrant if settings.retrieval_config else QdrantConfig()

        self._config = config
        self._collection_name = config.collection_name

        # Initialize client directly to local disk for Kaggle environment
        self._client = QdrantClient(path="./qdrant_data")

        logger.info(
            "qdrant_store_initialized",
            mode=config.mode,
            collection=self._collection_name,
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    def create_collection(
        self,
        dense_dim: int = 1024,
        recreate: bool = False,
    ) -> None:
        """Create a collection with dense + sparse vector support."""
        try:
            exists = self._client.collection_exists(self._collection_name)

            if exists and recreate:
                self._client.delete_collection(self._collection_name)
                logger.info("collection_deleted", collection=self._collection_name)

            if not exists or recreate:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        self._config.dense_vector_name: VectorParams(
                            size=dense_dim,
                            distance=Distance.COSINE,
                            hnsw_config=models.HnswConfigDiff(
                                m=self._config.hnsw_config.get("m", 16),
                                ef_construct=self._config.hnsw_config.get("ef_construct", 100),
                                on_disk=self._config.hnsw_config.get("on_disk", False),
                            ),
                        ),
                    },
                    sparse_vectors_config={
                        self._config.sparse_vector_name: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                        ),
                    },
                )

                # Create payload indexes for filterable fields
                settings = get_settings()
                if settings.retrieval_config:
                    for field in settings.retrieval_config.metadata.filterable_fields:
                        try:
                            self._client.create_payload_index(
                                collection_name=self._collection_name,
                                field_name=field,
                                field_schema=PayloadSchemaType.KEYWORD,
                            )
                        except Exception:
                            pass  # Index might already exist

                logger.info(
                    "collection_created",
                    collection=self._collection_name,
                    dense_dim=dense_dim,
                )
        except Exception as e:
            raise IndexingError(f"Failed to create collection: {e}") from e

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert chunks with their vectors into Qdrant.

        Returns the number of points upserted.
        """
        try:
            total = 0
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_dense = dense_vectors[i : i + batch_size]
                batch_sparse = (
                    sparse_vectors[i : i + batch_size] if sparse_vectors else None
                )

                points = []
                for j, chunk in enumerate(batch_chunks):
                    # Build vector dict
                    vector: dict[str, Any] = {
                        self._config.dense_vector_name: batch_dense[j],
                    }

                    # Build payload from chunk metadata
                    payload = {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "chunking_strategy": chunk.chunking_strategy.value,
                        "token_count": chunk.token_count,
                        "section_title": chunk.section_title,
                        "section_hierarchy": chunk.section_hierarchy,
                        "parent_chunk_id": chunk.parent_chunk_id or "",
                        # Metadata fields for filtering
                        "source": chunk.metadata.source,
                        "department": chunk.metadata.department,
                        "date": chunk.metadata.date.isoformat() if chunk.metadata.date else "",
                        "version": chunk.metadata.version,
                        "tags": chunk.metadata.tags,
                        "access_level": chunk.metadata.access_level,
                    }

                    # Add sparse vector if available
                    sparse_dict = {}
                    if batch_sparse and j < len(batch_sparse):
                        sv = batch_sparse[j]
                        sparse_dict = {
                            self._config.sparse_vector_name: QdrantSparseVector(
                                indices=sv.indices,
                                values=sv.values,
                            )
                        }

                    point = PointStruct(
                        id=hash(chunk.chunk_id) % (2**63),  # Qdrant needs int or UUID
                        vector={**vector, **sparse_dict} if sparse_dict else vector,
                        payload=payload,
                    )
                    points.append(point)

                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
                total += len(points)

                logger.debug(
                    "batch_upserted",
                    batch_start=i,
                    batch_size=len(points),
                    total=total,
                )

            logger.info("upsert_complete", total_points=total)
            return total

        except Exception as e:
            raise IndexingError(f"Upsert failed: {e}") from e

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search using dense vectors."""
        try:
            start = time.perf_counter()

            qdrant_filter = self._build_filter(filters) if filters else None

            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                using=self._config.dense_vector_name,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._parse_results(results.points, elapsed_ms, "dense")

        except Exception as e:
            raise IndexingError(f"Dense search failed: {e}") from e

    def search_sparse(
        self,
        query_sparse: SparseVector,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search using sparse vectors."""
        try:
            start = time.perf_counter()

            qdrant_filter = self._build_filter(filters) if filters else None

            results = self._client.query_points(
                collection_name=self._collection_name,
                query=QdrantSparseVector(
                    indices=query_sparse.indices,
                    values=query_sparse.values,
                ),
                using=self._config.sparse_vector_name,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._parse_results(results.points, elapsed_ms, "sparse")

        except Exception as e:
            raise IndexingError(f"Sparse search failed: {e}") from e

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """Build a Qdrant filter from a dict of field conditions."""
        conditions = []
        for field, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(field=field, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(field=field, match=MatchValue(value=value))
                )
        return Filter(must=conditions)

    def _parse_results(
        self,
        points: list,
        latency_ms: float,
        retriever_name: str,
    ) -> list[RetrievalResult]:
        """Parse Qdrant results into RetrievalResult models."""
        results = []
        for point in points:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                content=payload.get("content", ""),
                chunk_index=payload.get("chunk_index", 0),
                token_count=payload.get("token_count", 0),
                section_title=payload.get("section_title", ""),
                section_hierarchy=payload.get("section_hierarchy", []),
                parent_chunk_id=payload.get("parent_chunk_id") or None,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=point.score or 0.0,
                    retriever_name=retriever_name,
                    latency_ms=latency_ms,
                )
            )
        return results

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection stats."""
        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "collection": self._collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception:
            return {"collection": self._collection_name, "status": "not_found"}

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection_name)
        logger.info("collection_deleted", collection=self._collection_name)
