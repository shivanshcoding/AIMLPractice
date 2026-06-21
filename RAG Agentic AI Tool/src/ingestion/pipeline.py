"""
Ingestion Pipeline Orchestrator.

Orchestrates the full ingestion flow:
  Load Documents → Analyze Structure → Chunk → Embed → Index
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.core.interfaces import BaseEmbedder
from src.core.models import ChunkingStrategy, Document
from src.chunking.selector import ChunkingSelector
from src.indexing.manager import IndexManager
from src.ingestion.loader import DocumentLoader

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline orchestrator.

    Coordinates document loading, chunking, embedding, and indexing.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        chunking_selector: ChunkingSelector,
        loader: DocumentLoader | None = None,
    ) -> None:
        self._index_manager = index_manager
        self._chunking_selector = chunking_selector
        self._loader = loader or DocumentLoader()

    def ingest_file(
        self,
        file_path: str,
        chunking_strategy: str = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest a single file."""
        doc = self._loader.load_file(file_path, metadata=metadata)
        return self._ingest_document(doc, chunking_strategy)

    def ingest_directory(
        self,
        dir_path: str,
        chunking_strategy: str = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest all supported files from a directory."""
        docs = self._loader.load_directory(dir_path, metadata=metadata)
        total_stats: dict[str, Any] = {
            "documents_processed": 0,
            "total_chunks": 0,
            "total_time_s": 0.0,
        }

        for doc in docs:
            stats = self._ingest_document(doc, chunking_strategy)
            total_stats["documents_processed"] += 1
            total_stats["total_chunks"] += stats.get("chunks_indexed", 0)
            total_stats["total_time_s"] += stats.get("total_time_s", 0.0)

        logger.info("directory_ingestion_complete", **total_stats)
        return total_stats

    def ingest_text(
        self,
        text: str,
        chunking_strategy: str = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest raw text."""
        doc = self._loader.load_text(text, metadata=metadata)
        return self._ingest_document(doc, chunking_strategy)

    def _ingest_document(
        self, document: Document, chunking_strategy: str
    ) -> dict[str, Any]:
        """Internal: chunk and index a single document."""
        # Chunk
        strategy = ChunkingStrategy(chunking_strategy)
        chunks = self._chunking_selector.chunk(document, strategy=strategy)

        logger.info(
            "document_chunked",
            document_id=document.document_id,
            strategy=chunking_strategy,
            chunks=len(chunks),
        )

        # Index
        stats = self._index_manager.index_chunks(chunks)
        return stats
