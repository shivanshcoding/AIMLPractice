"""
Parent-Child Chunking Strategy.

Creates large parent chunks, then subdivides each into smaller child chunks.
Retrieval searches child chunks (more specific), but returns the parent
context (more complete) — best of both worlds.
"""

from __future__ import annotations

import uuid

from src.core.interfaces import BaseChunker
from src.core.models import Chunk, ChunkingStrategy, Document
from src.chunking.base import BaseChunkerMixin
from src.chunking.recursive import RecursiveChunker


class ParentChildChunker(BaseChunker, BaseChunkerMixin):
    """
    Hierarchical parent-child chunking.

    Parent chunks are large context windows.
    Child chunks are smaller, more specific segments within each parent.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 500,
        child_chunk_overlap: int = 50,
        **kwargs,
    ) -> None:
        self._parent_chunker = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        self._child_chunker = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

    @property
    def strategy_name(self) -> str:
        return "parent_child"

    def chunk(self, document: Document) -> list[Chunk]:
        return self.chunk_text(
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata,
        )

    def chunk_text(self, text: str, document_id: str = "", **kwargs) -> list[Chunk]:
        metadata = kwargs.get("metadata", None)
        all_chunks: list[Chunk] = []

        # Step 1: Create parent chunks
        parent_chunks = self._parent_chunker.chunk_text(
            text=text, document_id=document_id, metadata=metadata
        )

        for parent in parent_chunks:
            # Override the strategy and assign a stable parent_id
            parent.chunking_strategy = ChunkingStrategy.PARENT_CHILD

            # Step 2: Create child chunks within each parent
            child_chunks = self._child_chunker.chunk_text(
                text=parent.content, document_id=document_id, metadata=metadata
            )

            child_ids = []
            for child in child_chunks:
                child.chunking_strategy = ChunkingStrategy.PARENT_CHILD
                child.parent_chunk_id = parent.chunk_id
                # Adjust child positions relative to parent
                child.start_char = parent.start_char + child.start_char
                child.end_char = parent.start_char + child.end_char
                child_ids.append(child.chunk_id)
                all_chunks.append(child)

            # Update parent with child references
            parent.child_chunk_ids = child_ids
            all_chunks.append(parent)

        return all_chunks

    def get_parent_chunk(self, chunks: list[Chunk], child_chunk_id: str) -> Chunk | None:
        """Given a child chunk ID, find its parent chunk."""
        child = next((c for c in chunks if c.chunk_id == child_chunk_id), None)
        if child is None or child.parent_chunk_id is None:
            return None
        return next((c for c in chunks if c.chunk_id == child.parent_chunk_id), None)
