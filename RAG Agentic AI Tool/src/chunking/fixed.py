"""
Fixed-Size Chunking Strategy.

Splits text into fixed-size chunks based on token count with configurable overlap.
Simplest strategy — good baseline for benchmarking.
"""

from __future__ import annotations

from src.core.interfaces import BaseChunker
from src.core.models import Chunk, ChunkingStrategy, Document
from src.chunking.base import BaseChunkerMixin


class FixedChunker(BaseChunker, BaseChunkerMixin):
    """Token-based fixed-size chunking with configurable overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **kwargs,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = self._get_tokenizer()

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into fixed-size token chunks."""
        return self.chunk_text(
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata,
        )

    def chunk_text(self, text: str, document_id: str = "", **kwargs) -> list[Chunk]:
        """Split raw text into fixed-size token chunks."""
        metadata = kwargs.get("metadata", None)
        tokens = self._tokenizer.encode(text)
        chunks: list[Chunk] = []
        step = self._chunk_size - self._chunk_overlap

        i = 0
        chunk_index = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self._chunk_size]
            chunk_text = self._tokenizer.decode(chunk_tokens)

            # Approximate character positions
            start_char = len(self._tokenizer.decode(tokens[:i]))
            end_char = start_char + len(chunk_text)

            chunks.append(
                self.create_chunk(
                    content=chunk_text,
                    document_id=document_id,
                    strategy=ChunkingStrategy.FIXED,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )
            )
            chunk_index += 1
            i += step

        return chunks
