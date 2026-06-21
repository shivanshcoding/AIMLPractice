"""
Recursive Character Chunking Strategy.

Uses LangChain's RecursiveCharacterTextSplitter which tries to keep
semantically related text together by splitting on a hierarchy of separators.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import BaseChunker
from src.core.models import Chunk, ChunkingStrategy, Document
from src.chunking.base import BaseChunkerMixin


class RecursiveChunker(BaseChunker, BaseChunkerMixin):
    """Recursive character-based chunking with separator hierarchy."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        **kwargs,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk(self, document: Document) -> list[Chunk]:
        return self.chunk_text(
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata,
        )

    def chunk_text(self, text: str, document_id: str = "", **kwargs) -> list[Chunk]:
        metadata = kwargs.get("metadata", None)
        raw_chunks = self._splitter.split_text(text)
        chunks: list[Chunk] = []

        current_pos = 0
        for i, chunk_text in enumerate(raw_chunks):
            # Find actual position in source text
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)
            current_pos = start_char + 1  # Allow overlap

            chunks.append(
                self.create_chunk(
                    content=chunk_text,
                    document_id=document_id,
                    strategy=ChunkingStrategy.RECURSIVE,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )
            )

        return chunks
