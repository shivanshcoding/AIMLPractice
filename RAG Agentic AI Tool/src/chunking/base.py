"""
Base chunker with shared utilities for all chunking strategies.
"""

from __future__ import annotations

import tiktoken
from typing import Any

from src.core.interfaces import BaseChunker
from src.core.models import Chunk, ChunkingStrategy, Document, DocumentMetadata


class BaseChunkerMixin:
    """Shared utilities for all chunkers."""

    _tokenizer: tiktoken.Encoding | None = None

    @classmethod
    def _get_tokenizer(cls) -> tiktoken.Encoding:
        """Get a cached tokenizer for token counting."""
        if cls._tokenizer is None:
            cls._tokenizer = tiktoken.get_encoding("cl100k_base")
        return cls._tokenizer

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text using tiktoken."""
        tokenizer = BaseChunkerMixin._get_tokenizer()
        return len(tokenizer.encode(text))

    @staticmethod
    def create_chunk(
        content: str,
        document_id: str,
        strategy: ChunkingStrategy,
        chunk_index: int,
        start_char: int = 0,
        end_char: int = 0,
        metadata: DocumentMetadata | None = None,
        parent_chunk_id: str | None = None,
        section_title: str = "",
        section_hierarchy: list[str] | None = None,
    ) -> Chunk:
        """Create a Chunk with computed token count."""
        return Chunk(
            document_id=document_id,
            content=content,
            chunking_strategy=strategy,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            token_count=BaseChunkerMixin.count_tokens(content),
            metadata=metadata or DocumentMetadata(),
            parent_chunk_id=parent_chunk_id,
            section_title=section_title,
            section_hierarchy=section_hierarchy or [],
        )
