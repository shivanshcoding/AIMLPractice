"""
Semantic Chunking Strategy.

Uses embedding similarity between adjacent sentences to detect
natural topic boundaries. Chunks at points where the semantic
meaning shifts significantly.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

from src.core.interfaces import BaseChunker, BaseEmbedder
from src.core.models import Chunk, ChunkingStrategy, Document
from src.chunking.base import BaseChunkerMixin

logger = structlog.get_logger(__name__)


class SemanticChunker(BaseChunker, BaseChunkerMixin):
    """
    Embedding-based semantic chunking.

    Splits text into sentences, computes embeddings, and identifies
    breakpoints where cosine similarity between adjacent sentences
    drops below a threshold.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold: float = 95,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        **kwargs,
    ) -> None:
        self._embedder = embedder
        self._threshold_type = breakpoint_threshold_type
        self._threshold = breakpoint_threshold
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Split on sentence boundaries while preserving the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarities(self, sentences: list[str]) -> list[float]:
        """Compute cosine similarities between adjacent sentence embeddings."""
        if len(sentences) < 2:
            return []

        embeddings = self._embedder.embed(sentences)
        similarities = []

        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            # Cosine similarity
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append(float(sim))

        return similarities

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """Find indices where similarity drops below threshold."""
        if not similarities:
            return []

        diffs = []
        for i in range(len(similarities)):
            diffs.append(1.0 - similarities[i])

        # Determine threshold based on type
        if self._threshold_type == "percentile":
            threshold = float(np.percentile(diffs, self._threshold))
        elif self._threshold_type == "standard_deviation":
            mean = float(np.mean(diffs))
            std = float(np.std(diffs))
            threshold = mean + self._threshold * std
        elif self._threshold_type == "interquartile":
            q1 = float(np.percentile(diffs, 25))
            q3 = float(np.percentile(diffs, 75))
            iqr = q3 - q1
            threshold = q3 + self._threshold * iqr
        else:
            threshold = float(np.percentile(diffs, 95))

        breakpoints = [i for i, d in enumerate(diffs) if d >= threshold]
        return breakpoints

    def chunk(self, document: Document) -> list[Chunk]:
        return self.chunk_text(
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata,
        )

    def chunk_text(self, text: str, document_id: str = "", **kwargs) -> list[Chunk]:
        metadata = kwargs.get("metadata", None)
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [
                self.create_chunk(
                    content=text,
                    document_id=document_id,
                    strategy=ChunkingStrategy.SEMANTIC,
                    chunk_index=0,
                    metadata=metadata,
                )
            ]

        # Compute similarities and find breakpoints
        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)

        # Group sentences into chunks at breakpoints
        chunks: list[Chunk] = []
        current_group: list[str] = []
        chunk_index = 0
        current_pos = 0

        for i, sentence in enumerate(sentences):
            current_group.append(sentence)

            is_breakpoint = i in breakpoints
            group_text = " ".join(current_group)
            is_too_large = len(group_text) > self._max_chunk_size

            if (is_breakpoint or is_too_large) and len(group_text) >= self._min_chunk_size:
                start_char = text.find(current_group[0], current_pos)
                if start_char == -1:
                    start_char = current_pos

                chunks.append(
                    self.create_chunk(
                        content=group_text,
                        document_id=document_id,
                        strategy=ChunkingStrategy.SEMANTIC,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(group_text),
                        metadata=metadata,
                    )
                )
                current_pos = start_char + len(group_text)
                chunk_index += 1
                current_group = []

        # Handle remaining sentences
        if current_group:
            group_text = " ".join(current_group)
            start_char = text.find(current_group[0], current_pos)
            if start_char == -1:
                start_char = current_pos

            chunks.append(
                self.create_chunk(
                    content=group_text,
                    document_id=document_id,
                    strategy=ChunkingStrategy.SEMANTIC,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(group_text),
                    metadata=metadata,
                )
            )

        logger.info(
            "semantic_chunking_complete",
            sentences=len(sentences),
            breakpoints=len(breakpoints),
            chunks=len(chunks),
        )
        return chunks
