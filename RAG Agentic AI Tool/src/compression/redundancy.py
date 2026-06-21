"""
Redundancy Removal.

Removes near-duplicate chunks using embedding similarity.
"""

from __future__ import annotations

import numpy as np
import structlog

from src.core.interfaces import BaseCompressor, BaseEmbedder
from src.core.models import CompressedContext, RetrievalResult
from src.chunking.base import BaseChunkerMixin

logger = structlog.get_logger(__name__)


class RedundancyRemover(BaseCompressor):
    """Remove near-duplicate chunks based on embedding similarity."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold

    @property
    def compressor_name(self) -> str:
        return "redundancy_removal"

    def compress(
        self,
        query: str,
        documents: list[RetrievalResult],
    ) -> CompressedContext:
        if len(documents) <= 1:
            total_tokens = sum(d.chunk.token_count for d in documents)
            return CompressedContext(
                documents=documents,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                compression_ratio=1.0,
                compression_method=self.compressor_name,
            )

        # Compute embeddings
        texts = [d.chunk.content for d in documents]
        embeddings = self._embedder.embed(texts)
        embeddings_np = np.array(embeddings)

        # Greedy deduplication
        keep_indices = [0]
        for i in range(1, len(embeddings_np)):
            is_duplicate = False
            for j in keep_indices:
                sim = np.dot(embeddings_np[i], embeddings_np[j]) / (
                    np.linalg.norm(embeddings_np[i]) * np.linalg.norm(embeddings_np[j]) + 1e-8
                )
                if sim > self._threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)

        # Filter
        deduplicated = [documents[i] for i in keep_indices]
        original_tokens = sum(d.chunk.token_count for d in documents)
        compressed_tokens = sum(d.chunk.token_count for d in deduplicated)

        logger.info(
            "redundancy_removed",
            original=len(documents),
            deduplicated=len(deduplicated),
            removed=len(documents) - len(deduplicated),
            token_savings=original_tokens - compressed_tokens,
        )

        return CompressedContext(
            documents=deduplicated,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            compression_method=self.compressor_name,
        )
