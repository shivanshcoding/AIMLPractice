"""
BM25 Sparse Index.

Lightweight BM25 index for sparse retrieval (keyword matching).
Complements dense retrieval in the hybrid pipeline.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from src.core.models import Chunk, RetrievalResult

logger = structlog.get_logger(__name__)


class BM25Store:
    """
    BM25 sparse retrieval index.

    Maintains an in-memory BM25 index over chunk content.
    Supports persistence to disk.
    """

    def __init__(
        self,
        persist_path: str = "data/bm25_index.pkl",
        lowercase: bool = True,
        remove_stopwords: bool = True,
        **kwargs,
    ) -> None:
        self._persist_path = Path(persist_path)
        self._lowercase = lowercase
        self._remove_stopwords = remove_stopwords
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

        # Basic English stopwords
        self._stopwords = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "it", "its", "this", "that", "these", "those", "be", "been", "has",
            "have", "had", "do", "does", "did", "will", "would", "could", "should",
            "not", "no", "so", "if", "than", "then", "can", "may", "just",
        }

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        if self._lowercase:
            text = text.lower()
        # Simple word tokenization
        tokens = text.split()
        # Remove punctuation
        tokens = [t.strip(".,!?;:\"'()[]{}") for t in tokens]
        # Remove stopwords
        if self._remove_stopwords:
            tokens = [t for t in tokens if t and t not in self._stopwords]
        return [t for t in tokens if t]

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("bm25_index_built", num_chunks=len(chunks))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to existing index (rebuilds)."""
        self._chunks.extend(chunks)
        self._tokenized_corpus.extend([self._tokenize(c.content) for c in chunks])
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("bm25_index_updated", total_chunks=len(self._chunks))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search the BM25 index."""
        if self._bm25 is None or not self._chunks:
            return []

        start = time.perf_counter()
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Apply metadata filters if specified
        if filters:
            for i, chunk in enumerate(self._chunks):
                for field, value in filters.items():
                    chunk_val = getattr(chunk.metadata, field, None)
                    if chunk_val is None:
                        scores[i] = 0.0
                        continue
                    if isinstance(value, list):
                        if chunk_val not in value and (
                            not isinstance(chunk_val, list)
                            or not set(value) & set(chunk_val)
                        ):
                            scores[i] = 0.0
                    elif chunk_val != value:
                        scores[i] = 0.0

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    RetrievalResult(
                        chunk=self._chunks[idx],
                        score=float(scores[idx]),
                        retriever_name="bm25",
                        latency_ms=elapsed_ms,
                        sparse_score=float(scores[idx]),
                    )
                )

        return results

    def save(self) -> None:
        """Persist the BM25 index to disk."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(self._persist_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("bm25_index_saved", path=str(self._persist_path))

    def load(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if not self._persist_path.exists():
            return False
        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info("bm25_index_loaded", num_chunks=len(self._chunks))
            return True
        except Exception as e:
            logger.error("bm25_load_failed", error=str(e))
            return False

    @property
    def size(self) -> int:
        return len(self._chunks)
