"""
BGE Reranker v2 M3.

Cross-encoder reranker optimized for multilingual retrieval.
Uses FlagEmbedding library for native support.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from src.core.interfaces import BaseReranker
from src.core.models import RetrievalResult
from src.core.exceptions import RerankerError

logger = structlog.get_logger(__name__)


class BGEReranker(BaseReranker):
    """
    BGE Reranker v2 M3 using FlagEmbedding.

    Cross-encoder model that scores (query, document) pairs for relevance.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 1024,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the reranker model."""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(
                    self._model_name,
                    use_fp16=self._device == "cuda",
                )
                logger.info("bge_reranker_loaded", model=self._model_name)
            except ImportError:
                # Fallback to sentence-transformers CrossEncoder
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self._model_name, device=self._device
                )
                logger.info("cross_encoder_fallback_loaded", model=self._model_name)
        return self._model

    @property
    def reranker_name(self) -> str:
        return f"bge_reranker:{self._model_name}"

    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank documents using BGE Reranker."""
        if not documents:
            return []

        try:
            start = time.perf_counter()
            model = self._load_model()

            # Build (query, document) pairs
            pairs = [[query, doc.chunk.content] for doc in documents]

            # Score pairs
            if hasattr(model, "compute_score"):
                # FlagReranker
                scores = model.compute_score(
                    pairs, batch_size=self._batch_size, normalize=self._normalize
                )
                if isinstance(scores, (int, float)):
                    scores = [scores]
            else:
                # CrossEncoder fallback
                scores = model.predict(pairs, batch_size=self._batch_size)
                if self._normalize:
                    import numpy as np
                    scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
                    scores = scores.tolist()

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Attach reranker scores
            for i, doc in enumerate(documents):
                doc.reranker_score = float(scores[i]) if i < len(scores) else 0.0

            # Sort by reranker score descending
            reranked = sorted(documents, key=lambda d: d.reranker_score or 0.0, reverse=True)
            reranked = reranked[:top_k]

            # Update scores to use reranker scores
            for doc in reranked:
                doc.score = doc.reranker_score or doc.score
                doc.latency_ms = elapsed_ms

            logger.info(
                "bge_reranking_complete",
                input_count=len(documents),
                output_count=len(reranked),
                latency_ms=round(elapsed_ms, 1),
            )
            return reranked

        except Exception as e:
            raise RerankerError(f"BGE reranking failed: {e}") from e
