"""
Cross-Encoder Reranker.

Generic cross-encoder reranker using sentence-transformers.
Supports any HuggingFace cross-encoder model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from src.core.interfaces import BaseReranker
from src.core.models import RetrievalResult
from src.core.exceptions import RerankerError

logger = structlog.get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = "cuda",
        batch_size: int = 128,
        max_length: int = 512,
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
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name, device=self._device, max_length=self._max_length
            )
            logger.info("cross_encoder_loaded", model=self._model_name)
        return self._model

    @property
    def reranker_name(self) -> str:
        return f"cross_encoder:{self._model_name}"

    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not documents:
            return []

        try:
            start = time.perf_counter()
            model = self._load_model()

            pairs = [[query, doc.chunk.content] for doc in documents]
            scores = model.predict(pairs, batch_size=self._batch_size)

            if self._normalize:
                scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
                scores = scores.tolist()

            elapsed_ms = (time.perf_counter() - start) * 1000

            for i, doc in enumerate(documents):
                doc.reranker_score = float(scores[i]) if i < len(scores) else 0.0

            reranked = sorted(documents, key=lambda d: d.reranker_score or 0.0, reverse=True)
            reranked = reranked[:top_k]

            for doc in reranked:
                doc.score = doc.reranker_score or doc.score
                doc.latency_ms = elapsed_ms

            return reranked

        except Exception as e:
            raise RerankerError(f"Cross-encoder reranking failed: {e}") from e
