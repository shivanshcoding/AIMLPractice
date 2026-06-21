"""
Generic HuggingFace / SentenceTransformers Embedding Wrapper.

Supports any HuggingFace embedding model (BGE-large, E5, Jina, Nomic)
through the unified SentenceTransformers interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from src.core.interfaces import BaseEmbedder
from src.core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)


class HuggingFaceEmbedder(BaseEmbedder):
    """Generic SentenceTransformers embedding wrapper."""

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        dense_dim: int = 1024,
        **kwargs,
    ) -> None:
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize_embeddings
        self._dense_dim = dense_dim
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name, device=self._device
                )
                # Update dense_dim from model
                self._dense_dim = self._model.get_sentence_embedding_dimension()
                logger.info(
                    "hf_embedder_loaded",
                    model=self._model_name,
                    dim=self._dense_dim,
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load model {self._model_name}: {e}") from e
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dense_dim(self) -> int:
        return self._dense_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings."""
        try:
            model = self._load_model()

            # E5 models require "query: " or "passage: " prefix
            if "e5" in self._model_name.lower():
                texts = [f"passage: {t}" for t in texts]

            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            return [e.tolist() for e in embeddings]

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"HF embedding failed: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Generate a dense embedding for a query."""
        model = self._load_model()

        # Handle model-specific query prefixes
        text = query
        if "e5" in self._model_name.lower():
            text = f"query: {query}"
        elif "bge" in self._model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {query}"

        embeddings = model.encode(
            [text],
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embeddings[0].tolist()
