"""
BGE-M3 Embedding Model.

Multi-Functionality: dense + sparse + ColBERT in a single forward pass.
Multi-Linguality: 100+ languages.
Multi-Granularity: up to 8192 tokens.

This is the primary embedding model for the RAG engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from src.core.interfaces import BaseEmbedder
from src.core.models import SparseVector
from src.core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)


class BGEM3Embedder(BaseEmbedder):
    """
    BGE-M3 embedder using FlagEmbedding library.

    Generates dense (1024d) and sparse vectors in a single forward pass.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        use_sparse: bool = True,
        use_colbert: bool = False,
        dense_dim: int = 1024,
        **kwargs,
    ) -> None:
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize_embeddings
        self._use_sparse = use_sparse
        self._use_colbert = use_colbert
        self._dense_dim = dense_dim
        self._model = None

        logger.info(
            "bge_m3_embedder_initializing",
            model=model,
            device=device,
            use_sparse=use_sparse,
        )

    def _load_model(self) -> Any:
        """Lazy-load the BGE-M3 model on first use."""
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    self._model_name,
                    use_fp16=self._device == "cuda",
                )
                logger.info("bge_m3_model_loaded", model=self._model_name)
            except ImportError:
                logger.warning(
                    "FlagEmbedding not installed, falling back to sentence-transformers"
                )
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device=self._device)
                logger.info("fallback_model_loaded", model=self._model_name)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dense_dim(self) -> int:
        return self._dense_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
        try:
            model = self._load_model()

            # Check if it's a BGEM3FlagModel or SentenceTransformer
            if hasattr(model, "encode"):
                # BGEM3FlagModel path
                if hasattr(model, "model_name_or_path"):
                    # It's a SentenceTransformer fallback
                    embeddings = model.encode(
                        texts,
                        batch_size=self._batch_size,
                        normalize_embeddings=self._normalize,
                        show_progress_bar=False,
                    )
                    return [e.tolist() for e in embeddings]
                else:
                    # BGEM3FlagModel
                    output = model.encode(
                        texts,
                        batch_size=self._batch_size,
                        max_length=self._max_length,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )
                    return [e.tolist() for e in output["dense_vecs"]]

            raise EmbeddingError("Unknown model type loaded")

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(
                f"BGE-M3 embedding failed: {e}",
                details={"model": self._model_name, "batch_size": len(texts)},
            ) from e

    def embed_query(self, query: str) -> list[float]:
        """Generate a dense embedding for a single query."""
        results = self.embed([query])
        return results[0]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse embeddings using BGE-M3's learned sparse component."""
        try:
            model = self._load_model()

            if not self._use_sparse:
                raise EmbeddingError("Sparse embeddings disabled in config.")

            # Check for BGEM3FlagModel
            if not hasattr(model, "encode") or hasattr(model, "model_name_or_path"):
                raise EmbeddingError(
                    "Sparse embeddings require FlagEmbedding BGEM3FlagModel."
                )

            output = model.encode(
                texts,
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            sparse_vectors: list[SparseVector] = []
            for sparse_dict in output["lexical_weights"]:
                indices = list(sparse_dict.keys())
                values = [float(v) for v in sparse_dict.values()]
                sparse_vectors.append(
                    SparseVector(indices=[int(i) for i in indices], values=values)
                )

            return sparse_vectors

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"BGE-M3 sparse embedding failed: {e}") from e

    def embed_dense_and_sparse(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVector]]:
        """
        Generate BOTH dense and sparse embeddings in a single forward pass.

        This is more efficient than calling embed() and embed_sparse() separately.
        """
        try:
            model = self._load_model()

            if not hasattr(model, "encode") or hasattr(model, "model_name_or_path"):
                # Fallback: dense only
                dense = self.embed(texts)
                return dense, []

            output = model.encode(
                texts,
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=self._use_sparse,
                return_colbert_vecs=False,
            )

            dense = [e.tolist() for e in output["dense_vecs"]]
            sparse = []

            if self._use_sparse and "lexical_weights" in output:
                for sparse_dict in output["lexical_weights"]:
                    indices = [int(i) for i in sparse_dict.keys()]
                    values = [float(v) for v in sparse_dict.values()]
                    sparse.append(SparseVector(indices=indices, values=values))

            return dense, sparse

        except Exception as e:
            raise EmbeddingError(f"BGE-M3 dual embedding failed: {e}") from e
