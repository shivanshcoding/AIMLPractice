"""
Embedding Model Manager.

Creates embedders from config and manages their lifecycle.
Supports benchmarking across multiple embedding models.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from config.settings import ModelConfig, get_settings
from src.core.interfaces import BaseEmbedder
from src.core.exceptions import ConfigurationError
from src.embeddings.bge_m3 import BGEM3Embedder
from src.embeddings.hf_embeddings import HuggingFaceEmbedder

logger = structlog.get_logger(__name__)


def create_embedder(config: ModelConfig) -> BaseEmbedder:
    """
    Create an embedder from a ModelConfig.

    Uses BGEM3Embedder for bge-m3, HuggingFaceEmbedder for everything else.
    """
    model_lower = config.model.lower()

    if "bge-m3" in model_lower or "bge_m3" in model_lower:
        return BGEM3Embedder(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            normalize_embeddings=config.normalize_embeddings,
            use_sparse=config.use_sparse,
            use_colbert=config.use_colbert,
            dense_dim=config.dense_dim,
        )
    else:
        return HuggingFaceEmbedder(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            normalize_embeddings=config.normalize_embeddings,
            dense_dim=config.dense_dim,
        )


def get_primary_embedder() -> BaseEmbedder:
    """Get the primary embedding model from settings."""
    settings = get_settings()
    if settings.models is None:
        raise ConfigurationError("Models config not loaded.")
    return create_embedder(settings.models.embedding_model)


def get_all_embedders() -> dict[str, BaseEmbedder]:
    """
    Get all configured embedding models (primary + alternatives).

    Useful for benchmarking comparisons.
    """
    settings = get_settings()
    if settings.models is None:
        raise ConfigurationError("Models config not loaded.")

    embedders: dict[str, BaseEmbedder] = {
        "primary": create_embedder(settings.models.embedding_model),
    }

    for name, config in settings.models.alternative_embeddings.items():
        embedders[name] = create_embedder(config)

    return embedders


class EmbeddingBenchmark:
    """Benchmark utility for comparing embedding models."""

    @staticmethod
    def benchmark_model(
        embedder: BaseEmbedder,
        test_texts: list[str],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Benchmark an embedding model on a set of test texts.

        Returns:
            Dict with latency stats and memory info.
        """
        import psutil

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure embedding latency
        start = time.perf_counter()
        embeddings = embedder.embed(test_texts)
        elapsed = time.perf_counter() - start

        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "model": embedder.model_name,
            "dim": len(embeddings[0]) if embeddings else 0,
            "total_texts": len(test_texts),
            "total_latency_s": round(elapsed, 3),
            "avg_latency_ms": round(elapsed / len(test_texts) * 1000, 2),
            "texts_per_second": round(len(test_texts) / elapsed, 1),
            "memory_delta_mb": round(mem_after - mem_before, 1),
            "memory_total_mb": round(mem_after, 1),
        }
