"""
Embedding Benchmark.

Compares embedding models for retrieval accuracy and latency.
"""

from __future__ import annotations

import structlog

from src.core.models import BenchmarkResult

logger = structlog.get_logger(__name__)

class EmbeddingBenchmark:
    """Benchmark embedding models."""
    def run(self) -> list[BenchmarkResult]:
        logger.info("embedding_benchmark_run")
        return []
