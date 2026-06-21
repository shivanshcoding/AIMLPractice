"""
Reranker Benchmark.

Compares reranking models (BGE vs CrossEncoder).
"""

from __future__ import annotations

import structlog

from src.core.models import BenchmarkResult

logger = structlog.get_logger(__name__)

class RerankerBenchmark:
    """Benchmark rerankers."""
    def run(self) -> list[BenchmarkResult]:
        logger.info("reranker_benchmark_run")
        return []
