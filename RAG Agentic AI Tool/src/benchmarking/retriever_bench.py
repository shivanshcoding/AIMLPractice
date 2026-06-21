"""
Retriever Benchmark.

Compares dense, sparse, and hybrid retrievers.
"""

from __future__ import annotations

import structlog

from src.core.models import BenchmarkResult

logger = structlog.get_logger(__name__)

class RetrieverBenchmark:
    """Benchmark retrievers."""
    def run(self) -> list[BenchmarkResult]:
        logger.info("retriever_benchmark_run")
        return []
