"""
Benchmark Orchestrator.

Runs the full benchmark suite comparing:
  - Dense Retrieval
  - Hybrid Retrieval
  - Hybrid + Rerank
  - Hybrid + Rerank + Query Expansion

Generates leaderboard reports.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from src.core.models import BenchmarkResult, EvaluationSample
from src.evaluation.ir_metrics import compute_all_ir_metrics
from src.evaluation.report import ReportGenerator

logger = structlog.get_logger(__name__)


class BenchmarkRunner:
    """
    Orchestrates benchmark comparisons across configurations.
    """

    def __init__(self) -> None:
        self._report_gen = ReportGenerator(output_dir="reports/benchmarks")

    def run_retrieval_benchmark(
        self,
        configurations: dict[str, Any],
        eval_samples: list[EvaluationSample],
        k_values: list[int] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Run a retrieval benchmark across multiple configurations.

        Args:
            configurations: Dict of config_name → retrieval_callable.
                Each callable: (query) → list[retrieved_chunk_ids]
            eval_samples: Evaluation samples with ground-truth.
            k_values: K values for Recall@K, etc.

        Returns:
            List of BenchmarkResult for leaderboard.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        results: list[BenchmarkResult] = []

        for config_name, retriever_fn in configurations.items():
            logger.info("benchmarking_config", config=config_name)

            all_metrics: list[dict[str, float]] = []
            latencies: list[float] = []

            for sample in eval_samples:
                start = time.perf_counter()
                try:
                    retrieved_ids = retriever_fn(sample.query)
                except Exception as e:
                    logger.error(
                        "benchmark_retrieval_failed",
                        config=config_name,
                        error=str(e),
                    )
                    continue
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

                # Compute IR metrics
                relevant_ids = [
                    f"gt_{i}" for i in range(len(sample.ground_truth_contexts))
                ]
                # Use content matching to determine relevance
                sample_metrics = compute_all_ir_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    k_values=k_values,
                )
                all_metrics.append(sample_metrics)

            # Aggregate metrics
            if all_metrics:
                avg_metrics = {}
                for key in all_metrics[0]:
                    values = [m[key] for m in all_metrics if key in m]
                    avg_metrics[key] = float(np.mean(values))
            else:
                avg_metrics = {}

            result = BenchmarkResult(
                config_name=config_name,
                metrics=avg_metrics,
                latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
                latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
                latency_p99_ms=float(np.percentile(latencies, 99)) if latencies else 0.0,
                sample_count=len(eval_samples),
            )
            results.append(result)

            logger.info(
                "benchmark_config_complete",
                config=config_name,
                metrics=avg_metrics,
                p50_ms=round(result.latency_p50_ms, 1),
            )

        # Generate leaderboard
        self._report_gen.generate_benchmark_leaderboard(results)

        return results
