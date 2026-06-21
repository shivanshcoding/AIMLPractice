"""
RAGAS Evaluation Integration.

Wraps RAGAS metrics for RAG pipeline evaluation:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
  - Answer Correctness
"""

from __future__ import annotations

from typing import Any

import structlog

from src.core.models import EvaluationResult, EvaluationSample

logger = structlog.get_logger(__name__)


class RagasEvaluator:
    """RAGAS integration for RAG evaluation."""

    def __init__(self, llm_model: str | None = None) -> None:
        self._llm_model = llm_model
        self._metrics = None

    def _initialize_metrics(self) -> list:
        """Lazy-initialize RAGAS metrics."""
        if self._metrics is None:
            try:
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                )
                self._metrics = [
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                ]
            except ImportError:
                logger.warning("RAGAS not installed. Install with: pip install ragas")
                self._metrics = []
        return self._metrics

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample using RAGAS."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            import pandas as pd

            # Create RAGAS dataset format
            data = {
                "question": [sample.query],
                "answer": [sample.predicted_answer],
                "contexts": [sample.retrieved_contexts],
                "ground_truth": [sample.ground_truth_answer],
            }

            from datasets import Dataset
            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                ],
            )

            return EvaluationResult(
                faithfulness=result.get("faithfulness"),
                answer_relevancy=result.get("answer_relevancy"),
                context_precision=result.get("context_precision"),
                context_recall=result.get("context_recall"),
                answer_correctness=result.get("answer_correctness"),
            )

        except Exception as e:
            logger.error("ragas_evaluation_failed", error=str(e))
            return EvaluationResult()

    def evaluate_batch(
        self, samples: list[EvaluationSample]
    ) -> list[EvaluationResult]:
        """Evaluate a batch of samples."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            from datasets import Dataset

            data = {
                "question": [s.query for s in samples],
                "answer": [s.predicted_answer for s in samples],
                "contexts": [s.retrieved_contexts for s in samples],
                "ground_truth": [s.ground_truth_answer for s in samples],
            }

            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                ],
            )

            # RAGAS returns aggregate scores
            aggregate = EvaluationResult(
                faithfulness=result.get("faithfulness"),
                answer_relevancy=result.get("answer_relevancy"),
                context_precision=result.get("context_precision"),
                context_recall=result.get("context_recall"),
                answer_correctness=result.get("answer_correctness"),
            )

            return [aggregate]

        except Exception as e:
            logger.error("ragas_batch_evaluation_failed", error=str(e))
            return [EvaluationResult()]
