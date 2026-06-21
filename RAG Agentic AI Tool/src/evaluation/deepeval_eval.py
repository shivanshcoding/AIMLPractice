"""
DeepEval Integration.

Pytest-style RAG evaluation for CI/CD integration.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.core.models import EvaluationResult, EvaluationSample

logger = structlog.get_logger(__name__)


class DeepEvalEvaluator:
    """DeepEval integration for pytest-style RAG evaluation."""

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate using DeepEval metrics."""
        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
            )
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=sample.query,
                actual_output=sample.predicted_answer,
                expected_output=sample.ground_truth_answer,
                retrieval_context=sample.retrieved_contexts,
            )

            # Faithfulness
            faithfulness = FaithfulnessMetric(threshold=0.7)
            faithfulness.measure(test_case)

            # Answer Relevancy
            relevancy = AnswerRelevancyMetric(threshold=0.7)
            relevancy.measure(test_case)

            # Context Precision
            precision = ContextualPrecisionMetric(threshold=0.7)
            precision.measure(test_case)

            # Context Recall
            recall = ContextualRecallMetric(threshold=0.7)
            recall.measure(test_case)

            return EvaluationResult(
                faithfulness=faithfulness.score,
                answer_relevancy=relevancy.score,
                context_precision=precision.score,
                context_recall=recall.score,
            )

        except ImportError:
            logger.warning("DeepEval not installed. Install with: pip install deepeval")
            return EvaluationResult()
        except Exception as e:
            logger.error("deepeval_evaluation_failed", error=str(e))
            return EvaluationResult()
