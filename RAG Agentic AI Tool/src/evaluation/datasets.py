"""
Evaluation Dataset Management.

Manages golden datasets for evaluation and generates synthetic test data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.core.models import EvaluationSample

logger = structlog.get_logger(__name__)


class EvaluationDataset:
    """Manages evaluation datasets."""

    def __init__(self, data_dir: str = "data/eval") -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str = "default") -> list[EvaluationSample]:
        """Load an evaluation dataset from disk."""
        path = self._data_dir / f"{name}.json"
        if not path.exists():
            logger.warning("eval_dataset_not_found", path=str(path))
            return []

        with open(path, "r") as f:
            data = json.load(f)

        return [EvaluationSample(**s) for s in data]

    def save(self, samples: list[EvaluationSample], name: str = "default") -> None:
        """Save an evaluation dataset to disk."""
        path = self._data_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump([s.model_dump() for s in samples], f, indent=2)
        logger.info("eval_dataset_saved", path=str(path), count=len(samples))

    @staticmethod
    def create_sample_dataset() -> list[EvaluationSample]:
        """Create a small sample evaluation dataset for testing."""
        return [
            EvaluationSample(
                query="What is retrieval-augmented generation?",
                ground_truth_answer=(
                    "RAG is a technique that enhances LLM responses by retrieving "
                    "relevant documents from a knowledge base and using them as context "
                    "for generation."
                ),
                ground_truth_contexts=[
                    "Retrieval-Augmented Generation (RAG) combines information retrieval "
                    "with language generation to produce more accurate and grounded responses."
                ],
            ),
            EvaluationSample(
                query="How does BM25 scoring work?",
                ground_truth_answer=(
                    "BM25 uses term frequency, inverse document frequency, and document "
                    "length normalization to score document relevance to a query."
                ),
                ground_truth_contexts=[
                    "BM25 (Best Matching 25) is a bag-of-words retrieval function that "
                    "ranks documents based on query terms appearing in each document."
                ],
            ),
            EvaluationSample(
                query="Compare dense and sparse retrieval methods.",
                ground_truth_answer=(
                    "Dense retrieval uses learned embeddings to capture semantic meaning, "
                    "while sparse retrieval relies on exact term matching. Hybrid approaches "
                    "combine both for best results."
                ),
                ground_truth_contexts=[
                    "Dense retrieval uses neural embeddings for semantic search.",
                    "Sparse retrieval uses term-based methods like BM25.",
                ],
            ),
        ]
