"""
Custom exception hierarchy for the RAG Retrieval Engine.

All exceptions inherit from RAGEngineError for unified error handling.
"""

from __future__ import annotations


class RAGEngineError(Exception):
    """Base exception for all RAG engine errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(RAGEngineError):
    """Raised when configuration is invalid or missing."""


class ChunkingError(RAGEngineError):
    """Raised when document chunking fails."""


class EmbeddingError(RAGEngineError):
    """Raised when embedding generation fails."""


class RetrievalError(RAGEngineError):
    """Raised when retrieval execution fails."""


class RerankerError(RAGEngineError):
    """Raised when reranking fails."""


class CompressionError(RAGEngineError):
    """Raised when context compression fails."""


class LLMError(RAGEngineError):
    """Raised when LLM invocation fails."""


class IngestionError(RAGEngineError):
    """Raised when document ingestion fails."""


class EvaluationError(RAGEngineError):
    """Raised when evaluation execution fails."""


class IndexingError(RAGEngineError):
    """Raised when vector store operations fail."""
