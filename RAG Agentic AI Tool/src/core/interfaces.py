"""
Abstract base classes (interfaces) for all pluggable components.

Every chunker, embedder, retriever, reranker, compressor, and LLM client
must implement these interfaces. This enables:
  - Dependency injection
  - Runtime component swapping via config
  - Clean testing with mocks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from src.core.models import (
    Chunk,
    CompressedContext,
    Document,
    DocumentStructure,
    QueryAnalysis,
    RetrievalResult,
    SparseVector,
)


# =============================================================================
# Chunker Interface
# =============================================================================
class BaseChunker(ABC):
    """Interface for document chunking strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        ...

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...

    @abstractmethod
    def chunk_text(self, text: str, document_id: str = "") -> list[Chunk]:
        """Split raw text into chunks (convenience method)."""
        ...


# =============================================================================
# Embedder Interface
# =============================================================================
class BaseEmbedder(ABC):
    """Interface for embedding models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def dense_dim(self) -> int:
        """Return the dense vector dimensionality."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate a dense embedding for a single query."""
        ...

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse embeddings (optional — not all models support this)."""
        raise NotImplementedError(
            f"{self.model_name} does not support sparse embeddings."
        )

    def embed_query_sparse(self, query: str) -> SparseVector:
        """Generate a sparse embedding for a single query."""
        results = self.embed_sparse([query])
        return results[0]


# =============================================================================
# Retriever Interface
# =============================================================================
class BaseRetriever(ABC):
    """Interface for retrieval execution."""

    @property
    @abstractmethod
    def retriever_name(self) -> str:
        """Return the retriever identifier."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        ...


# =============================================================================
# Reranker Interface
# =============================================================================
class BaseReranker(ABC):
    """Interface for document reranking."""

    @property
    @abstractmethod
    def reranker_name(self) -> str:
        """Return the reranker identifier."""
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank documents by relevance to the query."""
        ...


# =============================================================================
# Compressor Interface
# =============================================================================
class BaseCompressor(ABC):
    """Interface for context compression."""

    @property
    @abstractmethod
    def compressor_name(self) -> str:
        """Return the compressor identifier."""
        ...

    @abstractmethod
    def compress(
        self,
        query: str,
        documents: list[RetrievalResult],
    ) -> CompressedContext:
        """Compress retrieved documents to reduce token count."""
        ...


# =============================================================================
# LLM Interface
# =============================================================================
class BaseLLM(ABC):
    """Interface for language model invocation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async text generation."""
        ...

    def generate_structured(
        self, prompt: str, schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        """Generate and parse structured output into a Pydantic model."""
        import json

        raw = self.generate(prompt, **kwargs)
        # Attempt to extract JSON from the response
        try:
            # Try parsing directly
            return schema.model_validate_json(raw)
        except Exception:
            # Try extracting JSON from markdown code blocks
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                return schema.model_validate_json(json_str)
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                return schema.model_validate_json(json_str)
            # Try finding JSON object in text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return schema.model_validate_json(raw[start:end])
            raise


# =============================================================================
# Document Analyzer Interface
# =============================================================================
class BaseDocumentAnalyzer(ABC):
    """Interface for document structure analysis."""

    @abstractmethod
    def analyze(self, document: Document) -> DocumentStructure:
        """Analyze document structure for auto chunking selection."""
        ...
