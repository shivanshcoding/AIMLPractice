"""
Pydantic data models for the RAG Retrieval Engine.

These models are the shared language across all components.
Every function accepts and returns these types — never raw dicts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================
class QueryType(str, Enum):
    """Classification of query intent."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    DOCUMENT_AWARE = "document_aware"
    AUTO = "auto"


class RetrieverType(str, Enum):
    """Available retriever types."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QueryTransformType(str, Enum):
    """Available query transformation methods."""

    NONE = "none"
    MULTI_QUERY = "multi_query"
    QUERY_EXPANSION = "query_expansion"
    HYDE = "hyde"
    STEP_BACK = "step_back"


# =============================================================================
# Document & Chunk Models
# =============================================================================
class DocumentMetadata(BaseModel):
    """Rich metadata attached to every document."""

    source: str = ""
    department: str = ""
    date: datetime | None = None
    version: str = ""
    tags: list[str] = Field(default_factory=list)
    access_level: str = "public"
    file_type: str = ""
    file_size_bytes: int = 0
    page_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A source document before chunking."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A chunked segment of a document."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    # Parent-child relationships
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = Field(default_factory=list)
    # Chunking metadata
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    # Source metadata (inherited from document)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    # Section context (from document-aware chunking)
    section_title: str = ""
    section_hierarchy: list[str] = Field(default_factory=list)


class SparseVector(BaseModel):
    """Sparse vector representation (for BM25/learned sparse)."""

    indices: list[int]
    values: list[float]


# =============================================================================
# Retrieval Models
# =============================================================================
class RetrievalResult(BaseModel):
    """A single retrieval result with scoring metadata."""

    chunk: Chunk
    score: float = 0.0
    retriever_name: str = ""
    reranker_score: float | None = None
    latency_ms: float = 0.0
    # Dense and sparse scores (for hybrid fusion debugging)
    dense_score: float | None = None
    sparse_score: float | None = None


class QueryAnalysis(BaseModel):
    """Result of query understanding / classification."""

    original_query: str
    query_type: QueryType = QueryType.UNKNOWN
    classification_confidence: float = 0.0
    transformed_queries: list[str] = Field(default_factory=list)
    query_transform_type: QueryTransformType = QueryTransformType.NONE
    retriever_type: RetrieverType = RetrieverType.HYBRID
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    # HyDE hypothetical document (if applicable)
    hypothetical_document: str | None = None


class CompressedContext(BaseModel):
    """Context after compression and optimization."""

    documents: list[RetrievalResult]
    original_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 0.0
    compression_method: str = ""


class RetrievalContext(BaseModel):
    """Final output of the retrieval agent."""

    query: str
    query_analysis: QueryAnalysis
    documents: list[RetrievalResult]
    compressed_context: CompressedContext | None = None
    confidence_score: float = 0.0
    # Strategy tracking
    retrieval_strategy: str = ""
    retrieval_iterations: int = 1
    # Performance metrics
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    compression_latency_ms: float = 0.0
    total_token_count: int = 0


# =============================================================================
# Document Structure Analysis
# =============================================================================
class DocumentStructure(BaseModel):
    """Analysis of document structure for auto chunking selection."""

    heading_count: int = 0
    heading_density: float = 0.0  # headings per 1000 chars
    avg_paragraph_length: float = 0.0
    table_count: int = 0
    list_count: int = 0
    code_block_count: int = 0
    has_hierarchical_sections: bool = False
    total_characters: int = 0
    total_tokens: int = 0
    recommended_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE


# =============================================================================
# Evaluation Models
# =============================================================================
class EvaluationSample(BaseModel):
    """A single evaluation sample (ground-truth + prediction)."""

    query: str
    ground_truth_answer: str = ""
    ground_truth_contexts: list[str] = Field(default_factory=list)
    predicted_answer: str = ""
    retrieved_contexts: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Evaluation metrics for a single sample or aggregate."""

    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # RAGAS metrics
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    # IR metrics
    mrr: float | None = None
    ndcg: float | None = None
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    # Metadata
    retrieval_strategy: str = ""
    latency_ms: float = 0.0


class BenchmarkResult(BaseModel):
    """Aggregated benchmark comparison result."""

    config_name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    sample_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
