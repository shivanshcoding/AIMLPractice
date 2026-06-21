"""
LangGraph Retrieval Agent State Schema.

Defines the typed state that flows through every node in the retrieval graph.
Uses TypedDict with Annotated reducers for proper state merging.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from operator import add

from src.core.models import (
    CompressedContext,
    QueryAnalysis,
    QueryType,
    RetrievalResult,
)


class RetrievalState(TypedDict, total=False):
    """
    State schema for the LangGraph retrieval workflow.

    This state flows through every node and accumulates results.
    Each node reads what it needs and writes its outputs.
    """

    # --- Input ---
    query: str
    metadata_filters: dict[str, Any]

    # --- Query Understanding ---
    query_analysis: QueryAnalysis | None

    # --- Retrieval ---
    retrieval_strategy: str
    retrieved_documents: list[RetrievalResult]
    reranked_documents: list[RetrievalResult]

    # --- Compression ---
    compressed_context: CompressedContext | None

    # --- Confidence & Self-Correction ---
    confidence_score: float
    iteration_count: int
    max_iterations: int

    # --- Performance Tracking ---
    retrieval_latency_ms: float
    reranking_latency_ms: float
    compression_latency_ms: float
    total_latency_ms: float
    total_token_count: int

    # --- Error Tracking ---
    errors: list[str]
